"""
Prescription Safety Agent: checks every proposed medication before a doctor sees it.

Input state keys:  soap_note.proposed_medications, records
Output state keys: safety_check, safety_history (appended), agent_log

Three checks run for each proposed drug:
  1. Allergy     : drug name/class vs. the patient's allergy list (+ cross-reactivity)
  2. Renal (lab) : drug class vs. the patient's latest eGFR
  3. Interaction : drug vs. each current medication, via RxNorm, falling back to
                   the local table in drug_knowledge.py

This agent is deliberately RULE-BASED, not an LLM. A safety verdict has to be
deterministic and explainable: the same inputs always give the same verdict,
and every flag points to the exact record entry that caused it.
"""

import requests

import config
from agents.common import log_entry
from agents.drug_knowledge import (
    CROSS_REACTIVITY, RENAL_RULES, classes_for, local_interactions, normalize_drug,
)

_RXCUI_CACHE: dict[str, str | None] = {}


class RxNormClient:
    """Thin wrapper around the public RxNorm REST API with a per-run circuit breaker.

    Every call uses a 3s timeout and is wrapped in try/except. After the first
    failure, `available` flips to False and later calls in the same safety
    check are skipped. A bad connection costs at most one timeout, not one per
    drug pair.
    """

    def __init__(self) -> None:
        self.available = True
        self.last_error: str | None = None

    def _get(self, path: str, params: dict) -> dict | None:
        """GET {RXNORM_BASE_URL}/{path}. Returns the parsed JSON, or None on any failure."""
        if not self.available:
            return None
        try:
            resp = requests.get(f"{config.RXNORM_BASE_URL}/{path}", params=params,
                                timeout=config.RXNORM_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - demo safety: fall back, never crash
            self.available = False
            self.last_error = f"{path}: {type(exc).__name__} {exc}"[:160]
            return None

    def lookup_rxcui(self, drug_name: str) -> str | None:
        """Normalize a drug name to its RxNorm concept id (RxCUI) via approximate match."""
        if drug_name in _RXCUI_CACHE:
            return _RXCUI_CACHE[drug_name]
        data = self._get("approximateTerm.json", {"term": drug_name, "maxEntries": 1})
        if data is None:
            return None
        candidates = (data.get("approximateGroup") or {}).get("candidate") or []
        rxcui = candidates[0].get("rxcui") if candidates else None
        _RXCUI_CACHE[drug_name] = rxcui
        return rxcui

    def interactions(self, rxcuis: list[str]) -> list[dict] | None:
        """Query the RxNav interaction endpoint for a list of RxCUIs.

        Returns a list of {severity, description}, or None if the call failed.
        (NLM retired this endpoint in Jan 2024 and it now returns 404, so in
        practice this returns None and the local table is used.)
        """
        data = self._get("interaction/list.json", {"rxcuis": " ".join(rxcuis)})
        if data is None:
            return None
        found = []
        for group in data.get("fullInteractionTypeGroup", []):
            for itype in group.get("fullInteractionType", []):
                for pair in itype.get("interactionPair", []):
                    sev = (pair.get("severity") or "").lower()
                    found.append({"severity": "high" if sev in ("high", "n/a", "") else "medium",
                                  "description": pair.get("description", "")})
        return found


def _latest_lab(labs: list[dict], test_name: str) -> dict | None:
    """Return the most recent lab result for `test_name` (case-insensitive), or None."""
    matches = [lab for lab in labs if lab.get("test", "").lower() == test_name.lower()]
    return max(matches, key=lambda lab: lab.get("date", "")) if matches else None


def check_allergies(drug: str, allergies: list[dict]) -> list[dict]:
    """Flag the drug if it, or its class, matches an allergy or a known cross-reactive class."""
    issues = []
    drug_classes = classes_for(drug)
    drug_tokens = [t for t in drug.split("-") if len(t) > 3]
    for allergy in allergies:
        a_class = (allergy.get("allergy_class") or "").lower()
        substance = (allergy.get("substance") or "").lower()
        label = f"{allergy.get('substance')} ({allergy.get('reaction', 'reaction unknown')}, {allergy.get('severity', '?')})"

        if a_class in drug_classes or any(tok in substance for tok in drug_tokens):
            issues.append({"check": "allergy", "severity": "high", "drug": drug,
                           "detail": f"{drug} matches a documented allergy: {label}."})
        elif any(c in drug_classes for c in CROSS_REACTIVITY.get(a_class, [])):
            issues.append({"check": "allergy", "severity": "medium", "drug": drug,
                           "detail": f"{drug} may cross-react with the documented allergy: {label}."})
    return issues


def check_renal(drug: str, labs: list[dict]) -> tuple[list[dict], str]:
    """Compare the drug against renal rules using the patient's latest eGFR.

    Returns (issues, description of the check), so the check itself is always
    recorded, even when nothing is flagged.
    """
    egfr = _latest_lab(labs, "eGFR")
    if egfr is None:
        return [], f"Renal check for {drug}: SKIPPED, no eGFR on file"

    issues = []
    terms = {drug, *classes_for(drug)}
    for rule in RENAL_RULES:
        if rule["applies_to"] in terms and egfr["value"] < rule["egfr_below"]:
            issues.append({"check": "renal", "severity": rule["severity"], "drug": drug,
                           "detail": f"eGFR {egfr['value']} {egfr['unit']} ({egfr['date']}). {rule['message']}"})
    status = "FLAG" if issues else "OK"
    return issues, f"Renal check for {drug}: eGFR {egfr['value']} ({egfr['date']}), {status}"


def check_interactions(drug: str, current_meds: list[dict], rx: RxNormClient) -> tuple[list[dict], str]:
    """Check the drug against every current medication.

    Tries RxNorm first. The local table ALWAYS runs as a safety net: if RxNorm
    answers, its results are added, and if it fails, the local table is the
    whole answer. Returns (issues, source label).
    """
    current = [normalize_drug(m["name"]) for m in current_meds]
    issues: list[dict] = []

    rx_results = None
    rxcuis = [rx.lookup_rxcui(d) for d in [drug, *current]]
    if rx.available and all(rxcuis):
        rx_results = rx.interactions(rxcuis)

    for other in current:
        for hit in local_interactions(drug, other):
            issues.append({"check": "interaction", "severity": hit["severity"], "drug": drug,
                           "detail": f"{drug} + {other}: {hit['description']}", "source": "local"})

    if rx_results:
        seen = {i["detail"] for i in issues}
        for hit in rx_results:
            detail = f"{drug}: {hit['description']}"
            if detail not in seen:
                issues.append({"check": "interaction", "severity": hit["severity"], "drug": drug,
                               "detail": detail, "source": "rxnorm"})

    source = "RxNorm + local rules" if rx_results is not None else "local fallback table"
    return issues, source


def prescription_safety_agent(state: dict) -> dict:
    """Graph node: run all three checks on every proposed medication and give a verdict.

    Verdict is FLAGGED if any high- or medium-severity issue is found, else SAFE.
    `checks_performed` lists every check that ran, so the Reflection Agent (and
    the doctor) can confirm that nothing was skipped.
    """
    records = state.get("records") or {}
    proposed = (state.get("soap_note") or {}).get("proposed_medications") or []

    if not proposed:
        safety = {"verdict": "SAFE", "reasoning": "No medication was proposed, so there was nothing to check.",
                  "issues": [], "checks_performed": [], "medications_checked": [],
                  "interaction_source": "n/a"}
        return {"safety_check": safety, "safety_history": [safety],
                "agent_log": [log_entry("Prescription Safety", "No medication proposed", "rules")]}

    rx = RxNormClient()
    issues: list[dict] = []
    checks: list[str] = []
    meds_checked: list[dict] = []
    sources: set[str] = set()

    for med in proposed:
        drug = normalize_drug(med.get("name", ""))
        rxcui = rx.lookup_rxcui(drug)
        known = bool(classes_for(drug))
        meds_checked.append({"name": med.get("name"), "normalized": drug, "rxcui": rxcui,
                             "in_local_knowledge_base": known})

        allergy_issues = check_allergies(drug, records.get("allergies", []))
        issues += allergy_issues
        checks.append(f"Allergy check for {drug}: {len(records.get('allergies', []))} allergies reviewed, "
                      f"{'FLAG' if allergy_issues else 'OK'}")

        renal_issues, renal_desc = check_renal(drug, records.get("recent_labs", []))
        issues += renal_issues
        checks.append(renal_desc)

        ix_issues, ix_source = check_interactions(drug, records.get("current_medications", []), rx)
        issues += ix_issues
        sources.add(ix_source)
        checks.append(f"Interaction check for {drug} vs {len(records.get('current_medications', []))} "
                      f"current meds via {ix_source}: {'FLAG' if ix_issues else 'OK'}")

        if not known:
            checks.append(f"NOTE: {drug} is not in the local knowledge base, so class-based checks are limited")

    flagged = [i for i in issues if i["severity"] in ("high", "medium")]
    verdict = "FLAGGED" if flagged else "SAFE"
    names = ", ".join(m["normalized"] for m in meds_checked)
    if flagged:
        reasoning = f"{len(flagged)} safety issue(s) found for {names}: " + " | ".join(i["detail"] for i in flagged)
    else:
        reasoning = f"No allergy, renal or interaction concerns found for {names} against this patient's record."

    safety = {
        "verdict": verdict,
        "reasoning": reasoning,
        "issues": issues,
        "checks_performed": checks,
        "medications_checked": meds_checked,
        "interaction_source": ", ".join(sorted(sources)),
        "rxnorm_status": "reachable" if rx.available else f"unavailable ({rx.last_error})",
    }
    summary = f"Verdict: {verdict} for {names} ({len(flagged)} issue(s))"
    source = f"rules + {safety['interaction_source']}"
    return {"safety_check": safety, "safety_history": [safety],
            "agent_log": [log_entry("Prescription Safety", summary, source)]}
