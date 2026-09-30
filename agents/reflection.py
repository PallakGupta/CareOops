"""
Reflection Agent: a separate, critical second look at Documentation + Safety output.

Input state keys:  intake, records, soap_note, safety_check, revision_count
Output state keys: reflection_flags, reflection_history, revision_target,
                   revision_count, status, agent_log

It NEVER edits the note or the verdict. It only returns flags:
    {target_agent, severity, issue, evidence, source}
target_agent is "documentation", "prescription_safety", or None (informational).

Flags come from two layers:
  1. Deterministic rule checks: cheap, always on, catch known failure modes
     such as "note says NKDA but the record lists allergies".
  2. An LLM critique (GPT-4o).

The agent also DECIDES routing: if a flag targets another agent and we have
not revised yet, it sets `revision_target`. graph.py reads that value in its
conditional edge.
"""

import json

import config
from agents.common import LLMError, call_llm_json, log_entry
from agents.drug_knowledge import normalize_drug
from agents.prescription_safety import check_allergies

ROUTABLE_AGENTS = ("documentation", "prescription_safety")
NKDA_PHRASES = ("no known drug allergies", "no known allergies", "nkda", "no drug allergies")

SYSTEM_PROMPT = """You are a clinical QA reviewer. You receive a patient record,
the structured intake, a drafted SOAP note, and a prescription safety check.
Do NOT rewrite anything. Find problems only:
- contradictions between the note and the record (allergies, meds, labs, history)
- claims in the note that are NOT supported by the record or intake (hallucinations)
- safety checks that are missing, or a verdict that contradicts the evidence
Respond with ONLY a JSON object: {"flags": [ {"target_agent": "documentation" |
"prescription_safety" | null, "severity": "high" | "medium" | "info",
"issue": string, "evidence": string quoting the record/note } ]}.
target_agent meaning: "documentation" if the note or plan must be changed
(e.g. it contradicts the record, or keeps a drug the safety check FLAGGED
without justification); "prescription_safety" if the safety check itself is
wrong or incomplete; null for informational remarks that need no rework.
Return {"flags": []} if everything is consistent. Do not flag stylistic issues."""


def rule_based_checks(state: dict) -> list[dict]:
    """Deterministic consistency checks that do not depend on an LLM.

    - Note claims "no known allergies" but the record lists allergies (hallucination).
    - A current medication from the record is missing from the note.
    - Medication was proposed but no safety checks were recorded (missing check).
    - Safety said SAFE although a proposed drug matches an allergy (contradiction).
    """
    flags = []
    records = state.get("records") or {}
    note = state.get("soap_note") or {}
    safety = state.get("safety_check") or {}
    note_text = " ".join(str(note.get(k, "")) for k in ("subjective", "objective", "assessment", "plan")).lower()
    allergies = records.get("allergies", [])

    if allergies and any(p in note_text for p in NKDA_PHRASES):
        flags.append({
            "target_agent": "documentation", "severity": "high",
            "issue": "The SOAP note says the patient has no known drug allergies, but the record lists allergies. "
                     "This claim is not supported by the record.",
            "evidence": "record allergies: " + ", ".join(a["substance"] for a in allergies),
        })

    missing = [m["name"] for m in records.get("current_medications", []) if m["name"].lower() not in note_text]
    if missing:
        flags.append({
            "target_agent": "documentation", "severity": "medium",
            "issue": "The note does not mention current medication(s) from the record: " + ", ".join(missing),
            "evidence": "record current_medications",
        })

    proposed = note.get("proposed_medications") or []
    if proposed and not safety.get("checks_performed"):
        flags.append({
            "target_agent": "prescription_safety", "severity": "high",
            "issue": "Medication was proposed but no safety checks were recorded.",
            "evidence": "proposed: " + ", ".join(m.get("name", "?") for m in proposed),
        })

    if safety.get("verdict") == "SAFE":
        for med in proposed:
            if check_allergies(normalize_drug(med.get("name", "")), allergies):
                flags.append({
                    "target_agent": "prescription_safety", "severity": "high",
                    "issue": f"The safety verdict is SAFE, but {med.get('name')} matches a documented allergy.",
                    "evidence": "record allergies vs proposed_medications",
                })

    return [{**f, "source": "rule"} for f in flags]


def _llm_flags(state: dict) -> list[dict]:
    """Get the GPT-4o critique flags, normalized to the flag schema."""
    payload = {k: state.get(k) for k in ("records", "intake", "soap_note", "safety_check")}
    result = call_llm_json("Reflection", SYSTEM_PROMPT, json.dumps(payload, indent=2))

    raw = result.get("flags")
    if not isinstance(raw, list):
        raise LLMError("Reflection", "model output has no 'flags' list")

    flags = []
    for f in raw:
        if not isinstance(f, dict) or not f.get("issue"):
            continue
        target = f.get("target_agent")
        flags.append({
            "target_agent": target if target in ROUTABLE_AGENTS else None,
            "severity": f.get("severity", "info"),
            "issue": f["issue"],
            "evidence": f.get("evidence", ""),
            "source": "llm",
        })
    return flags


def reflection_agent(state: dict) -> dict:
    """Graph node: collect flags and decide whether to route back ONCE.

    Routing rule (read by graph.route_after_reflection):
      - If any flag targets an agent AND revision_count < MAX_REFLECTION_REVISIONS,
        set revision_target (documentation wins over prescription_safety, because
        re-running documentation automatically re-runs the safety check too).
      - Otherwise set revision_target = None and hand off to the doctor.
    """
    revision_count = state.get("revision_count", 0)
    pass_number = len(state.get("reflection_history") or []) + 1

    flags = rule_based_checks(state) + _llm_flags(state)

    targets = {f["target_agent"] for f in flags if f["target_agent"] in ROUTABLE_AGENTS}
    revision_target = None
    if targets and revision_count < config.MAX_REFLECTION_REVISIONS:
        revision_target = "documentation" if "documentation" in targets else "prescription_safety"
        revision_count += 1

    if revision_target:
        summary = f"Pass {pass_number}: {len(flags)} flag(s). Routing back to {revision_target}"
        status = f"revising_{revision_target}"
    else:
        summary = f"Pass {pass_number}: {len(flags)} flag(s). Handing off to the doctor"
        status = "awaiting_doctor_approval"

    return {
        "reflection_flags": flags,
        "reflection_history": [{"pass": pass_number, "flags": flags, "routed_to": revision_target}],
        "revision_target": revision_target,
        "revision_count": revision_count,
        "status": status,
        "agent_log": [log_entry("Reflection", summary, f"rules + {config.OPENAI_MODEL}")],
    }
