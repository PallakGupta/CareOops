"""
Documentation Agent: drafts a SOAP note from the intake + records.

Input state keys:  intake, records, (on a revision pass) reflection_flags, soap_note
Output state keys: soap_note, previous_soap_note (revision only), agent_log

The note is structured JSON rather than free text, and `proposed_medications`
is a separate machine-readable list. That list is what the Prescription Safety
Agent checks, so it never has to parse drug names out of prose.
"""

import json

import config
from agents.common import call_llm_json, log_entry, require_keys

SYSTEM_PROMPT = """You are a clinical documentation assistant drafting a SOAP note
for a physician to review. Rules:
- Use ONLY facts from the provided intake and patient record. Never invent
  vitals, exam findings, labs, allergies or history.
- In Subjective, list EVERY allergy and EVERY current medication from the
  record (including as-needed ones) exactly as they appear there.
- If no in-person exam happened, say so in Objective.
- In Plan, give the concrete first-line treatment you would recommend,
  including any medication with dose, frequency and duration. Put every
  medication you recommend starting in proposed_medications; a separate safety
  agent and the physician review it before anything is prescribed. Leave
  proposed_medications empty only if no medication is clinically indicated.
- Use generic drug names in proposed_medications.
Respond with ONLY a JSON object with keys:
  subjective (string), objective (string), assessment (string), plan (string),
  proposed_medications (list of {name, dose, frequency, duration, reason};
  empty list if no medication is proposed)."""

REVISION_INSTRUCTIONS = """
This is a REVISION. A reviewer raised the flags below about your previous draft.
Fix every flag that is supported by the patient record. Return the full note."""

REQUIRED_KEYS = ["subjective", "objective", "assessment", "plan", "proposed_medications"]


def _build_prompt(state: dict, is_revision: bool) -> str:
    """Assemble the user prompt: intake + record, plus the flags and old draft on a revision."""
    parts = [
        "INTAKE (structured symptoms):\n" + json.dumps(state["intake"], indent=2),
        "PATIENT RECORD:\n" + json.dumps(state["records"], indent=2),
    ]
    if is_revision:
        parts.append(REVISION_INSTRUCTIONS)
        parts.append("REVIEWER FLAGS:\n" + json.dumps(state.get("reflection_flags", []), indent=2))
        parts.append("PREVIOUS DRAFT:\n" + json.dumps(state.get("soap_note", {}), indent=2))
    return "\n\n".join(parts)


def documentation_agent(state: dict) -> dict:
    """Draft (or, on a revision pass, redraft) the SOAP note.

    A revision pass happens when the Reflection Agent routes back here. Then
    `revision_count` > 0, the prompt includes the reviewer flags, and the old
    note is kept as `previous_soap_note` so the doctor can see what changed.
    """
    is_revision = state.get("revision_count", 0) > 0
    result = call_llm_json("Documentation", SYSTEM_PROMPT, _build_prompt(state, is_revision))
    require_keys("Documentation", result, REQUIRED_KEYS)
    meds_raw = result.get("proposed_medications")
    result["proposed_medications"] = [m for m in meds_raw if isinstance(m, dict) and m.get("name")] \
        if isinstance(meds_raw, list) else []

    meds = ", ".join(m.get("name", "?") for m in result["proposed_medications"]) or "none"
    label = "Revised SOAP note" if is_revision else "SOAP note drafted"
    update = {
        "soap_note": result,
        "agent_log": [log_entry("Documentation", f"{label}. Proposed medication: {meds}", config.OPENAI_MODEL)],
    }
    if is_revision:
        update["previous_soap_note"] = state.get("soap_note")
    return update
