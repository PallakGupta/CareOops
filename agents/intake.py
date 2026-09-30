"""
Intake Agent: turns the patient's free-text chat message into structured symptoms.

Input state keys:  patient_message, patient_id
Output state keys: intake, agent_log
"""

import config
from agents.common import call_llm_json, log_entry, require_keys

SYSTEM_PROMPT = """You are a clinical intake assistant. Extract structured symptoms
from a patient's chat message. Do not diagnose. Do not invent symptoms that are
not in the message. Respond with ONLY a JSON object with these keys:
  chief_complaint (string), duration (string, "unspecified" if not stated),
  severity (string: mild / moderate / severe, plus a 0-10 score if given),
  associated_symptoms (list of strings), pertinent_negatives (list of strings,
  symptoms the patient explicitly denies), red_flags (list of strings, e.g.
  chest pain, difficulty breathing, fainting; empty list if none)."""

REQUIRED_KEYS = ["chief_complaint", "duration", "severity", "associated_symptoms"]


def intake_agent(state: dict) -> dict:
    """Extract chief complaint, duration, severity and associated symptoms as JSON.

    The schema is validated, so later agents can rely on these keys existing.
    Optional list fields default to empty lists.
    """
    message = state["patient_message"]
    result = call_llm_json("Intake", SYSTEM_PROMPT, f"Patient message:\n{message}")
    require_keys("Intake", result, REQUIRED_KEYS)
    for key in ("associated_symptoms", "pertinent_negatives", "red_flags"):
        if not isinstance(result.get(key), list):
            result[key] = []

    intake = {**result, "raw_message": message}
    summary = f"Chief complaint: {intake['chief_complaint']} ({intake['duration']}, {intake['severity']})"
    return {"intake": intake, "agent_log": [log_entry("Intake", summary, config.OPENAI_MODEL)]}
