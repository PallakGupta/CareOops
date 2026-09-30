"""
Records Agent: looks up the patient's history in the (mock) EHR.

Input state keys:  patient_id
Output state keys: records, agent_log

No LLM is involved. This agent is a pure data lookup, so it can never
hallucinate a medication or an allergy.
"""

import json
from functools import lru_cache

import config
from agents.common import log_entry


@lru_cache(maxsize=1)
def _load_mock_ehr() -> dict:
    """Read data/mock_patients.json once and index the patients by id."""
    with open(config.MOCK_PATIENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data["patients"]}


def list_patients() -> list[dict]:
    """Return [{id, name, age, sample_message}, ...] for the UI dropdown."""
    return [
        {"id": p["id"], "name": p["name"], "age": p["age"], "sample_message": p.get("sample_message", "")}
        for p in _load_mock_ehr().values()
    ]


def get_patient_record(patient_id: str) -> dict | None:
    """Fetch one patient's record: demographics, allergies, meds, labs, past visits.

    THIS is the EHR integration seam. Today it reads a local JSON file. To use a
    real EHR, replace the body with a FHIR call (e.g. GET /Patient/{id},
    /AllergyIntolerance, /MedicationStatement, /Observation) that returns the
    same dict shape. Nothing else in the pipeline has to change.

    Future work: real EHR/FHIR integration, auth, and consent checks.
    """
    return _load_mock_ehr().get(patient_id)


def records_agent(state: dict) -> dict:
    """Graph node: attach the patient's record to the shared state.

    An unknown patient id does not crash the graph. It returns an empty record
    with a `not_found` marker, so the downstream agents and the doctor can see it.
    """
    patient_id = state["patient_id"]
    record = get_patient_record(patient_id)

    if record is None:
        record = {
            "id": patient_id, "name": "Unknown patient", "not_found": True,
            "allergies": [], "current_medications": [], "recent_labs": [], "past_visits": [],
        }
        summary = f"No record found for {patient_id}"
    else:
        summary = (
            f"{record['name']}, {record['age']}{record.get('sex', '')}: "
            f"{len(record['current_medications'])} meds, {len(record['allergies'])} allergies, "
            f"{len(record['recent_labs'])} recent labs"
        )

    return {"records": record, "agent_log": [log_entry("Records", summary, "mock EHR")]}
