"""
CareOps FastAPI backend.

Run:  uvicorn api:app --port 8010     (docs at http://127.0.0.1:8010/docs)

Endpoints:
  POST /run_visit              -> run all agents, return the full state (paused for approval)
  POST /run_visit/stream       -> same run, streamed as NDJSON (one line per agent; used by the UI)
  POST /visits/{id}/decision   -> doctor Approve / Reject / Edit, resumes the graph to END
  GET  /visits/{id}            -> inspect the checkpointed state of any visit
  GET  /patients, /health, /graph

Future work: authentication/authorization (who may approve a note), audit
logging, and database persistence. Nothing here is persisted: restarting the
server clears all visits.
"""

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from agents.records import get_patient_record, list_patients
from graph import get_visit_state, graph_mermaid, run_visit, stream_visit, submit_decision

app = FastAPI(title="CareOps API", version="0.1.0",
              description="Multi-agent clinical documentation + prescription safety pipeline")


class VisitRequest(BaseModel):
    """Body for /run_visit: which patient, and what they typed in the chat."""
    patient_id: str = Field(..., examples=["P001"])
    message: str = Field(..., min_length=1, examples=["My knee has hurt for 2 weeks, 6/10."])


class DecisionRequest(BaseModel):
    """Body for the doctor's decision on a paused visit."""
    action: Literal["approve", "reject", "edit"]
    edited_plan: str | None = None
    notes: str | None = None


def _require_patient(patient_id: str) -> None:
    """Return 404 early for unknown patients instead of running the graph on an empty record."""
    if get_patient_record(patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id {patient_id}")


@app.get("/health")
def health() -> dict:
    """Liveness check, plus the runtime configuration, so the UI can display it."""
    return {"status": "ok", "service": "careops", "llm_configured": bool(config.OPENAI_API_KEY),
            "mode": config.describe_mode()}


@app.get("/patients")
def patients() -> list[dict]:
    """List the mock patients (for the UI dropdown)."""
    return list_patients()


@app.post("/run_visit")
def run_visit_endpoint(req: VisitRequest) -> dict:
    """Run intake -> records -> documentation -> safety -> reflection and return the full state.

    The graph pauses before doctor approval. Use the returned `thread_id` with
    /visits/{thread_id}/decision to finish it. If an agent fails (e.g. GPT-4o
    unreachable), this returns 502 with the failing agent and reason.
    """
    _require_patient(req.patient_id)
    result = run_visit(req.patient_id, req.message)
    if result["type"] == "error":
        raise HTTPException(status_code=502, detail={"agent": result["agent"], "error": result["error"],
                                                     "thread_id": result["thread_id"]})
    return result


@app.post("/run_visit/stream")
def run_visit_stream(req: VisitRequest) -> StreamingResponse:
    """Same as /run_visit, but streams one JSON line per completed agent (NDJSON).

    This is what powers the live "Intake done ✓" status list in the Streamlit UI.
    """
    _require_patient(req.patient_id)

    def events():
        for event in stream_visit(req.patient_id, req.message):
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/visits/{thread_id}")
def visit_state(thread_id: str) -> dict:
    """Inspect the checkpointed state of a visit, the proof that state is persisted between steps."""
    result = get_visit_state(thread_id)
    if not result["state"]:
        raise HTTPException(status_code=404, detail="Unknown visit")
    return result


@app.post("/visits/{thread_id}/decision")
def decision(thread_id: str, req: DecisionRequest) -> dict:
    """Doctor's Approve / Reject / Edit. Resumes the paused graph and returns the final state."""
    try:
        return submit_decision(thread_id, req.action, req.edited_plan, req.notes)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/graph", response_class=PlainTextResponse)
def graph_diagram() -> str:
    """Mermaid source for the agent graph (paste into mermaid.live for a diagram)."""
    return graph_mermaid()
