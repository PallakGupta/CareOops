"""
CareOps orchestrator: the LangGraph StateGraph that wires the agents together.

    START -> intake -> records -> documentation -> prescription_safety -> reflection
                                      ^                   ^                  |
                                      |___________________|__ (flag, once) __|
                                                                             |
                                          (no flags / already revised)       v
                                                   doctor_approval  <-- PAUSE here
                                                         |
                                                        END

Key ideas to explain:
- State: one shared dict (CareOpsState). Each node returns ONLY the keys it
  changes, and LangGraph merges them in. `agent_log` and `reflection_history`
  use an `operator.add` reducer, so entries are appended instead of overwritten.
- Conditional edge: `route_after_reflection` reads `revision_target`, which the
  Reflection Agent sets at most MAX_REFLECTION_REVISIONS (=1) times.
- Checkpointer: MemorySaver stores the state after every node, keyed by
  thread_id. That makes state inspectable (get_visit_state) and lets the graph
  PAUSE before `doctor_approval` (human-in-the-loop) and resume later when the
  doctor clicks a button.

Future work: persistent checkpointer (SQLite/Postgres) instead of in-memory,
a bounded "loop until safe" policy, and authentication on who may approve.
"""

import operator
import time
import uuid
from typing import Annotated, Iterator, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.common import LLMError
from agents.documentation import documentation_agent
from agents.intake import intake_agent
from agents.prescription_safety import prescription_safety_agent
from agents.records import records_agent
from agents.reflection import reflection_agent


class CareOpsState(TypedDict, total=False):
    """The single shared state object that flows through every node."""
    patient_id: str
    patient_message: str
    intake: dict
    records: dict
    soap_note: dict
    previous_soap_note: dict
    safety_check: dict
    safety_history: Annotated[list, operator.add]
    reflection_flags: list
    reflection_history: Annotated[list, operator.add]
    revision_count: int
    revision_target: str | None
    doctor_decision: dict
    final_soap_note: dict
    status: str
    agent_log: Annotated[list, operator.add]


# Human-readable names for the UI's live status list.
NODE_LABELS = {
    "intake": "Intake Agent",
    "records": "Records Agent",
    "documentation": "Documentation Agent",
    "prescription_safety": "Prescription Safety Agent",
    "reflection": "Reflection Agent",
    "doctor_approval": "Doctor Approval",
}

VALID_DECISIONS = {"approve": "approved", "reject": "rejected", "edit": "approved_with_edits"}


def doctor_approval_node(state: dict) -> dict:
    """Human-in-the-loop step. The graph PAUSES before this node (interrupt_before).

    When the doctor clicks Approve/Reject/Edit, `submit_decision` writes
    `doctor_decision` into the checkpointed state and resumes the graph. This
    node then records the final status. An "edit" overwrites the plan text in
    `final_soap_note`.

    Future work: write the approved note back to the EHR, and re-run the safety
    agent automatically when the doctor edits the medication.
    """
    decision = state.get("doctor_decision") or {}
    action = decision.get("action", "approve")
    final_note = dict(state.get("soap_note") or {})
    if action == "edit" and decision.get("edited_plan"):
        final_note["plan"] = decision["edited_plan"]

    status = VALID_DECISIONS.get(action, "approved")
    summary = f"Doctor decision: {status.upper()}" + (f" ({decision['notes']})" if decision.get("notes") else "")
    return {
        "final_soap_note": final_note,
        "status": status,
        "agent_log": [{"agent": "Doctor Approval", "summary": summary, "source": "human",
                       "timestamp": time.strftime("%H:%M:%S")}],
    }


def route_after_reflection(state: dict) -> str:
    """Conditional edge after Reflection: go back to the flagged agent once, else to the doctor.

    The Reflection Agent already applied the "only once" rule when it set
    `revision_target`, so this function is a pure read of the state.
    """
    return state.get("revision_target") or "doctor_approval"


def build_graph():
    """Define the nodes and edges, then compile with a checkpointer and a pause before approval."""
    builder = StateGraph(CareOpsState)

    builder.add_node("intake", intake_agent)
    builder.add_node("records", records_agent)
    builder.add_node("documentation", documentation_agent)
    builder.add_node("prescription_safety", prescription_safety_agent)
    builder.add_node("reflection", reflection_agent)
    builder.add_node("doctor_approval", doctor_approval_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "records")
    builder.add_edge("records", "documentation")
    builder.add_edge("documentation", "prescription_safety")
    builder.add_edge("prescription_safety", "reflection")
    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"documentation": "documentation",
         "prescription_safety": "prescription_safety",
         "doctor_approval": "doctor_approval"},
    )
    builder.add_edge("doctor_approval", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["doctor_approval"])


graph = build_graph()


# ---------------------------------------------------------------------------
# Service helpers used by both api.py and app.py (so they share one code path).
# ---------------------------------------------------------------------------

def _thread_config(thread_id: str) -> dict:
    """LangGraph config that selects which checkpointed conversation (visit) to use."""
    return {"configurable": {"thread_id": thread_id}}


def get_visit_state(thread_id: str) -> dict:
    """Read the checkpointed state for a visit: values plus which node would run next."""
    snapshot = graph.get_state(_thread_config(thread_id))
    return {"thread_id": thread_id, "next": list(snapshot.next), "state": dict(snapshot.values)}


def stream_visit(patient_id: str, patient_message: str, thread_id: str | None = None) -> Iterator[dict]:
    """Run the graph and yield one event per finished agent, then a final event.

    Events:
      {"type": "agent_done", "node": ..., "label": ..., "log": [...]}   after each node
      {"type": "paused", "thread_id": ..., "next": [...], "state": {...}} at the approval pause
      {"type": "error", "agent": ..., "error": ..., "thread_id": ..., "state": {...}}
          if an agent failed (e.g. GPT-4o unreachable). The visit stops there;
          nothing unverified is ever shown to the doctor as a finished note.
    """
    thread_id = thread_id or str(uuid.uuid4())
    initial_state = {
        "patient_id": patient_id,
        "patient_message": patient_message,
        "revision_count": 0,
        "status": "running",
    }
    try:
        for chunk in graph.stream(initial_state, _thread_config(thread_id), stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                yield {"type": "agent_done", "node": node, "label": NODE_LABELS.get(node, node),
                       "log": (update or {}).get("agent_log", [])}
    except Exception as exc:  # noqa: BLE001 - surface any agent failure as a clean event
        agent = exc.agent if isinstance(exc, LLMError) else "pipeline"
        reason = exc.reason if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
        yield {"type": "error", "agent": agent, "error": reason, **get_visit_state(thread_id)}
        return
    yield {"type": "paused", **get_visit_state(thread_id)}


def run_visit(patient_id: str, patient_message: str) -> dict:
    """Run the whole pipeline up to the doctor-approval pause and return the last event."""
    final = None
    for event in stream_visit(patient_id, patient_message):
        final = event
    return final


def submit_decision(thread_id: str, action: str, edited_plan: str | None = None, notes: str | None = None) -> dict:
    """Record the doctor's decision in the checkpointed state and resume the graph to END."""
    if action not in VALID_DECISIONS:
        raise ValueError(f"action must be one of {sorted(VALID_DECISIONS)}")
    cfg = _thread_config(thread_id)
    if not graph.get_state(cfg).next:
        raise ValueError(f"Visit {thread_id} is not awaiting approval")
    graph.update_state(cfg, {"doctor_decision": {"action": action, "edited_plan": edited_plan, "notes": notes}})
    graph.invoke(None, cfg)
    return get_visit_state(thread_id)


def graph_mermaid() -> str:
    """Mermaid diagram of the compiled graph (shown in the UI and at GET /graph)."""
    return graph.get_graph().draw_mermaid()


if __name__ == "__main__":
    import json
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else "P001"
    from agents.records import get_patient_record
    msg = (get_patient_record(pid) or {}).get("sample_message", "I feel unwell.")
    result = run_visit(pid, msg)
    for entry in result["state"].get("agent_log", []):
        print(f"[{entry['agent']}] {entry['summary']}  ({entry['source']})")
    if result["type"] == "error":
        sys.exit(f"\nFAILED in {result['agent']}: {result['error']}")
    print("\nSafety verdict:", result["state"]["safety_check"]["verdict"])
    print("Next node:", result["next"])
    print(json.dumps(result["state"]["reflection_flags"], indent=2))
