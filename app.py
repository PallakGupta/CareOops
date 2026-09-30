"""
CareOps Streamlit frontend.

Run:  streamlit run app.py

Two views (switch in the sidebar):
  1. Patient chat     : pick a patient, send a symptom message, and watch
                        each agent finish live.
  2. Doctor approval  : SOAP note + safety verdict + reflection flags, with
                        Approve / Reject / Edit.

Backend: talks to the FastAPI server (api.py) when it is running. If it is not
reachable, it runs the same LangGraph pipeline in-process. It works either
way, and the sidebar shows which backend is in use.
"""

import json

import requests
import streamlit as st

import config

st.set_page_config(page_title="CareOps", page_icon="🩺", layout="wide")

NEXT_STEP = {
    None: "Intake Agent is reading the message",
    "intake": "Records Agent is pulling the patient's history",
    "records": "Documentation Agent is drafting the SOAP note",
    "documentation": "Prescription Safety Agent is checking the medication",
    "prescription_safety": "Reflection Agent is reviewing the note and the safety check",
    "reflection": "Routing based on the Reflection Agent's flags",
}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "info": "🔵"}


# ---------------------------------------------------------------------------
# Backend adapter: FastAPI over HTTP, or the graph in-process as a fallback.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def api_available() -> bool:
    """True if the CareOps FastAPI backend answers /health within 1 second (re-checked every 10s)."""
    try:
        resp = requests.get(f"{config.API_URL}/health", timeout=1)
        return resp.ok and resp.json().get("service") == "careops"
    except (requests.RequestException, ValueError):
        return False


def get_patients() -> list[dict]:
    """Patient list for the dropdown (from the API, or straight from the mock EHR)."""
    if api_available():
        return requests.get(f"{config.API_URL}/patients", timeout=5).json()
    from agents.records import list_patients
    return list_patients()


def stream_visit(patient_id: str, message: str):
    """Yield pipeline events (one per finished agent, then a final 'paused' event)."""
    if api_available():
        with requests.post(f"{config.API_URL}/run_visit/stream",
                           json={"patient_id": patient_id, "message": message},
                           stream=True, timeout=(3, 180)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line)
    else:
        import graph
        yield from graph.stream_visit(patient_id, message)


def submit_decision(thread_id: str, action: str, edited_plan: str | None, notes: str | None) -> dict:
    """Send the doctor's decision and return the final visit state."""
    if api_available():
        resp = requests.post(f"{config.API_URL}/visits/{thread_id}/decision",
                             json={"action": action, "edited_plan": edited_plan, "notes": notes}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    import graph
    return graph.submit_decision(thread_id, action, edited_plan, notes)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Draw the sidebar (view switch, run mode, backend in use) and return the chosen view."""
    st.sidebar.title("🩺 CareOps")
    st.sidebar.caption("Multi-agent clinical documentation & prescription safety")
    view = st.sidebar.radio("View", ["Patient chat", "Doctor approval"], key="view")

    st.sidebar.divider()
    st.sidebar.markdown("**Models**")
    (st.sidebar.success if config.OPENAI_API_KEY else st.sidebar.error)(config.describe_mode())
    st.sidebar.markdown("**Backend**")
    if api_available():
        st.sidebar.info(f"FastAPI at {config.API_URL}")
    else:
        st.sidebar.info("In-process LangGraph (API not running)")

    with st.sidebar.expander("Agent graph (Mermaid)"):
        import graph
        st.code(graph.graph_mermaid(), language="text")
    return view


# ---------------------------------------------------------------------------
# View 1: Patient chat
# ---------------------------------------------------------------------------

def go_to_doctor_view() -> None:
    """Button callback: switch the sidebar radio to the doctor view."""
    st.session_state.view = "Doctor approval"


def render_patient_view() -> None:
    """Patient picks a profile, sends a symptom message, and watches the agents run live."""
    st.header("Patient chat")
    patients = get_patients()
    by_label = {f"{p['id']} · {p['name']} ({p['age']})": p for p in patients}
    label = st.selectbox("Patient", list(by_label))
    patient = by_label[label]

    message = st.text_area("Describe your symptoms", value=patient["sample_message"],
                           height=110, key=f"msg_{patient['id']}")
    st.caption("Pre-filled with a sample message for this patient. Edit it freely: "
               "GPT-4o analyzes whatever you type.")

    if not st.button("Submit to care team", type="primary", disabled=not message.strip()):
        if st.session_state.get("visit"):
            st.button("Open Doctor approval view →", on_click=go_to_doctor_view)
        return

    st.session_state.pop("visit", None)
    seen: dict[str, int] = {}
    failure = None
    with st.status("Running CareOps agents…", expanded=True) as status:
        try:
            steps = st.container()
            working = st.empty()
            working.markdown(f"⏳ *{NEXT_STEP[None]}…*")
            for event in stream_visit(patient["id"], message):
                if event["type"] == "agent_done":
                    seen[event["node"]] = seen.get(event["node"], 0) + 1
                    again = " (revision pass)" if seen[event["node"]] > 1 else ""
                    for entry in event["log"]:
                        steps.markdown(f"✓ **{event['label']} done**{again}: {entry['summary']}  \n"
                                       f"<small>source: {entry['source']}</small>", unsafe_allow_html=True)
                    working.markdown(f"⏳ *{NEXT_STEP.get(event['node'], 'Working')}…*")
                elif event["type"] == "paused":
                    st.session_state.visit = event
                elif event["type"] == "error":
                    failure = event
            working.empty()
        except Exception as exc:  # noqa: BLE001 - show a clean message instead of a stack trace
            failure = {"agent": "connection", "error": str(exc)}

        if failure:
            status.update(label=f"Stopped: {failure['agent']} failed", state="error", expanded=True)
        else:
            status.update(label="All agents done ✓, awaiting doctor approval", state="complete", expanded=True)

    if failure:
        st.error(f"**The visit was stopped because the {failure['agent']} step failed.** "
                 f"No note was sent to the doctor.\n\nReason: {failure['error']}")
        return

    st.success("Your message has been sent to the care team. A doctor will review it shortly.")
    st.button("Open Doctor approval view →", type="primary", on_click=go_to_doctor_view)


# ---------------------------------------------------------------------------
# View 2: Doctor approval
# ---------------------------------------------------------------------------

def render_soap(note: dict) -> None:
    """Render a SOAP note dict as four labelled sections plus a medication table."""
    for key, title in [("subjective", "S: Subjective"), ("objective", "O: Objective"),
                       ("assessment", "A: Assessment"), ("plan", "P: Plan")]:
        st.markdown(f"**{title}**")
        st.write(note.get(key, "—"))
    meds = note.get("proposed_medications") or []
    st.markdown("**Proposed medications**")
    if meds:
        st.table([{k: m.get(k, "") for k in ("name", "dose", "frequency", "duration", "reason")} for m in meds])
    else:
        st.write("None")


def render_safety(safety: dict) -> None:
    """Show the SAFE/FLAGGED verdict, its reasoning, each issue, and the checks that ran."""
    verdict = safety.get("verdict", "UNKNOWN")
    if verdict == "SAFE":
        st.success(f"**Verdict: SAFE**\n\n{safety.get('reasoning', '')}")
    else:
        st.error(f"**Verdict: {verdict}**")
        for issue in safety.get("issues", []):
            st.markdown(f"{SEVERITY_ICON.get(issue['severity'], '⚪')} **{issue['check'].title()}**: {issue['detail']}")
    with st.expander("Checks performed"):
        for check in safety.get("checks_performed", []):
            st.markdown(f"- {check}")
        for med in safety.get("medications_checked", []):
            st.caption(f"{med['normalized']}: RxCUI {med.get('rxcui') or 'n/a'}")
        st.caption(f"Interaction source: {safety.get('interaction_source', 'n/a')} · "
                   f"RxNorm: {safety.get('rxnorm_status', 'n/a')}")


def render_reflection(history: list[dict]) -> None:
    """Show every reflection pass, its flags, and where (if anywhere) it routed the work."""
    if not history:
        st.write("No reflection output.")
        return
    for rpass in history:
        routed = f" → routed back to **{rpass['routed_to']}**" if rpass.get("routed_to") else " → handed to doctor"
        st.markdown(f"**Pass {rpass['pass']}**: {len(rpass['flags'])} flag(s){routed}")
        if not rpass["flags"]:
            st.caption("No issues found.")
        for flag in rpass["flags"]:
            target = flag.get("target_agent") or "informational"
            st.markdown(f"{SEVERITY_ICON.get(flag.get('severity'), '⚪')} `{target}` · `{flag.get('source')}`: "
                        f"{flag['issue']}  \n<small>Evidence: {flag.get('evidence', '')}</small>",
                        unsafe_allow_html=True)


def render_decision_panel(visit: dict) -> None:
    """Approve / Reject / Edit buttons. They resume the paused graph through the checkpointer."""
    state = visit["state"]
    awaiting = "doctor_approval" in visit.get("next", [])
    if not awaiting:
        decision = state.get("doctor_decision") or {}
        st.info(f"Decision recorded: **{state.get('status', '').upper()}**"
                + (f" · notes: {decision['notes']}" if decision.get("notes") else ""))
        if decision.get("action") == "edit":
            st.markdown("**Final plan (edited by doctor)**")
            st.write(state.get("final_soap_note", {}).get("plan", ""))
        return

    notes = st.text_input("Notes (optional)", key="decision_notes")
    col1, col2, col3 = st.columns(3)
    action = None
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        action = "approve"
    if col2.button("❌ Reject", use_container_width=True):
        action = "reject"
    if col3.button("✏️ Edit", use_container_width=True):
        st.session_state.editing = True

    edited_plan = None
    if st.session_state.get("editing"):
        edited_plan = st.text_area("Edit plan", value=state["soap_note"].get("plan", ""), height=120,
                                   key=f"edit_{visit['thread_id']}")
        if st.button("Save edits & approve", type="primary"):
            action = "edit"

    if action:
        st.session_state.visit = submit_decision(visit["thread_id"], action, edited_plan, notes or None)
        st.session_state.editing = False
        st.rerun()


def render_doctor_view() -> None:
    """One screen with everything the doctor needs to sign off, plus the full agent trace."""
    st.header("Doctor approval")
    visit = st.session_state.get("visit")
    if not visit:
        st.info("No visit waiting. Submit a patient message in the **Patient chat** view first.")
        return

    state = visit["state"]
    records = state.get("records", {})
    st.markdown(f"**Patient:** {records.get('name')} · {records.get('age')}{records.get('sex', '')} · "
                f"{', '.join(records.get('conditions', []))}  \n"
                f"**Visit id:** `{visit['thread_id']}` · **Status:** `{state.get('status')}` · "
                f"**Reflection revisions:** {state.get('revision_count', 0)}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("SOAP note (draft)")
        render_soap(state.get("soap_note", {}))
        if state.get("previous_soap_note"):
            with st.expander("First draft, before the Reflection Agent's revision"):
                render_soap(state["previous_soap_note"])
    with right:
        st.subheader("Prescription safety")
        render_safety(state.get("safety_check", {}))
        for i, earlier in enumerate(state.get("safety_history", [])[:-1], start=1):
            names = ", ".join(m["normalized"] for m in earlier.get("medications_checked", []))
            with st.expander(f"Earlier verdict (pass {i}, before revision): {earlier['verdict']} for {names}"):
                st.write(earlier.get("reasoning", ""))
                for issue in earlier.get("issues", []):
                    st.markdown(f"{SEVERITY_ICON.get(issue['severity'], '⚪')} **{issue['check'].title()}**: "
                                f"{issue['detail']}")
        st.subheader("Reflection flags")
        render_reflection(state.get("reflection_history", []))

    st.divider()
    st.subheader("Decision")
    render_decision_panel(visit)

    st.divider()
    with st.expander("Agent trace (every step, in order)"):
        for entry in state.get("agent_log", []):
            st.markdown(f"`{entry['timestamp']}` **{entry['agent']}**: {entry['summary']} "
                        f"<small>({entry['source']})</small>", unsafe_allow_html=True)
    with st.expander("Intake (structured symptoms)"):
        st.json(state.get("intake", {}))
    with st.expander("Patient record (mock EHR)"):
        st.json(records)
    with st.expander("Raw checkpointed state"):
        st.json(visit)


def main() -> None:
    """Streamlit entry point: sidebar picks the view, then that view renders."""
    view = render_sidebar()
    if view == "Patient chat":
        render_patient_view()
    else:
        render_doctor_view()


main()
