# CareOps

A multi-agent AI system that reduces doctor burnout. It turns a patient's chat
message into a doctor-ready SOAP note with a prescription safety check, and
leaves the final approval to a human.

```
intake -> records -> documentation -> prescription_safety -> reflection
                          ^                   ^                   |
                          |___________________|__ (flag, once) ___|
                                                                  v
                                            doctor_approval (graph pauses here) -> END
```

| Agent | What it does | Engine |
|---|---|---|
| Intake | Chat message to structured symptoms (JSON) | GPT-4o |
| Records | `get_patient_record(id)` from the local EHR file (`data/mock_patients.json`) | Data lookup |
| Documentation | Drafts the SOAP note plus a structured `proposed_medications` list | GPT-4o |
| Prescription Safety | Allergy, eGFR and interaction checks (RxNorm with local fallback), then SAFE / FLAGGED | Deterministic rules |
| Reflection | Rule checks + GPT-4o critique that returns flags only. Routes back **once** | Rules + GPT-4o |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env     # then put your OPENAI_API_KEY in .env
```

## Run (two terminals)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --port 8010
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8510
```

Open http://localhost:8510. If the API is not running, the UI automatically
runs the graph in-process. The sidebar shows which backend is in use.

To run from the command line without any UI: `python graph.py P003`

A full visit makes 3 to 5 GPT-4o calls and takes about 20 to 40 seconds.

## Failure handling

- **GPT-4o fails** (missing key, network, rate limit, bad JSON): the visit stops
  at that agent and the UI names the agent and the reason. Nothing unverified
  is sent to the doctor. `POST /run_visit` returns HTTP 502.
- **RxNorm fails**: 3-second timeout, with a per-run circuit breaker, then the
  local interaction table in `agents/drug_knowledge.py` is used.

## Notes

- NLM retired the RxNav **Drug Interaction** API in January 2024 (it now returns 404).
  CareOps calls RxNorm live for drug-name to RxCUI normalization, and it
  attempts the interaction endpoint, which falls back to the local table.
- State is checkpointed per visit (`MemorySaver`). You can inspect it at
  `GET /visits/{thread_id}` or in the "Raw checkpointed state" expander.

## Future work (intentionally not built)

- Real EHR/FHIR integration (swap the body of `get_patient_record`)
- Authentication / authorization for approvers
- Database persistence (a SQLite/Postgres checkpointer instead of in-memory)
- Bounded "loop until safe" retry policy beyond one reflection pass
- A licensed drug-interaction database. Re-running the safety check when a doctor edits the medication
