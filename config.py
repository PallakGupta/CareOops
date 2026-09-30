"""
CareOps central configuration.

Every agent that needs language understanding calls GPT-4o (key from .env).
The Prescription Safety Agent calls the public RxNorm API, with a local
drug-knowledge fallback for the interaction lookup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MOCK_PATIENTS_PATH = DATA_DIR / "mock_patients.json"

# --- LLM (OpenAI) -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 2

# --- RxNorm (NLM public API, no key required) --------------------------------
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_TIMEOUT_SECONDS = 3

# --- Orchestration -----------------------------------------------------------
# How many times the Reflection Agent may send work back to another agent.
# Kept at 1 on purpose: one self-correction pass, then the doctor decides.
# Future work: a bounded "loop until safe" policy with escalation rules.
MAX_REFLECTION_REVISIONS = 1

# --- Service URLs -------------------------------------------------------------
API_URL = os.getenv("CAREOPS_API_URL", "http://127.0.0.1:8010")


def describe_mode() -> str:
    """One-line description of the runtime configuration (shown in the UI)."""
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY missing: add it to .env, then restart"
    return f"{OPENAI_MODEL} + live RxNorm"
