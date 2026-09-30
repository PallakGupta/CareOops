"""
Shared helpers used by every agent:

- `call_llm_json`: the ONE place GPT-4o is called. It returns parsed JSON or
  raises `LLMError`, so a failed call can never be passed off as a real note.
- `require_keys`: validates the shape of an LLM answer.
- `log_entry`: builds the status line each agent appends to `agent_log`,
  which the UI renders as the live "Intake done ✓" list.
"""

import json
from datetime import datetime
from typing import Any

import config


class LLMError(RuntimeError):
    """Raised when GPT-4o is unreachable or returns unusable output. The visit stops and the UI shows why."""

    def __init__(self, agent: str, reason: str):
        super().__init__(f"{agent}: {reason}")
        self.agent = agent
        self.reason = reason


def call_llm_json(agent: str, system_prompt: str, user_prompt: str) -> dict:
    """Ask GPT-4o for a JSON object and return it parsed.

    Uses OpenAI JSON mode (`response_format=json_object`) at temperature 0,
    with automatic retries on transient errors. Any failure (missing key,
    network, rate limit, invalid JSON) raises LLMError naming the agent.
    """
    if not config.OPENAI_API_KEY:
        raise LLMError(agent, "OPENAI_API_KEY is not set (add it to .env)")

    from langchain_openai import ChatOpenAI

    try:
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
            api_key=config.OPENAI_API_KEY,
        ).bind(response_format={"type": "json_object"})
        response = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        result = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise LLMError(agent, f"model returned invalid JSON ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 - normalize every provider error into one type
        raise LLMError(agent, f"{type(exc).__name__}: {exc}") from exc

    if not isinstance(result, dict):
        raise LLMError(agent, "model did not return a JSON object")
    return result


def require_keys(agent: str, result: dict, keys: list[str]) -> None:
    """Raise LLMError if the model's JSON is missing any required key."""
    missing = [k for k in keys if k not in result]
    if missing:
        raise LLMError(agent, f"model output missing keys: {', '.join(missing)}")


def log_entry(agent: str, summary: str, source: str) -> dict:
    """Build one status line for the shared `agent_log` (rendered live in the UI)."""
    return {
        "agent": agent,
        "summary": summary,
        "source": source,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
