"""Ask the Copilot — the store manager's GenAI assistant. WALLED OFF by design.

This module reads FROM the deterministic outputs (the store's computed plan,
flags, roster story, rationale) and answers the manager's questions about them.
It never feeds anything INTO the forecast or the labor math, and it cannot
change the plan — disagreement is routed to the manager-override workflow
(human in the loop; shift assignment stays in Toast Scheduling / Square Shifts).

Key handling: BACKEND-ONLY — the operator sets ANTHROPIC_API_KEY (env var) or
.streamlit/secrets.toml (gitignored). Store managers never see or handle a key,
exactly as they wouldn't in production. Without a key the tab degrades
gracefully; the rest of the app is fully functional.
"""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-haiku-4-5"   # cheap and fast; plenty for grounded Q&A

SYSTEM_TEMPLATE = """You are the Solstice Labor Copilot assistant for the store manager at {store_name}.

Your ONLY job is to help this manager understand and interrogate their staffing plan.
Everything you know about their store is in the STORE DATA below — computed by a
demand-forecast model and a deterministic labor engine, NOT by you.

Rules you must follow:
1. Ground every number in the STORE DATA. Never invent or estimate numbers that are
   not derivable from it by simple arithmetic. If asked something the data cannot
   answer, say so plainly.
2. For what-if questions ("what if I keep 3 people Saturday morning?"), do the simple
   arithmetic from the data (hours x wage, days per week) and show your work briefly.
3. You cannot change the plan, and neither does talking to you. If the manager
   disagrees with a recommendation for a reason the model can't see (local event,
   training, staffing constraint, weather they know is coming), tell them their
   judgment wins: point them to the Manager override on the action plan with the
   most fitting reason code from: {reason_codes}. Overridden slots are excluded
   from the plan automatically.
4. Shift assignment (who works when) happens in {pos_label} — this tool sets the
   staffing TARGETS; people decisions stay with the manager.
5. Respect the service guardrails: never suggest cutting a daypart flagged
   "understaffed at peak". If asked to, explain the flag instead.
6. Be concise and plain-spoken — you're talking to a busy store manager, not an
   analyst. Short paragraphs, dollar figures rounded, no jargon.

STORE DATA (deterministic model outputs — your single source of truth):
{context_json}
"""


def build_context(store: dict, plan_dayparts: list[dict], actions: list[dict],
                  roster: dict, rationale_lines: dict, weekly: dict) -> str:
    """Compact JSON grounding block for the system prompt."""
    ctx = {
        "store": store,
        "typical_week_by_daypart": plan_dayparts,
        "action_plan": actions,
        "roster_story": roster,
        "weekly_summary": weekly,
        "forecast_rationale": rationale_lines,
        "notes": {
            "hours_are_fleet_scale": "all hours/dollars at real fleet scale",
            "min_shift_hours": 3,
            "two_person_minimum_while_open": True,
            "plan_floor_labor_pct": 0.27,
        },
    }
    return json.dumps(ctx, indent=1, default=str)


def make_system(store_name: str, pos_system: str, context_json: str,
                reason_codes: list[str]) -> str:
    pos_label = ("Toast Scheduling (powered by Sling)" if pos_system == "toast"
                 else "Square Shifts")
    return SYSTEM_TEMPLATE.format(
        store_name=store_name,
        pos_label=pos_label,
        reason_codes=", ".join(reason_codes),
        context_json=context_json,
    )


def get_api_key() -> str | None:
    """Backend-only key resolution: env var, then Streamlit secrets (gitignored)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")  # .streamlit/secrets.toml
    except Exception:
        return None


def stream_reply(client, model: str, system: str, history: list[dict]):
    """Streaming context manager for the chat turn. History = [{role, content}, ...]."""
    return client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        messages=history,
    )
