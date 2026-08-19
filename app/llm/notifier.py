# ============================================================================
# StormSentinel Backend — Notification Composer
# Turns a detected risk change (old snapshot -> new snapshot) into a short,
# plain-language email body via Gemini. Same "explain the given data, don't
# invent detail" grounding discipline as the chat assistant.
#
# Explicitly informational-only framing (a deliberate project decision, not
# an oversight) — this is never presented as a life-saving alert, since the
# model has real, documented gaps. Every email states outright that it is
# not a substitute for official emergency alerts.
# ============================================================================

from google.genai import types

from app.config import get_settings
from app.llm.client import get_client
from app.llm.assistant import HEAD_LABELS

SYSTEM_INSTRUCTION = """You write short notification emails for StormSentinel AI, a
weather-risk-scoring app. You are given one location's PREVIOUS scores and
CURRENT scores for a set of hazards (0-100, higher = higher modeled risk),
and which hazards changed significantly.

Write a brief, plain-language email body (3-5 sentences, no subject line,
no greeting like "Dear user") that:
1. States which hazard(s) rose and by roughly how much, in plain words
   (e.g. "wildfire risk has climbed sharply") rather than restating every
   raw number.
2. Stays neutral and factual — no dramatic or alarmist language.
3. Explicitly says this is informational only and NOT a substitute for
   official emergency alerts (National Weather Service or local emergency
   services), and that people should follow official guidance if issued.
4. Does not invent any weather detail beyond the scores given.
Output plain text only, no markdown, no HTML."""


def compose_notification_body(city: str, country: str, old_scores: dict, new_scores: dict, changed_hazards: list[str]) -> str:
    settings = get_settings()
    client = get_client()

    lines = [f"Location: {city}, {country}", "", "Changed hazards:"]
    for key in changed_hazards:
        label = HEAD_LABELS.get(key, key)
        lines.append(f"- {label}: {round(old_scores[key])} -> {round(new_scores[key])}")

    prompt = "\n".join(lines)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
        ),
    )
    return response.text
