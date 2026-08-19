# ============================================================================
# StormSentinel Backend — Chat Assistant
# Explains a specific PredictionSnapshot in natural language. Deliberately
# NOT a general meteorology chatbot — it's instructed to reason only about
# the numbers this specific snapshot actually produced, including the
# model's own documented gaps (US-only hazards, low-confidence wildfire
# cities), rather than inventing forecast detail the model never computed.
#
# Stateless by design: the backend may run multiple workers or restart
# between requests, so conversation history is reloaded from the
# chat_messages table on every call rather than kept in an in-memory
# client.chats.create() session (the current google-genai SDK's chat
# sessions don't accept a history= parameter to resume from, so rebuilding
# from stored messages via client.models.generate_content is the correct
# fit here, not a workaround).
# ============================================================================

from google.genai import types

from app.config import get_settings
from app.llm.client import get_client
from app.ml.model import US_ONLY_HAZARDS, LOW_CONFIDENCE_WILDFIRE_CITIES

HEAD_LABELS = {
    "wildfire": "Wildfire",
    "tornado": "Tornado",
    "hail": "Hail",
    "thunderstorm_wind": "Thunderstorm Wind",
    "flash_flood": "Flash Flood",
    "heat": "Extreme Heat",
    "drought": "Drought",
}


def _snapshot_warnings(snapshot) -> list[str]:
    """
    Reconstructs the same caveats /predict would have surfaced, from the
    fields actually persisted on the snapshot. raw_weather/has_true_baseline
    aren't stored on the snapshot row, so the heat-baseline warning can't be
    reconstructed here — the other two can, from is_us_location and city.
    """
    warnings = []
    if not snapshot.is_us_location:
        warnings.append(
            "Tornado, Hail, and Thunderstorm Wind were trained exclusively on US "
            "storm data and have no real grounding for this location — treat those "
            "three scores as unvalidated here."
        )
    if snapshot.city in LOW_CONFIDENCE_WILDFIRE_CITIES:
        warnings.append(
            f"{snapshot.city}'s wildfire score is documented as lower-confidence: "
            f"satellite fire detections here show likely industrial/flare "
            f"contamination that couldn't be cleanly filtered out during training."
        )
    return warnings


def _build_system_instruction(snapshot) -> str:
    scores_lines = []
    for key, label in HEAD_LABELS.items():
        score = getattr(snapshot, f"{key}_score")
        tag = " (US-only-validated hazard)" if key in US_ONLY_HAZARDS else ""
        scores_lines.append(f"- {label}: {round(score)}/100{tag}")

    warnings = _snapshot_warnings(snapshot)
    warnings_block = (
        "\n".join(f"- {w}" for w in warnings)
        if warnings
        else "- None for this location."
    )

    return f"""You are the StormSentinel AI assistant, explaining ONE specific risk
prediction to the user who just ran it. You are not a general meteorology
chatbot and you have no live weather data of your own.

LOCATION: {snapshot.city}, {snapshot.country}
COMPOSITE THREAT INDEX: {round(snapshot.composite_score)}/100

HAZARD SCORES (0-100, higher = higher modeled risk):
{chr(10).join(scores_lines)}

DOCUMENTED MODEL LIMITATIONS RELEVANT TO THIS LOCATION:
{warnings_block}

RULES:
1. Explain and interpret ONLY the numbers given above. Never invent specific
   weather details (temperatures, wind speeds, storm timing) that aren't
   provided to you — you don't have that data.
2. If asked about limitations, be candid using the documented caveats above.
   Don't downplay or hide them.
3. Never claim this is a substitute for official emergency alerts. If risk
   sounds high, encourage checking official sources (national weather
   service, local emergency management) for real-time warnings.
4. Keep answers conversational and concise — this is a chat panel, not a
   report.
5. If asked something with no connection to this prediction (e.g. general
   trivia), answer briefly but steer back to what you can actually help
   with: this specific risk assessment."""


def generate_reply(snapshot, history: list[dict], user_message: str) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": str}, oldest
    first — the prior turns of this snapshot's conversation, as stored in
    chat_messages.
    """
    settings = get_settings()
    client = get_client()

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_build_system_instruction(snapshot),
            temperature=0.4,
        ),
    )
    return response.text
