# ============================================================================
# StormSentinel Backend — Gemini Client Factory
# Single place that constructs the google-genai client, reused by both the
# chat assistant and the notification email composer. Uses the current
# `google-genai` SDK (`from google import genai`) — NOT the deprecated
# `google-generativeai` legacy package. See Hard-Won Lessons in project docs.
# ============================================================================

from functools import lru_cache

from google import genai

from app.config import get_settings


@lru_cache
def get_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)
