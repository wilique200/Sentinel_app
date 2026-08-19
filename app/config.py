# ============================================================================
# StormSentinel Backend — Configuration
# Reads settings from environment variables (.env in local dev, real env
# vars in production). Never commit actual secrets — .env is gitignored,
# .env.example shows the required shape.
# ============================================================================

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/stormsentinel"

    # Auth
    jwt_secret_key: str = "CHANGE_THIS_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # LLM — free tier (Gemini), not a paid API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"

    # External APIs (no keys needed — Open-Meteo, NOAA, FIRMS are free/public)
    firms_map_key: str = ""  # only needed if the backend itself pulls fresh FIRMS data

    # Email notifications — Resend free tier. NOTE (verified, not assumed):
    # sending from the sandbox address onboarding@resend.dev only delivers
    # to the email on the Resend account itself. Sending to arbitrary user
    # recipients requires verifying a real domain in the Resend dashboard
    # and setting resend_from_email to an address on that domain.
    resend_api_key: str = ""
    resend_from_email: str = "onboarding@resend.dev"
    resend_from_name: str = "StormSentinel AI"

    # Periodic re-check — POST /internal/recheck is hit by a free external
    # cron (e.g. cron-job.org), not Render's own cron (not in the free
    # tier). Protected by a shared secret header, not a user JWT, since no
    # user is logged in when the cron fires.
    cron_secret: str = "CHANGE_THIS_IN_PRODUCTION"
    # Composite-score point change that counts as "significant" enough to
    # notify on, independent of any individual hazard crossing into HIGH.
    significant_change_threshold: int = 15

    # Used to build links back to the app in notification emails.
    frontend_url: str = "http://localhost:3000"

    # CORS — the frontend's origin(s), comma-separated
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

    @property
    def cors_origins_list(self):
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings():
    return Settings()
