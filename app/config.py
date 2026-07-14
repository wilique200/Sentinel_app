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

    # External APIs (no keys needed — Open-Meteo, NOAA, FIRMS are free/public)
    firms_map_key: str = ""  # only needed if the backend itself pulls fresh FIRMS data

    # CORS — the frontend's origin(s), comma-separated
    cors_origins: str = "https://stormsentinel-frontend-dun.vercel.app"
    class Config:
        env_file = ".env"

    @property
    def cors_origins_list(self):
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings():
    return Settings()
