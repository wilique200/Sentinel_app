# ============================================================================
# StormSentinel Backend — Application Entry Point
# ============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import engine, Base
from app.routers import auth, predictions

settings = get_settings()

# Creates tables if they don't exist. Fine for a portfolio project — a real
# production app would use Alembic migrations instead of create_all, but
# that's a reasonable v2 upgrade, not a v1 blocker.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="StormSentinel AI API",
    description="Multi-hazard weather risk intelligence — 7-head multi-task model, global coverage.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predictions.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "stormsentinel-api"}
