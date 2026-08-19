from datetime import datetime

from pydantic import BaseModel


class SavedLocationCreate(BaseModel):
    city: str
    country: str
    lat: float
    lon: float


class SavedLocationOut(BaseModel):
    id: int
    city: str
    country: str
    lat: float
    lon: float
    created_at: datetime

    class Config:
        from_attributes = True


class SavedLocationWithLatest(SavedLocationOut):
    """Adds the most recent prediction for this location, if any exists yet —
    lets the frontend show a live composite score on the saved-locations
    list without a second round-trip per card."""
    latest_composite_score: int | None = None
    latest_queried_at: datetime | None = None


class SnapshotHistoryPoint(BaseModel):
    """One past prediction for a saved location — same `scores` dict shape
    as PredictResponse so the frontend can reuse its hazard-key handling."""
    queried_at: datetime
    scores: dict[str, int]
    composite_score: int
