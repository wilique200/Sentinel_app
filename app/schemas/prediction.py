from pydantic import BaseModel


class LocationQuery(BaseModel):
    query: str


class GeocodeCandidate(BaseModel):
    city: str
    state: str
    country_code: str
    country_name: str
    region: str | None
    lat: float
    lon: float
    is_us: bool
    display_name: str
    disambiguation_label: str
    population: int


class PredictRequest(BaseModel):
    city: str
    country_code: str
    lat: float
    lon: float
    region: str | None = None
    save_snapshot: bool = True  # log to prediction_snapshots; false for quick previews


class HazardWarning(BaseModel):
    type: str
    hazards: list[str]
    message: str


class PredictResponse(BaseModel):
    scores: dict[str, int]
    composite_score: int
    raw_weather: dict[str, float]
    warnings: list[HazardWarning]
    is_us: bool
    snapshot_id: int | None = None
