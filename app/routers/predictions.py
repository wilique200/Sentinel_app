# ============================================================================
# StormSentinel Backend — Predictions Router
# POST /geocode — search a location, get disambiguation candidates
# POST /predict — run the model for a resolved location (auth required —
#                 every prediction gets logged as a PredictionSnapshot)
# ============================================================================

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.prediction import PredictionSnapshot
from app.schemas.prediction import LocationQuery, GeocodeCandidate, PredictRequest, PredictResponse
from app.ml.geocoding import geocode_location
from app.ml.inference import predict_for_location

router = APIRouter(tags=["predictions"])


@router.post("/geocode", response_model=list[GeocodeCandidate])
def geocode(payload: LocationQuery):
    candidates = geocode_location(payload.query)
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No location found matching '{payload.query}'")
    return candidates


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = predict_for_location(
            city=payload.city,
            country_code=payload.country_code,
            lat=payload.lat,
            lon=payload.lon,
            region=payload.region,
        )
    except ValueError as e:
        # Open-Meteo rejected the request or similar upstream failure —
        # surface the real reason rather than a generic 500.
        raise HTTPException(status_code=502, detail=str(e))

    snapshot_id = None
    if payload.save_snapshot:
        snapshot = PredictionSnapshot(
            user_id=user.id,
            saved_location_id=None,  # set by the locations router if this came from a saved location
            city=payload.city,
            country=payload.country_code,
            lat=payload.lat,
            lon=payload.lon,
            wildfire_score=result["scores"]["wildfire"],
            tornado_score=result["scores"]["tornado"],
            hail_score=result["scores"]["hail"],
            thunderstorm_wind_score=result["scores"]["thunderstorm_wind"],
            flash_flood_score=result["scores"]["flash_flood"],
            heat_score=result["scores"]["heat"],
            drought_score=result["scores"]["drought"],
            composite_score=result["composite_score"],
            is_us_location=result["is_us"],
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        snapshot_id = snapshot.id

    return PredictResponse(**result, snapshot_id=snapshot_id)
