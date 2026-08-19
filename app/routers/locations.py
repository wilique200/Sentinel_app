# ============================================================================
# StormSentinel Backend — Saved Locations Router
# GET    /locations              — list the user's saved locations, each
#                                   with its most recent prediction if one
#                                   exists (one query, no per-card round trip)
# POST   /locations               — save a location (city/country/lat/lon —
#                                   country is the ISO country_code, same
#                                   convention as PredictionSnapshot.country)
# DELETE /locations/{id}          — remove a saved location
# GET    /locations/{id}/history  — chronological list of past predictions
#                                   for this location, for trend charting
# POST   /locations/{id}/predict  — run a fresh prediction for a saved
#                                   location, linking the resulting snapshot
#                                   back to it via saved_location_id
# ============================================================================

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.location import SavedLocation
from app.models.prediction import PredictionSnapshot
from app.schemas.location import SavedLocationCreate, SavedLocationOut, SavedLocationWithLatest, SnapshotHistoryPoint
from app.schemas.prediction import PredictResponse
from app.ml.inference import predict_for_location
from app.ml.geocoding import get_region_for_country
from app.ml.model import HEAD_ORDER

router = APIRouter(prefix="/locations", tags=["locations"])


@router.get("", response_model=list[SavedLocationWithLatest])
def list_locations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    locations = (
        db.query(SavedLocation)
        .filter(SavedLocation.user_id == user.id)
        .order_by(SavedLocation.created_at.desc())
        .all()
    )

    results = []
    for loc in locations:
        latest = (
            db.query(PredictionSnapshot)
            .filter(PredictionSnapshot.saved_location_id == loc.id)
            .order_by(PredictionSnapshot.queried_at.desc())
            .first()
        )
        results.append(
            SavedLocationWithLatest(
                id=loc.id,
                city=loc.city,
                country=loc.country,
                lat=loc.lat,
                lon=loc.lon,
                created_at=loc.created_at,
                latest_composite_score=round(latest.composite_score) if latest else None,
                latest_queried_at=latest.queried_at if latest else None,
            )
        )
    return results


@router.post("", response_model=SavedLocationOut, status_code=201)
def save_location(
    payload: SavedLocationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    location = SavedLocation(
        user_id=user.id,
        city=payload.city,
        country=payload.country,
        lat=payload.lat,
        lon=payload.lon,
    )
    db.add(location)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="This location is already saved")
    db.refresh(location)
    return location


@router.delete("/{location_id}", status_code=204)
def delete_location(
    location_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    location = (
        db.query(SavedLocation)
        .filter(SavedLocation.id == location_id, SavedLocation.user_id == user.id)
        .first()
    )
    if not location:
        raise HTTPException(status_code=404, detail="Saved location not found")
    db.delete(location)
    db.commit()
    return None


@router.get("/{location_id}/history", response_model=list[SnapshotHistoryPoint])
def get_location_history(
    location_id: int,
    limit: int = 30,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    location = (
        db.query(SavedLocation)
        .filter(SavedLocation.id == location_id, SavedLocation.user_id == user.id)
        .first()
    )
    if not location:
        raise HTTPException(status_code=404, detail="Saved location not found")

    snapshots = (
        db.query(PredictionSnapshot)
        .filter(PredictionSnapshot.saved_location_id == location.id)
        .order_by(PredictionSnapshot.queried_at.desc())
        .limit(limit)
        .all()
    )
    snapshots.reverse()  # chronological order, oldest first — what a chart wants

    return [
        SnapshotHistoryPoint(
            queried_at=s.queried_at,
            scores={key: round(getattr(s, f"{key}_score")) for key in HEAD_ORDER},
            composite_score=round(s.composite_score),
        )
        for s in snapshots
    ]


@router.post("/{location_id}/predict", response_model=PredictResponse)
def predict_for_saved_location(
    location_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    location = (
        db.query(SavedLocation)
        .filter(SavedLocation.id == location_id, SavedLocation.user_id == user.id)
        .first()
    )
    if not location:
        raise HTTPException(status_code=404, detail="Saved location not found")

    region = get_region_for_country(location.country)
    try:
        result = predict_for_location(
            city=location.city,
            country_code=location.country,
            lat=location.lat,
            lon=location.lon,
            region=region,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    snapshot = PredictionSnapshot(
        user_id=user.id,
        saved_location_id=location.id,
        city=location.city,
        country=location.country,
        lat=location.lat,
        lon=location.lon,
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

    return PredictResponse(**result, snapshot_id=snapshot.id)
