# ============================================================================
# StormSentinel Backend — Periodic Recheck (Cron Endpoint)
# POST /internal/recheck
#
# Meant to be hit by a free external cron (e.g. cron-job.org) on a schedule
# — Render's own cron/worker products aren't in the free tier, so scheduling
# lives outside this app entirely. This endpoint is the whole integration
# surface: point any scheduler at it with the right header and an interval.
#
# Auth is a shared secret header (X-Cron-Secret), not a user JWT — no user
# is logged in when a cron job fires. Compare this to get_current_user in
# app/auth/dependencies.py, which is for user-facing endpoints only.
#
# For every saved location: run a fresh prediction, compare it against that
# location's most recent prior snapshot, and — only if the change is
# significant — compose and send a notification email. Every location still
# gets a new snapshot logged regardless, so history stays complete even for
# non-notified checks.
# ============================================================================

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.config import get_settings
from app.models.location import SavedLocation
from app.models.prediction import PredictionSnapshot
from app.models.user import User
from app.ml.inference import predict_for_location
from app.ml.geocoding import get_region_for_country
from app.ml.model import HEAD_ORDER
from app.llm.notifier import compose_notification_body
from app.notifications.email import send_email

router = APIRouter(prefix="/internal", tags=["cron"])

HIGH_TIER_THRESHOLD = 65  # matches the frontend's HIGH tier boundary (lib/hazards.ts)


class RecheckResult(BaseModel):
    checked: int
    notified: int
    failed: int


def _significant_change(old_scores: dict, new_scores: dict, threshold: int) -> list[str]:
    """Returns the hazard keys whose change is worth surfacing: either a
    swing >= threshold points, or newly crossing into the HIGH tier."""
    changed = []
    for key in HEAD_ORDER:
        old, new = old_scores[key], new_scores[key]
        crossed_high = old <= HIGH_TIER_THRESHOLD < new
        if abs(new - old) >= threshold or crossed_high:
            changed.append(key)
    return changed


@router.post("/recheck", response_model=RecheckResult)
def recheck_saved_locations(x_cron_secret: str = Header(default="")):
    settings = get_settings()
    if not settings.cron_secret or x_cron_secret != settings.cron_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing cron secret")

    db: Session = SessionLocal()
    checked = notified = failed = 0

    try:
        locations = db.query(SavedLocation).all()
        for location in locations:
            checked += 1
            try:
                previous = (
                    db.query(PredictionSnapshot)
                    .filter(PredictionSnapshot.saved_location_id == location.id)
                    .order_by(PredictionSnapshot.queried_at.desc())
                    .first()
                )

                region = get_region_for_country(location.country)
                result = predict_for_location(
                    city=location.city,
                    country_code=location.country,
                    lat=location.lat,
                    lon=location.lon,
                    region=region,
                )
                new_scores = result["scores"]

                snapshot = PredictionSnapshot(
                    user_id=location.user_id,
                    saved_location_id=location.id,
                    city=location.city,
                    country=location.country,
                    lat=location.lat,
                    lon=location.lon,
                    wildfire_score=new_scores["wildfire"],
                    tornado_score=new_scores["tornado"],
                    hail_score=new_scores["hail"],
                    thunderstorm_wind_score=new_scores["thunderstorm_wind"],
                    flash_flood_score=new_scores["flash_flood"],
                    heat_score=new_scores["heat"],
                    drought_score=new_scores["drought"],
                    composite_score=result["composite_score"],
                    is_us_location=result["is_us"],
                )
                db.add(snapshot)
                db.commit()

                # Nothing to compare against yet — this is the baseline for
                # this location. Don't notify off a comparison that doesn't
                # exist; wait for the next check.
                if previous is None:
                    continue

                old_scores = {key: getattr(previous, f"{key}_score") for key in HEAD_ORDER}
                changed_hazards = _significant_change(old_scores, new_scores, settings.significant_change_threshold)
                if not changed_hazards:
                    continue

                user = db.query(User).filter(User.id == location.user_id).first()
                if not user:
                    continue

                body = compose_notification_body(
                    city=location.city,
                    country=location.country,
                    old_scores=old_scores,
                    new_scores=new_scores,
                    changed_hazards=changed_hazards,
                )
                sent = send_email(
                    to=user.email,
                    subject=f"StormSentinel: risk update for {location.city}",
                    body_text=(
                        f"{body}\n\n"
                        f"View the full breakdown: {settings.frontend_url}/dashboard\n\n"
                        f"This is an automated, informational update — not an official "
                        f"emergency alert. For real-time warnings, check your national "
                        f"weather service or local emergency management."
                    ),
                )
                if sent:
                    notified += 1
                else:
                    failed += 1

            except Exception:
                db.rollback()
                failed += 1
                continue

    finally:
        db.close()

    return RecheckResult(checked=checked, notified=notified, failed=failed)
