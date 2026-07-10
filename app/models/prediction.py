from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class PredictionSnapshot(Base):
    """
    Logs every prediction made, whether for a saved location or an ad-hoc
    search — saved_location_id is nullable so history and 'recent searches'
    both fall out of this one table for free. city/country/lat/lon are
    stored directly (not just via the FK) so history stays intact even if
    the user later un-saves that location.
    """
    __tablename__ = "prediction_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    saved_location_id = Column(Integer, ForeignKey("saved_locations.id", ondelete="SET NULL"), nullable=True)

    city = Column(String, nullable=False)
    country = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    wildfire_score = Column(Float, nullable=False)
    tornado_score = Column(Float, nullable=False)
    hail_score = Column(Float, nullable=False)
    thunderstorm_wind_score = Column(Float, nullable=False)
    flash_flood_score = Column(Float, nullable=False)
    heat_score = Column(Float, nullable=False)
    drought_score = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=False)

    # The model always computes all 7 scores regardless of location — this
    # flag drives the "not validated outside US" caveat on tornado/hail/
    # thunderstorm_wind at display time, rather than the scores themselves
    # being missing (they never are).
    is_us_location = Column(Boolean, nullable=False, default=True)

    queried_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="prediction_snapshots")
    saved_location = relationship("SavedLocation", back_populates="prediction_snapshots")
    chat_messages = relationship("ChatMessage", back_populates="prediction_snapshot", cascade="all, delete-orphan")
