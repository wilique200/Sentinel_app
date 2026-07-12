from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class SavedLocation(Base):
    __tablename__ = "saved_locations"
    __table_args__ = (UniqueConstraint("user_id", "city", "country", name="uq_user_city_country"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    city = Column(String, nullable=False)
    country = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="saved_locations")
    prediction_snapshots = relationship("PredictionSnapshot", back_populates="saved_location")
