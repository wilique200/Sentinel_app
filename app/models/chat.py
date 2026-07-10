from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class ChatMessage(Base):
    """
    Every message ties back to a specific prediction_snapshot — the
    assistant is always grounded in a real, specific prediction, never a
    vague 'current conditions' the model never actually computed.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    prediction_snapshot_id = Column(Integer, ForeignKey("prediction_snapshots.id", ondelete="CASCADE"), nullable=False)

    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="chat_messages")
    prediction_snapshot = relationship("PredictionSnapshot", back_populates="chat_messages")
