# ============================================================================
# StormSentinel Backend — Chat Router
# POST /predictions/{snapshot_id}/chat — send a message, get an assistant
#      reply grounded on that specific snapshot's data
# GET  /predictions/{snapshot_id}/chat — retrieve the conversation history
# Both require auth, and both 404 if the snapshot doesn't belong to the
# requesting user — a snapshot ID alone should never leak another user's
# prediction data.
# ============================================================================

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.prediction import PredictionSnapshot
from app.models.chat import ChatMessage
from app.schemas.chat import ChatSendRequest, ChatMessageOut
from app.llm.assistant import generate_reply

router = APIRouter(prefix="/predictions", tags=["chat"])


def _get_owned_snapshot(snapshot_id: int, user: User, db: Session) -> PredictionSnapshot:
    snapshot = (
        db.query(PredictionSnapshot)
        .filter(PredictionSnapshot.id == snapshot_id, PredictionSnapshot.user_id == user.id)
        .first()
    )
    if not snapshot:
        raise HTTPException(status_code=404, detail="Prediction snapshot not found")
    return snapshot


@router.get("/{snapshot_id}/chat", response_model=list[ChatMessageOut])
def get_chat_history(
    snapshot_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_owned_snapshot(snapshot_id, user, db)
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.prediction_snapshot_id == snapshot_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )


@router.post("/{snapshot_id}/chat", response_model=ChatMessageOut)
def send_chat_message(
    snapshot_id: int,
    payload: ChatSendRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    snapshot = _get_owned_snapshot(snapshot_id, user, db)

    prior_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.prediction_snapshot_id == snapshot_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in prior_messages]

    user_msg = ChatMessage(
        user_id=user.id,
        prediction_snapshot_id=snapshot_id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    db.commit()

    try:
        reply_text = generate_reply(snapshot, history, payload.message)
    except Exception as e:
        # Surface a real reason rather than a generic 500 — same pattern
        # used for the Open-Meteo failure path in the predictions router.
        raise HTTPException(status_code=502, detail=f"Assistant is unavailable right now: {e}")

    assistant_msg = ChatMessage(
        user_id=user.id,
        prediction_snapshot_id=snapshot_id,
        role="assistant",
        content=reply_text,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg
