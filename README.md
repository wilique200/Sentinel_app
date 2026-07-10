# StormSentinel AI — Backend

FastAPI backend for StormSentinel v2. Wraps the trained 7-head multi-task
PyTorch model, handles auth, and (in progress) will serve predictions,
saved locations, and an LLM-grounded assistant.

## Structure

```
app/
├── main.py              # FastAPI app entry point
├── config.py             # Settings (reads from .env)
├── database.py            # SQLAlchemy engine/session
├── models/                # SQLAlchemy ORM models (users, locations, predictions, chat)
├── schemas/                # Pydantic request/response schemas
├── auth/                    # Password hashing, JWT, get_current_user dependency
├── routers/                  # API route handlers (auth done; predictions/locations/chat next)
├── ml/                         # [next] Model architecture + feature engineering + geocoding
└── llm/                         # [next] Gemini-based assistant, grounded on real predictions
```

## Status

**Done:** database schema, auth (signup/login/JWT), app skeleton.
**Next:** `/predict` endpoint (ports the validated v2 inference logic),
saved locations CRUD, chat assistant.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values
```

You'll need:
- A free Postgres database — [Supabase](https://supabase.com), [Railway](https://railway.app), or [Neon](https://neon.tech) all have free tiers
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card required
- A random JWT secret: `python -c "import secrets; print(secrets.token_hex(32))"`

## Run locally

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the auto-generated interactive API
docs (Swagger UI) — you can test signup/login directly from the browser
without a frontend.

## Model artifacts

Once the `/predict` endpoint is built, this directory will also need the 4
files produced by the training pipeline, copied in alongside `app/`:
- `stormsentinel_model_v2.pt`
- `feature_scaler_v2.pkl`
- `feature_columns_v2.json`
- `climate_normals_v2.json`

## Why these choices

- **FastAPI over Flask/Django**: async-native, auto-generated OpenAPI docs,
  Pydantic validation built in — the right fit for an API-first ML service.
- **JWT over session cookies**: stateless, standard for a separated
  frontend/backend split (Next.js calling this API from a different origin).
- **Gemini over Claude/OpenAI for the assistant**: the only genuinely free,
  indefinite, no-credit-card tier among frontier-quality models as of 2026.
  The assistant module (`app/llm/`) is built behind a provider-agnostic
  interface so this is swappable later without a rewrite.
