"""
api/app.py
==========
FastAPI application — session-stateful GCL web API.

Each visitor gets a session that accumulates geometric state
(named invariants, resolution, field warmth) exactly like local testing.
Sessions expire after 30 minutes of inactivity and are discarded.
Nothing is written to disk.

Endpoints:
    POST /session/new       — start a session, get a session_id
    POST /session/{id}/process — process a prompt
    DELETE /session/{id}    — end a session explicitly
    GET  /session/{id}/state — current session field state
    GET  /health            — server status

Deploy to Railway:
    railway up
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ENGINE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ENGINE
    print("[app] Starting GCL session API...")
    from api.session_engine import SessionEngine
    _ENGINE = SessionEngine()
    print("[app] Ready.")
    yield
    print("[app] Shutdown.")


app = FastAPI(
    title="GCL — Geometric Clarity Lab",
    description=(
        "Geometric language field API. Each session starts fresh from "
        "the pre-built geometric baseline and accumulates state during "
        "the conversation. Sessions expire after 30 minutes of inactivity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the demo frontend
_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = _STATIC / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "GCL API running", "docs": "/docs"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Sentence + question for geometric processing.",
        example=(
            "Lightning travels through air by following the path of least "
            "resistance between clouds and the ground. "
            "Why does lightning take zigzag paths?"
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if _ENGINE is None:
        return {"status": "starting"}
    return {
        "status":          "ready",
        "active_sessions": _ENGINE.active_count(),
    }


@app.post("/session/new", summary="Start a new session")
async def new_session():
    """
    Creates a fresh LanguageProcessor session initialised from the
    geometric baseline. Returns a session_id to use in subsequent calls.
    The session accumulates named invariants and field state as you send prompts.
    """
    if _ENGINE is None:
        raise HTTPException(503, "Server is starting up")
    session_id = _ENGINE.new_session()
    return {"session_id": session_id}


@app.post("/session/{session_id}/process", summary="Process a prompt")
async def process(session_id: str, request: PromptRequest):
    """
    Process a prompt in the given session.
    The field state (named invariants, resolution, carry) accumulates
    across calls within the same session.
    """
    if _ENGINE is None:
        raise HTTPException(503, "Server is starting up")
    try:
        result = _ENGINE.process(session_id, request.prompt)
        return result
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found or expired")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/session/{session_id}", summary="End a session")
async def end_session(session_id: str):
    """Explicitly end a session and discard its state."""
    if _ENGINE is None:
        raise HTTPException(503, "Server is starting up")
    _ENGINE.end_session(session_id)
    return {"ended": session_id}


# ── Railway entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)
