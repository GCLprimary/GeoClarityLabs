"""
api/session_engine.py
=====================
Session-stateful GCL engine for web deployment.

Each user session gets its own LanguageProcessor instance initialised
from the frozen JSON baseline (ouro_truth_library.json + exhaust_memory.json).
State accumulates within a session — named invariants grow, resolution climbs,
the field warms up — exactly like the local testing experience.

When the session ends (tab close, timeout) the instance is discarded.
Nothing is written to disk during a session.

Usage:
    engine = SessionEngine()          # call once at app startup
    session = engine.new_session()    # one per connected user
    result  = session.process(prompt) # call repeatedly
    engine.end_session(session_id)    # on disconnect / timeout
"""

import time
import threading
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Session timeout — discard idle sessions after this many seconds
SESSION_TIMEOUT_S = 1800  # 30 minutes


@dataclass
class GCLSession:
    session_id:   str
    processor:    Any        # LanguageProcessor instance
    created_at:   float = field(default_factory=time.time)
    last_used:    float = field(default_factory=time.time)
    prompt_count: int   = 0


class SessionEngine:
    """
    Manages per-user LanguageProcessor sessions.
    Thread-safe. Runs a background reaper for idle sessions.
    """

    def __init__(self):
        self._sessions:  Dict[str, GCLSession] = {}
        self._lock       = threading.Lock()
        self._reaper     = threading.Thread(
            target=self._reap_idle_sessions,
            daemon=True, name="session-reaper"
        )
        self._reaper.start()
        print("[session_engine] Initialised — session timeout: "
              f"{SESSION_TIMEOUT_S // 60} min")

    # ── Public API ────────────────────────────────────────────────────────────

    def new_session(self) -> str:
        """
        Create a new session with a fresh LanguageProcessor.
        Returns the session_id to pass back to the client.
        """
        from language.processor import LanguageProcessor
        processor   = LanguageProcessor()
        session_id  = str(uuid.uuid4())
        session     = GCLSession(session_id=session_id, processor=processor)

        with self._lock:
            self._sessions[session_id] = session

        print(f"[session_engine] New session {session_id[:8]}...")
        return session_id

    def process(self, session_id: str, prompt: str) -> Dict[str, Any]:
        """
        Process a prompt in the given session.
        Raises KeyError if session_id is unknown or expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session {session_id} not found or expired")

        session.last_used   = time.time()
        session.prompt_count += 1

        raw = session.processor.process(prompt)
        return _shape_response(raw, session)

    def end_session(self, session_id: str) -> None:
        """Explicitly discard a session (on client disconnect)."""
        with self._lock:
            dropped = self._sessions.pop(session_id, None)
        if dropped:
            print(f"[session_engine] Ended session {session_id[:8]}... "
                  f"({dropped.prompt_count} prompts)")

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ── Background reaper ─────────────────────────────────────────────────────

    def _reap_idle_sessions(self):
        while True:
            time.sleep(60)  # check every minute
            now     = time.time()
            expired = []
            with self._lock:
                for sid, s in self._sessions.items():
                    if now - s.last_used > SESSION_TIMEOUT_S:
                        expired.append(sid)
                for sid in expired:
                    del self._sessions[sid]
            if expired:
                print(f"[session_engine] Reaped {len(expired)} idle session(s)")


# ── Response shaping ──────────────────────────────────────────────────────────

def _shape_response(raw: Dict[str, Any], session: GCLSession) -> Dict[str, Any]:
    """
    Shape the processor.process() return dict into the API response format.
    Keeps only the data the visualiser actually needs.
    """
    fp  = raw.get("fingerprint", {})
    geo = raw.get("geo_output",  {})

    return {
        # ── Core answer ───────────────────────────────────────────────────────
        "geo_output":    geo.get("text", ""),
        "parity_locked": geo.get("parity_locked", False),
        "answer":        raw.get("answer", ""),
        "was_primed":    raw.get("was_primed", False),
        "context_words": raw.get("context_words", []),

        # ── Fingerprint — per-word tensions for visualiser ────────────────────
        "fingerprint": {
            "direction":    fp.get("direction", ""),
            "mean_tension": round(fp.get("mean_tension", 0.0), 4),
            "net_tension":  round(fp.get("net_tension",  0.0), 4),
            "field_stress": round(fp.get("field_stress", 0.0), 4),
            "peak_pair":    fp.get("peak_pair", ""),
            "per_word":     fp.get("per_word", []),
            "top_groups":   fp.get("top_groups", []),
        },

        # ── Candidate words and scores ────────────────────────────────────────
        "geo": {
            "candidates":    geo.get("candidates", []),
            "pocket_scores": geo.get("pocket_scores", []),
            "target_region": geo.get("target_region", {}),
            "field_polarity":round(geo.get("field_polarity", 0.0), 4),
            "resolution":    round(geo.get("resolution",    0.15), 4),
            "template":      geo.get("template", ""),
            "confidence":    geo.get("confidence", ""),
        },

        # ── Session field state ───────────────────────────────────────────────
        "field": {
            "consensus":   round(raw.get("consensus",   0.0), 4),
            "persistence": round(raw.get("persistence", 0.0), 4),
            "gen_mode":    raw.get("gen_mode",    ""),
            "net_carry":   round(raw.get("net_carry",   0.0), 4),
        },

        # ── Session accumulation (grows over conversation) ────────────────────
        "session": {
            "prompt_count": session.prompt_count,
            "vocab_size":   raw.get("vocab_size",   0),
            "vocab_stable": raw.get("vocab_stable", 0),
            "named_count":  raw.get("named_count",  0),
            "newly_named":  raw.get("newly_named",  []),
        },

        "elapsed_ms": round(raw.get("elapsed", 0.0) * 1000, 1),
    }
