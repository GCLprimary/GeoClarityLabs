"""
core/field_state.py
===================
Persistent geometric field state for GCL.

Saves and restores the full geometric field identity between sessions so
the system continues from where it left off rather than cold-starting at
resolution 0.15 every time.

What persists:
  - fold_line:     spin_phase, resolution_score, imprint state
  - symbol_groups: the dual-pole group structure (the learned geometry)
  - bipolar:       ring phase, global clarity, field stress
  - mobius:        last_spin_phase for T3 delta accuracy
  - carry:         net carry and consensus state
  - conversation:  rolling window of recent exchanges for context priming

What does NOT persist:
  - Individual string sub_factors (too granular, re-derived from geometry)
  - Full fold_events list (can be thousands, not needed for restoration)
  - Per-session vocab (session-specific, intentionally ephemeral)
  - The warmup sweep (runs only if no saved state exists)

Usage:
    from core.field_state import field_state_manager

    # On startup:
    state = field_state_manager.load()
    if state:
        field_state_manager.apply_fold_line(fold_line_resonance, state)
        field_state_manager.apply_mobius(mobius_reader, state)
    else:
        run_warmup()

    # After each prompt:
    field_state_manager.add_exchange(anchor, top_words, net_tension, face, output)

    # On clean exit:
    field_state_manager.save(fold_line, symbol_grouping, bipolar_lattice,
                             mobius_reader, processor)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── File location ─────────────────────────────────────────────────────────────
_STATE_FILE  = Path(__file__).parent.parent / "field_state.json"
_SCHEMA_VER  = "1.0"
_CONV_WINDOW = 12   # max recent exchanges to retain


class FieldStateManager:
    """
    Saves and restores geometric field state between GCL sessions.
    Non-fatal by design: any failure falls back to normal cold-start warmup.
    """

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        fold_line,
        symbol_grouping,
        bipolar_lattice,
        mobius_reader,
        processor,
    ) -> bool:
        """
        Serialize current field state to field_state.json.
        Call on clean exit (quit command).
        Returns True on success.
        """
        try:
            fl_status   = fold_line.get_status()
            bl_status   = bipolar_lattice.get_status()
            grp_summary = symbol_grouping.get_group_summary()
            proc_status = processor.get_status()

            # ── Fold line — geometric oscillation state ────────────────────
            # Save the four resolution inputs directly so restoration is exact
            _lat = getattr(fold_line, "lattice_imprints", None)
            fold_data = {
                "spin_phase":            fl_status.get("spin_phase", 0.0),
                "spin_sign":             fl_status.get("spin_sign", 1),
                "coupling_accumulator":  float(getattr(fold_line, "coupling_accumulator", 0.0)),
                "field_persistence":     float(getattr(fold_line, "_field_persistence", 0.0)),
                "field_alignment":       float(getattr(fold_line, "_field_alignment", 0.0)),
                "field_named_count":     int(getattr(fold_line, "_field_named_count", 0)),
                "field_carry":           float(getattr(fold_line, "_field_carry", 0.0)),
                "resolution_score":      fl_status.get("resolution_score", 0.15),
                "total_fold_events":     fl_status.get("total_fold_events", 0),
                "imprint_sum":           float(getattr(fold_line, "_last_imprint_sum",
                                               fl_status.get("active_fold_zone", {})
                                               .get("strength", 0.0))),
                "imprinted_point_count": fl_status.get("imprinted_points", 0),
                # Lattice imprints array — enables full group restoration
                "lattice_imprints":      [round(float(v), 6) for v in _lat.tolist()]
                                         if _lat is not None else [],
            }

            # ── Symbol groups — dual-pole geometry ────────────────────────
            groups_data = []
            for g in grp_summary:
                # Only save groups with meaningful activation
                if (abs(g.get("base_tension", 0.0)) > 0.001
                        or g.get("tension_centroid", 0.0) > 0.001):
                    groups_data.append({
                        "group_id":         g.get("group_id"),
                        "members":          g.get("members", []),
                        "signed_values":    g.get("signed_values", []),
                        "net_signed_value": g.get("net_signed_value", 0),
                        "tension_centroid": round(g.get("tension_centroid", 0.0), 6),
                        "base_tension":     round(g.get("base_tension", 0.0), 6),
                        "size":             g.get("size", 0),
                    })

            # ── Bipolar lattice — ring/field summary ──────────────────────
            bipolar_data = {
                "ring_net_phase":           round(bl_status.get("ring_net_phase", 0.0), 6),
                "global_clarity":           round(bl_status.get("global_clarity", 0.0), 6),
                "golden_zone_tension":      round(bl_status.get("golden_zone_tension", 0.0), 6),
                "field_stress":             round(bl_status.get("field_stress", 0.5), 6),
                "fold_negotiation_signal":  round(bl_status.get("fold_negotiation_signal", 0.0), 6),
                "total_prompts_this_field": proc_status.get("process_count", 0),
            }

            # ── Möbius reader — T3 delta continuity ──────────────────────
            mobius_data = {
                "last_spin_phase":       round(getattr(mobius_reader, "_last_spin_phase", 0.0), 6),
                "last_convergence_gap":  round(getattr(mobius_reader, "_last_convergence_gap", 0.0), 6),
                "last_pole_offset":      round(getattr(mobius_reader, "_last_pole_offset", 0.0), 6),
                "last_surface_position": round(getattr(mobius_reader, "_last_surface_position", 0.0), 6),
                "last_face":             getattr(mobius_reader, "_last_face", "inner"),
            }

            # ── Carry state ───────────────────────────────────────────────
            carry_data = {
                "net_carry":          round(proc_status.get("net_carry", 0.0), 6),
                "carry_direction":    proc_status.get("carry_direction", "neutral"),
                "active_carry_count": proc_status.get("active_carries", 0),
                "last_consensus":     round(getattr(processor, "_last_consensus", 0.0), 6),
            }

            # ── Conversation window — preserve existing ───────────────────
            conv_data = self._load_conversation_window()
            conv_data["max_window"] = _CONV_WINDOW

            state = {
                "_schema_version":          _SCHEMA_VER,
                "_description":             "GCL geometric field state",
                "_saved_at":                datetime.now(timezone.utc).isoformat(),
                "_session_count":           self._get_session_count() + 1,
                "_total_prompts_processed": (self._get_total_prompts()
                                             + proc_status.get("process_count", 0)),
                "fold_line":    fold_data,
                "symbol_groups": groups_data,
                "bipolar":      bipolar_data,
                "mobius":       mobius_data,
                "carry":        carry_data,
                "conversation": conv_data,
            }

            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            print(f"[field_state] Saved — "
                  f"resolution={fold_data['resolution_score']:.3f}  "
                  f"groups={len(groups_data)}  "
                  f"total_prompts={state['_total_prompts_processed']}  "
                  f"conv={len(conv_data.get('recent_exchanges', []))}")
            return True

        except Exception as e:
            print(f"[field_state] Save failed: {e}")
            return False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load saved field state.
        Returns None if no valid state — caller falls back to normal warmup.
        """
        if not _STATE_FILE.exists():
            return None

        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            if state.get("_schema_version") != _SCHEMA_VER:
                print(f"[field_state] Schema version mismatch — cold start")
                return None

            res      = state.get("fold_line", {}).get("resolution_score", 0.15)
            groups   = len(state.get("symbol_groups", []))
            total    = state.get("_total_prompts_processed", 0)
            sessions = state.get("_session_count", 0)
            saved_at = state.get("_saved_at", "unknown")[:19]

            print(f"[field_state] Loaded — "
                  f"resolution={res:.3f}  "
                  f"groups={groups}  "
                  f"sessions={sessions}  "
                  f"total_prompts={total}  "
                  f"saved={saved_at}")
            return state

        except Exception as e:
            print(f"[field_state] Load failed ({e}) — cold start")
            return None

    # ── Apply state to live objects ───────────────────────────────────────────

    def apply_fold_line(self, fold_line, state: Dict,
                        symbol_grouping=None) -> bool:
        """
        Restore fold line geometric state.
        Pass symbol_grouping to also restore group imprints from lattice data.
        """
        import numpy as np
        try:
            fl = state.get("fold_line", {})
            if not fl:
                return False

            # Core oscillation state
            fold_line.spin_phase           = float(fl.get("spin_phase", 0.0))
            fold_line.spin_sign            = int(fl.get("spin_sign", 1))
            fold_line.coupling_accumulator = float(fl.get("coupling_accumulator", 0.0))

            # Restore the four resolution inputs via update_field_state()
            # This is the correct path — resolution is computed from these,
            # so seeding them exactly reproduces the saved resolution score
            if hasattr(fold_line, "update_field_state"):
                fold_line.update_field_state(
                    persistence = float(fl.get("field_persistence", 0.0)),
                    alignment   = float(fl.get("field_alignment", 0.0)),
                    named_count = int(fl.get("field_named_count", 0)),
                    carry       = float(fl.get("field_carry", 0.0)),
                )
            else:
                # Fallback: set attributes directly
                fold_line._field_persistence  = float(fl.get("field_persistence", 0.0))
                fold_line._field_alignment    = float(fl.get("field_alignment", 0.0))
                fold_line._field_named_count  = int(fl.get("field_named_count", 0))
                fold_line._field_carry        = float(fl.get("field_carry", 0.0))

            # Restore lattice imprints — enables group reconstruction
            lat = fl.get("lattice_imprints", [])
            if lat and hasattr(fold_line, "lattice_imprints"):
                fold_line.lattice_imprints = np.array(lat, dtype=float)
                # Rebuild symbol groups from restored imprints
                if symbol_grouping is not None:
                    try:
                        symbol_grouping._compute_groups()
                        grps = symbol_grouping.get_status().get("imprinted_groups", 0)
                        print(f"[field_state] Groups restored: {grps} active")
                    except Exception as ge:
                        print(f"[field_state] Group restore warning: {ge}")

            res = fold_line.get_resolution_score() if hasattr(fold_line, "get_resolution_score")                   else fl.get("resolution_score", 0.15)
            print(f"[field_state] Fold line restored — "
                  f"phase={fold_line.spin_phase:.4f}  "
                  f"res={res:.3f}  "
                  f"imprints={len(lat)}")
            return True

        except Exception as e:
            print(f"[field_state] Fold line restore failed: {e}")
            return False

    def apply_mobius(self, mobius_reader, state: Dict) -> bool:
        """Restore Möbius reader continuity — fixes T3 delta on session start."""
        try:
            mob = state.get("mobius", {})
            if not mob:
                return False

            mobius_reader._last_spin_phase       = float(mob.get("last_spin_phase", 0.0))
            mobius_reader._last_convergence_gap  = float(mob.get("last_convergence_gap", 0.0))
            mobius_reader._last_pole_offset      = float(mob.get("last_pole_offset", 0.0))
            mobius_reader._last_surface_position = float(mob.get("last_surface_position", 0.0))
            mobius_reader._last_face             = mob.get("last_face", "inner")
            return True

        except Exception as e:
            print(f"[field_state] Möbius restore failed: {e}")
            return False

    # ── Conversation window ───────────────────────────────────────────────────

    def add_exchange(
        self,
        anchor_word: str,
        top_words:   List[str],
        net_tension: float,
        face:        str,
        output:      str,
        candidates:  List[str] = None,
    ) -> None:
        """
        Record a processed exchange in the rolling conversation window.
        Called after each successful prompt.

        candidates: geo_output["candidates"] — geometrically selected words
        from the CURRENT prompt. These are cleaner than output text words
        for conversation priming because output text can contain parity-lock
        artifacts from prior sessions. When provided, candidates replace
        top_words as the primary context source.
        """
        try:
            state = self._load_raw() or self._empty_state()
            conv  = state.setdefault("conversation", {
                "recent_exchanges": [],
                "max_window": _CONV_WINDOW,
            })
            exchanges = conv.setdefault("recent_exchanges", [])

            # Prefer candidates[] (geometrically selected from current prompt)
            # over top_words (which may include parity-lock artifact words).
            # Candidates are the words _sample_vocabulary found relevant to
            # THIS prompt — they don't carry cross-session contamination.
            context_source = (
                [w for w in (candidates or []) if w and len(w) > 2][:8]
                or [w for w in (top_words or []) if w and len(w) > 2][:8]
            )
            exchanges.append({
                "anchor":      anchor_word or "",
                "top_words":   context_source,
                "net_tension": round(float(net_tension), 4),
                "face":        face or "unknown",
                "output":      (output or "")[:120],
                "ts":          datetime.now(timezone.utc).isoformat(),
            })

            # Trim to window
            max_w = conv.get("max_window", _CONV_WINDOW)
            if len(exchanges) > max_w:
                conv["recent_exchanges"] = exchanges[-max_w:]

            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        except Exception:
            pass  # conversation window is best-effort, never fatal

    def get_conversation_window(self) -> List[Dict]:
        """Return recent exchanges. Empty list if none saved."""
        return self._load_conversation_window().get("recent_exchanges", [])

    def get_context_words(self, n: int = 15) -> List[str]:
        """
        Synthesize a pool of high-value context words from recent exchanges.
        Used for question-only input routing (Need 3/2 in build plan).
        Returns the most recent anchor words and top content words,
        deduplicated, ordered by recency.
        """
        exchanges = self.get_conversation_window()
        seen: set = set()
        words: List[str] = []

        for ex in reversed(exchanges):  # most recent first
            anchor = ex.get("anchor", "")
            if anchor and anchor not in seen:
                seen.add(anchor)
                words.append(anchor)
            for w in ex.get("top_words", []):
                if w and w not in seen:
                    seen.add(w)
                    words.append(w)
            if len(words) >= n:
                break

        return words[:n]

    # ── Status / diagnostics ──────────────────────────────────────────────────

    def exists(self) -> bool:
        return _STATE_FILE.exists()

    def summary(self) -> str:
        """One-line summary of saved state."""
        raw = self._load_raw()
        if not raw:
            return "No saved state"
        res      = raw.get("fold_line", {}).get("resolution_score", 0.15)
        total    = raw.get("_total_prompts_processed", 0)
        sessions = raw.get("_session_count", 0)
        groups   = len(raw.get("symbol_groups", []))
        conv     = len(raw.get("conversation", {}).get("recent_exchanges", []))
        return (f"resolution={res:.3f}  sessions={sessions}  "
                f"total_prompts={total}  groups={groups}  conv_window={conv}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self) -> Optional[Dict]:
        if not _STATE_FILE.exists():
            return None
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _empty_state(self) -> Dict:
        return {
            "_schema_version":          _SCHEMA_VER,
            "_saved_at":                datetime.now(timezone.utc).isoformat(),
            "_session_count":           0,
            "_total_prompts_processed": 0,
            "fold_line":    {},
            "symbol_groups": [],
            "bipolar":      {},
            "mobius":       {},
            "carry":        {},
            "conversation": {"recent_exchanges": [], "max_window": _CONV_WINDOW},
        }

    def _load_conversation_window(self) -> Dict:
        raw = self._load_raw()
        if raw:
            return raw.get("conversation", {
                "recent_exchanges": [], "max_window": _CONV_WINDOW
            })
        return {"recent_exchanges": [], "max_window": _CONV_WINDOW}

    def _get_session_count(self) -> int:
        raw = self._load_raw()
        return raw.get("_session_count", 0) if raw else 0

    def _get_total_prompts(self) -> int:
        raw = self._load_raw()
        return raw.get("_total_prompts_processed", 0) if raw else 0


# Module-level singleton
field_state_manager = FieldStateManager()
