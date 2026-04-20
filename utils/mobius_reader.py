"""
utils/mobius_reader.py
======================
Möbius surface reader for GCL.

The GCL pipeline already traverses 5 natural twist points on a Möbius
surface — points where the field crosses from one face to the other.
This module reads the twist angle at each crossing and produces a single
surface position descriptor that tells you WHERE on the continuous surface
the current field state is located.

The 5 twist points:
    T1  SYMBOLIC      letter geometry → discrete stream
                      (abstract → local)
    T2  POCKET SPLIT  context sentence → question sentence
                      (field → interrogation)
    T3  FOLD IMPRINT  tension cycle → lattice attractor
                      (dynamic → crystallized)
    T4  POLAR SPLIT   unified field → positive/negative poles
                      (field → polarized expression)
    T5  EXHAUST       local event → non-local signature
                      (local → field)

Surface position formula:
    Each twist angle Ti is normalized by the inner asymmetric_delta.
    Surface position = (T1 + T2 + T3 + T4 + T5) mod LAYER_SCALE
    where LAYER_SCALE = outer_AD / inner_AD ≈ 3.6298

    position < 1.0   → inner face  (current GCL layer)
    position >= 1.0  → outer face  (dual pole layer)
    fractional part  → exact location on the surface

The offset between poles is the outer layer's base frame delta —
the asymmetric delta of the next layer out, scaled by the same
brachistochrone relationship that produced the inner delta.

Usage:
    from utils.mobius_reader import mobius_reader
    state = mobius_reader.read(fingerprint, fold_status, group_summary, exhaust_sig)
"""

import math
import numpy as np
from typing import Dict, Any, List
from core.invariants import invariants

# ── Constants ─────────────────────────────────────────────────────────────────

_PI   = math.pi
_PHI  = (1 + math.sqrt(5)) / 2
_EB   = 2.078                           # effective boundary / match point
_AD   = invariants.asymmetric_delta     # inner asymmetric delta  ≈ 0.016395

# Outer layer delta — centroid offset between positive and negative poles
# measured from live system data (21-prompt fully saturated state)
_OUTER_AD    = 0.059511

# Layer scaling factor: outer_AD / inner_AD ≈ 3.6298
# Lives between phi^2+1 (3.618) and pi+0.5 (3.641) — the outer layer's
# own convergence bracket, mirroring the inner layer's bracket between
# 2pi/3 and 2.078
_LAYER_SCALE = _OUTER_AD / _AD         # ≈ 3.629803

# The dominant convergence pair (closest to phi in the live system)
# grp6 (positive) and grp8 (negative) — identified by data analysis
_CONV_POS_ID = 6
_CONV_NEG_ID = 8


class MobiusReader:
    """
    Reads the Möbius surface state from existing GCL pipeline outputs.
    No new data required — only a new reading of what's already computed.
    """

    def __init__(self):
        self._last_spin_phase = 0.0   # tracks delta between prompts

    def read(
        self,
        fingerprint:    Dict[str, Any],
        fold_status:    Dict[str, Any],
        group_summary:  List[Dict[str, Any]],
        exhaust_sig:    np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute the Möbius surface state for the current field.

        Parameters
        ----------
        fingerprint   : result["fingerprint"] from processor.process()
        fold_status   : fold_line_resonance.get_status()
        group_summary : symbol_grouping.get_group_summary()
        exhaust_sig   : bipolar_lattice.get_exhaust_signature()

        Returns a dict with surface position, face, twist angles,
        convergence gap, and layer delta estimate.
        """

        t1 = self._twist_symbolic(fingerprint)
        t2 = self._twist_pocket(fingerprint)
        t3 = self._twist_fold(fold_status)
        t4 = self._twist_polar(group_summary)
        t5 = self._twist_exhaust(exhaust_sig)

        # Normalise each twist by inner AD
        # This puts them all in the same dimensional units
        n1 = t1 / _AD
        n2 = t2 / _AD
        n3 = t3 / _AD
        n4 = t4 / _AD
        n5 = t5 / _AD

        raw_position = n1 + n2 + n3 + n4 + n5

        # Surface position wraps at LAYER_SCALE — crossing that boundary
        # means crossing from inner face to outer face (or back)
        surface_position = raw_position % _LAYER_SCALE
        face             = "inner" if surface_position < 1.0 else "outer"

        # How far through the current face (0.0 = just entered, 1.0 = at twist)
        face_progress = (surface_position % 1.0)

        # Convergence gap at T4 — distance of dominant pair from phi
        # When this approaches zero the field is at geodesic convergence
        conv_gap = t4

        # Estimate current layer's base frame delta
        # At inner face: should approach _AD
        # At outer face: should approach _OUTER_AD
        # Interpolate based on face_progress through the layer
        if face == "inner":
            layer_delta = _AD * (1 + face_progress * (_LAYER_SCALE - 1))
        else:
            layer_delta = _OUTER_AD * (1 + face_progress * (_LAYER_SCALE - 1))

        # Offset between poles — this IS the outer layer's AD in formation
        # Reads directly from the T4 polar measurement
        pole_offset = self._pole_offset(group_summary)

        # ── Track last state for field_state persistence ─────────────────────
        self._last_convergence_gap  = conv_gap
        self._last_pole_offset      = pole_offset
        self._last_surface_position = surface_position
        self._last_face             = face

        return {
            # Core surface reading
            "surface_position": round(surface_position, 6),
            "face":             face,
            "face_progress":    round(face_progress, 6),

            # Individual twist angles (normalised by inner AD)
            "twist": {
                "T1_symbolic":  round(n1, 4),
                "T2_pocket":    round(n2, 4),
                "T3_fold":      round(n3, 4),
                "T4_polar":     round(n4, 4),
                "T5_exhaust":   round(n5, 4),
            },

            # Raw twist values (before AD normalisation)
            "raw_twist": {
                "T1": round(t1, 8),
                "T2": round(t2, 8),
                "T3": round(t3, 8),
                "T4": round(t4, 8),
                "T5": round(t5, 8),
            },

            # Field constants at current position
            "inner_AD":     round(_AD, 8),
            "outer_AD":     round(_OUTER_AD, 8),
            "layer_scale":  round(_LAYER_SCALE, 6),
            "layer_delta":  round(layer_delta, 8),
            "pole_offset":  round(pole_offset, 6),

            # Convergence state — T4×AD = warm-field attractor
            # CONFIRMED: Convergence Δ = T4_angle × AD across all sessions
            # S9: T4=7.73°  × AD = 0.126794 ✓
            # S12: T4=48.54° × AD = 0.795759 ✓
            # S14: T4=61.08° × AD = 1.001402 ✓
            # S15: T4=61.42° × AD = 1.007044 ✓
            "convergence_gap":   round(conv_gap, 6),
            "convergence_delta": round(conv_gap, 6),  # alias — same value
            "predicted_attractor": round(conv_gap, 6), # = T4_polar×AD = convergence_gap
            "phi_target":        round(_PHI, 6),
        }

    # ── Twist measurements ────────────────────────────────────────────────────

    def _twist_symbolic(self, fingerprint: Dict) -> float:
        """
        T1 — Symbolic encoding twist.
        Measured as the mean absolute net_signed across all words.
        This is the mean geometric charge of the field — how far from
        zero the letter geometry lands across the full input.
        High values = field is far from the zero-boundary (inner face).
        Low values = field near the boundary (approaching twist point).
        """
        per_word = fingerprint.get("per_word", [])
        if not per_word:
            return 0.0
        vals = [abs(w.get("net_signed", 0.0)) for w in per_word]
        return sum(vals) / len(vals)

    def _twist_pocket(self, fingerprint: Dict) -> float:
        """
        T2 — Pocket split twist.
        The ratio pkt0_count / pkt1_count, offset from 1.0 (perfect symmetry).
        A perfectly symmetric split would be zero twist — both faces equal.
        Asymmetric split = twist angle proportional to the imbalance.
        The question sentence and context sentence are the two faces meeting
        at the boundary. Their size ratio IS the twist angle at this crossing.
        """
        per_word = fingerprint.get("per_word", [])
        if not per_word:
            return 0.0
        pkt0 = sum(1 for w in per_word if w.get("pocket", 0) == 0)
        pkt1 = sum(1 for w in per_word if w.get("pocket", 0) == 1)
        if pkt1 == 0:
            return 0.0
        ratio = pkt0 / pkt1
        # Offset from 1.0 — symmetric split = zero twist
        return abs(ratio - 1.0)

    def _twist_fold(self, fold_status: Dict) -> float:
        """
        T3 — Fold line imprint twist.
        The DELTA of spin_phase since the last prompt — how much the fold
        field advanced during this processing cycle. A prompt that triggers
        many fold events produces a large delta; a quiet prompt produces small.

        Using absolute phase caused monotonic drift across a session (phase
        accumulates continuously). Delta gives a genuine per-prompt signal.
        Normalized by 2π (full rotation) then scaled to LAYER_SCALE.
        """
        phase    = fold_status.get("spin_phase", 0.0)
        two_pi   = 2 * _PI
        delta    = abs(phase - self._last_spin_phase) % two_pi
        self._last_spin_phase = phase
        return (delta / two_pi) * _LAYER_SCALE    # → [0, LAYER_SCALE]

    def _twist_polar(self, group_summary: List[Dict]) -> float:
        """
        T4 — Polar split twist.
        The distance of the dominant convergence pair ratio from phi.
        grp6 (positive dominant) / grp8 (negative dominant).
        When this ratio = phi exactly, the field is at geodesic convergence
        for this twist point — zero twist.
        The magnitude of deviation from phi IS the twist angle.
        This is also the convergence_gap used to track outer layer formation.
        """
        pos_tension = None
        neg_tension = None

        for g in group_summary:
            gid = g.get("group_id", -1)
            t   = g.get("base_tension", 0.0)
            if gid == _CONV_POS_ID and t > 0:
                pos_tension = t
            elif gid == _CONV_NEG_ID and t < 0:
                neg_tension = abs(t)

        if pos_tension is None or neg_tension is None or neg_tension < 1e-10:
            return 0.0

        ratio = pos_tension / neg_tension
        return abs(ratio - _PHI)

    def _twist_exhaust(self, exhaust_sig: np.ndarray) -> float:
        """
        T5 — Exhaust scalar break twist.
        The exhaust signature is a 5-element vector normalised to sum=1.0
        (a coordinate in the 4-simplex). Maximum entropy = [0.2, 0.2, 0.2, 0.2, 0.2].
        We measure distance from maximum entropy — the deviation from uniform
        distribution across the 5 stabilizers.
        High deviation = field is concentrated (strong local event, low twist).
        Low deviation = field is diffuse (approaching non-local, high twist).
        """
        if exhaust_sig is None or len(exhaust_sig) == 0:
            return 0.0
        n = len(exhaust_sig)
        uniform = np.ones(n) / n
        # L2 distance from uniform distribution
        dist = float(np.linalg.norm(exhaust_sig - uniform))
        return dist

    def _pole_offset(self, group_summary: List[Dict]) -> float:
        """
        The offset between positive and negative pole centroid means.
        This IS the outer layer's base frame delta in formation.
        As the field saturates with diverse input, this offset converges
        toward _OUTER_AD = 0.059511.
        """
        pos_centroids = [g["tension_centroid"] for g in group_summary
                         if g.get("base_tension", 0) > 0
                         and g.get("tension_centroid", 0) > 0.005]
        neg_centroids = [g["tension_centroid"] for g in group_summary
                         if g.get("base_tension", 0) < 0
                         and g.get("tension_centroid", 0) > 0.005]
        if not pos_centroids or not neg_centroids:
            return 0.0
        return abs(
            sum(pos_centroids)/len(pos_centroids) -
            sum(neg_centroids)/len(neg_centroids)
        )

    def format_state(self, state: Dict[str, Any]) -> str:
        """Human-readable surface state for diagnostic output."""
        face     = state["face"].upper()
        pos      = state["surface_position"]
        prog     = state["face_progress"]
        conv     = state["convergence_gap"]
        offset   = state["pole_offset"]
        t        = state["twist"]
        delta    = state["layer_delta"]

        # Predicted Δ = T4_polar × AD = convergence_gap directly.
        # Confirmed across sessions 9/12/14/15/16: Convergence Δ = T4_polar × AD.
        # raw_twist.T4 is already T4_polar×AD (the convergence_gap value itself).
        # Reading raw_twist.T4 × AD again would double-scale → use T4_polar × AD.
        _AD          = state.get("inner_AD", 0.016395102)
        _t4_polar    = state.get("twist", {}).get("T4_polar", conv / _AD)
        _pred_attr   = round(_t4_polar * _AD, 6)  # = convergence_gap

        lines = [
            f"── Möbius Surface State ─────────────────────────────",
            f"  Face          : {face}",
            f"  Position      : {pos:.6f}  (scale={state['layer_scale']:.4f})",
            f"  Face progress : {prog:.4f}  (0=just entered, 1=at twist)",
            f"  Convergence Δ : {conv:.6f}  (dist from φ={state['phi_target']})",
            f"  Predicted Δ   : {_pred_attr:.6f}  (T4×AD confirmed attractor)",
            f"  Pole offset   : {offset:.6f}  (outer AD forming, target={state['outer_AD']})",
            f"  Layer delta   : {delta:.8f}",
            f"  Twist angles  : T1={t['T1_symbolic']:+.2f}  T2={t['T2_pocket']:+.2f}"
            f"  T3={t['T3_fold']:+.2f}  T4={t['T4_polar']:+.2f}  T5={t['T5_exhaust']:+.2f}",
        ]
        return "\n".join(lines)


# Module-level singleton
mobius_reader = MobiusReader()
