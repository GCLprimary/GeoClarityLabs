"""
utils/fold_line_resonance.py
============================
Fold Line Resonance Bridge

Facilitates dialogue between the two fundamentally different geometric
processes in the system:

  GENERATED geometry  — Fibonacci golden-spiral lattice (GeometricMemory)
                        Positions derived from golden_angle = π(3 - √5).
                        Static, pre-formed, deterministic.

  ROTATIONAL geometry — Bipolar lattice waypoint angles + spin cycle.
                        Dynamic, never-closing, phase always incrementing
                        by an irrational step (asymmetric_delta).

These two follow the same ruleset but are produced through wildly
different processes. They have been running in parallel with no
communication channel. This module closes that gap.

THE FOLD LINE
─────────────
A fold line is the locus of points where a rotational phase angle
EXACTLY matches a generated (Fibonacci) lattice angle — within a
tolerance derived from asymmetric_delta. These are local intersection
events. From inside either process they look incidental. From outside
both, they are the only moments where the two geometries share a
coordinate — and therefore the only valid communication channel.

Resonance condition:
    |spin_phase - fibonacci_angle[i]| < _FOLD_TOLERANCE

where _FOLD_TOLERANCE = asymmetric_delta (0.01639510239)

This is not arbitrary. asymmetric_delta IS the gap between the two
systems (frame_delta = 2π/3 - effective_boundary). Using it as the
tolerance means: "close enough that the residual gap is smaller than
the system's own fundamental asymmetry." Below that threshold, the
two processes are geometrically indistinguishable at that point —
they have genuinely met.

THE COMMUNICATION SIGNAL
─────────────────────────
At each fold event:
  deviation = |spin_phase - fibonacci_angle[i]|   ∈ [0, _FOLD_TOLERANCE)
  coupling  = 1 - (deviation / _FOLD_TOLERANCE)   ∈ (0, 1]

  coupling → 1.0  : near-perfect resonance, strong imprint
  coupling → 0.0  : grazing contact, weak imprint

The rotational process leaves a phase imprint on the Fibonacci lattice
point. The Fibonacci lattice leaves a structural imprint (its local
density / curvature) on the rotational spin state.

Neither process changes its rules. They mark each other at the fold.

WHAT THIS ENABLES
─────────────────
1. Symbol grouping semantics (next step):
   Symbols whose Fibonacci lattice positions fall near an active fold
   line share a geometric context. That shared context IS the grouping
   criterion — no hardcoded vocabulary needed.

2. Cross-geometry persistence:
   A pattern that survives in BOTH the generated lattice AND the
   rotational record has passed two independent geometric tests.
   That dual survival is the strongest persistence signal the system
   can produce.

3. Resolution signal:
   When fold events cluster (many resonances firing in a short spin
   window), the two geometries are briefly coherent. That coherence
   IS the "security feeling" described in the design notes — resolution
   felt from inside the process.
"""

import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from core.invariants import invariants

# ── Constants ─────────────────────────────────────────────────────────────────
_GOLDEN_ANGLE   = math.pi * (3 - math.sqrt(5))   # Fibonacci lattice step
_FOLD_TOLERANCE = invariants.asymmetric_delta      # 0.01639510239
_SPIN_STEP      = invariants.asymmetric_delta      # irrational relative to 2π
_TWO_PI         = 2 * math.pi

# How many Fibonacci angles to precompute (matches GeometricMemory lattice_points)
_LATTICE_POINTS = 512


class FoldLineResonance:
    """
    Dialogue channel between rotational and generated geometry.

    Maintains:
      - A precomputed table of Fibonacci lattice angles (generated geometry)
      - A running spin phase (rotational geometry)
      - A history of fold events (intersection moments)

    On each tick(), the spin phase advances and is tested against every
    Fibonacci angle. Any match within _FOLD_TOLERANCE is recorded as a
    fold event with its coupling strength and the identities of both
    geometries at that point.
    """

    def __init__(self, lattice_points: int = _LATTICE_POINTS):
        self.lattice_points  = lattice_points
        self.spin_phase      = 0.0          # current rotational phase ∈ [0, 2π)
        self.spin_sign       = +1           # +1 / -1, flips at full cycle
        self.fold_events:    List[Dict[str, Any]] = []
        self.last_tick       = time.time()

        # Coupling accumulator — running mean of recent coupling strengths.
        # High value = the two geometries have been in frequent resonance.
        # This is the "security signal" — coherence between the two processes.
        self.coupling_accumulator = 0.0
        self._coupling_history:   List[float] = []

        # Field resolution state — fed back from the language processor
        # after each sentence. Spin sign decisions are aware of these.
        self._field_persistence  = 0.0   # last known persistence [0, 1]
        self._field_alignment    = 0.0   # last known carry alignment [-1, +1]
        self._field_named_count  = 0     # last known named invariant count
        self._field_carry        = 0.0   # last known net carry magnitude
        self._resolution_history: List[float] = []  # rolling resolution scores

        # Precompute Fibonacci lattice angles
        # These are the polar angles of the golden-spiral lattice points,
        # projected onto [0, 2π) for direct phase comparison.
        self.fibonacci_angles = self._precompute_fibonacci_angles(lattice_points)

        # Per-lattice-point imprint from rotational process.
        # When a fold fires at point i, its imprint accumulates coupling.
        # This is the "mark left on the generated geometry by the rotation."
        self.lattice_imprints = np.zeros(lattice_points, dtype=float)

        # Running spin phase record — the "mark left on the rotation by the
        # generated geometry." Each fold event records the Fibonacci density
        # (local curvature proxy) at the matched point.
        self.spin_imprint_history: List[Tuple[float, float]] = []  # (phase, density)

    # ── Precomputation ────────────────────────────────────────────────────────

    def _precompute_fibonacci_angles(self, n: int) -> np.ndarray:
        """
        Polar angles of the golden-spiral Fibonacci lattice, mapped to [0, 2π).

        The lattice is on a unit sphere. We project azimuthal angles (φ) onto
        the circle — these are the angles directly comparable to spin_phase.
        """
        indices      = np.arange(n)
        phi          = (_GOLDEN_ANGLE * indices) % _TWO_PI   # azimuthal, [0, 2π)
        return phi

    def _local_density(self, idx: int) -> float:
        """
        Proxy for local Fibonacci lattice curvature at point idx.

        Computed as the inverse of the mean angular distance to the two
        nearest neighbours in the precomputed angle table. Higher density
        = the generated geometry is more tightly packed here = stronger
        structural imprint on the rotational process at a fold event.
        """
        n = self.lattice_points
        # Angular distances to immediate neighbours (wrap-safe)
        prev_dist = abs(self.fibonacci_angles[idx] - self.fibonacci_angles[max(0, idx-1)])
        next_dist = abs(self.fibonacci_angles[idx] - self.fibonacci_angles[min(n-1, idx+1)])
        mean_dist = (prev_dist + next_dist) / 2.0
        if mean_dist < 1e-8:
            return 0.0
        return float(np.clip(1.0 / (mean_dist * n), 0.0, 1.0))

    # ── Core tick ─────────────────────────────────────────────────────────────

    def tick(self, external_wave_amp: float = 0.0) -> Dict[str, Any]:
        """
        Advance the spin phase by one step and test for fold events.

        The spin step is modulated by external_wave_amp so the rotational
        process is not isolated from the field — wave energy can accelerate
        or decelerate the phase, creating wave-driven resonance windows.

        Returns a summary of any fold events fired this tick.
        """
        current_time = time.time()
        delta_t      = max(current_time - self.last_tick, 1e-6)
        self.last_tick = current_time

        # Advance spin phase — irrational step ensures never-closing cycle
        # Wave amplitude modulates speed: high energy → faster rotation
        step = _SPIN_STEP * (1.0 + 0.3 * external_wave_amp) * self.spin_sign
        self.spin_phase = (self.spin_phase + step) % _TWO_PI

        # ── Spin sign — field-aware resolution ───────────────────────────────
        # The spin sign should reflect what the field has actually resolved,
        # not just where the phase clock happens to be.
        #
        # POSITIVE spin (+1, vertical build, odds dominant):
        #   Field has genuine structure — high persistence, positive alignment,
        #   named invariants providing stable attractors, carry building.
        #   System is in recognition mode: it knows this territory.
        #
        # NEGATIVE spin (-1, horizontal observer, evens dominant):
        #   Field is uncertain — low persistence, neutral/opposing alignment,
        #   few named invariants, carry near zero.
        #   System is in reconstruction mode: it needs to explore.
        #
        # BOUNDARY (near 0 or π):
        #   Only fires as a genuine transition signal when resolution score
        #   is crossing a threshold, not just because the phase clock hit π.
        #   This makes boundary mode meaningful — it means something changed.
        #
        # Resolution score: weighted combination of field signals.
        # Ranges [0, 1] where 1 = fully resolved, 0 = completely uncertain.
        resolution = self._compute_resolution_score()
        self._resolution_history.append(resolution)
        if len(self._resolution_history) > 16:
            self._resolution_history = self._resolution_history[-16:]

        # Spin sign logic: positive when field is resolved, negative when not.
        # Boundary only fires on genuine crossing (score crosses threshold).
        _RESOLUTION_THRESHOLD = 0.45   # above = recognition, below = reconstruction

        prev_resolution = (
            float(np.mean(self._resolution_history[:-1]))
            if len(self._resolution_history) > 1 else resolution
        )
        crossing = (
            (prev_resolution < _RESOLUTION_THRESHOLD <= resolution) or
            (prev_resolution >= _RESOLUTION_THRESHOLD > resolution)
        )

        if crossing:
            # Genuine transition — boundary mode is meaningful here
            # Don't flip spin_sign yet, let the phase reflect the transition
            pass  # spin_sign stays at current, phase position signals boundary
        elif resolution >= _RESOLUTION_THRESHOLD:
            # Field is genuinely resolved — recognition mode
            self.spin_sign = +1
        else:
            # Field is uncertain — reconstruction mode
            self.spin_sign = -1

        # Phase boundary: only apply the near-π boundary signal when the
        # resolution score is genuinely near the threshold (±0.1 band).
        # This prevents boundary mode from firing purely from phase clock.
        near_threshold = abs(resolution - _RESOLUTION_THRESHOLD) < 0.10
        if not near_threshold:
            # Force phase away from π-boundary zone if resolution is decisive
            # by nudging spin_phase slightly so InvariantEngine doesn't
            # read a false boundary. The phase still cycles naturally —
            # we just suppress the misleading π-proximity signal.
            if resolution > 0.55 and abs(self.spin_phase - math.pi) < _SPIN_STEP * 4:
                self.spin_phase = (self.spin_phase + _SPIN_STEP * 5) % _TWO_PI
            elif resolution < 0.35 and abs(self.spin_phase) < _SPIN_STEP * 4:
                self.spin_phase = (self.spin_phase + _SPIN_STEP * 5) % _TWO_PI

        # ── Fold detection ────────────────────────────────────────────────────
        tick_events = []
        deviations  = np.abs(self.fibonacci_angles - self.spin_phase)
        # Wrap-safe: also check distance through 0
        deviations  = np.minimum(deviations, _TWO_PI - deviations)

        resonant_indices = np.where(deviations < _FOLD_TOLERANCE)[0]

        for idx in resonant_indices:
            deviation = float(deviations[idx])
            coupling  = 1.0 - (deviation / _FOLD_TOLERANCE)   # ∈ (0, 1]

            # Rotational process marks the generated lattice.
            # Accumulation rate 0.15 (was 0.05) — allows imprints to
            # build from the fold events that are already firing correctly
            # without requiring 30+ consecutive hits on the same index.
            self.lattice_imprints[idx] = float(np.clip(
                self.lattice_imprints[idx] * 0.85 + coupling * 0.15,
                0.0, 1.0
            ))

            # Generated lattice marks the rotational process
            density = self._local_density(int(idx))
            self.spin_imprint_history.append((self.spin_phase, density))
            if len(self.spin_imprint_history) > 256:
                self.spin_imprint_history = self.spin_imprint_history[-256:]

            event = {
                "lattice_idx":  int(idx),
                "spin_phase":   round(self.spin_phase, 6),
                "deviation":    round(deviation, 8),
                "coupling":     round(coupling, 4),
                "density":      round(density, 4),
                "spin_sign":    self.spin_sign,
                "timestamp":    current_time,
            }
            tick_events.append(event)
            self.fold_events.append(event)

        # Keep fold history bounded
        if len(self.fold_events) > 1024:
            self.fold_events = self.fold_events[-1024:]

        # Update coupling accumulator — running mean of recent couplings
        if tick_events:
            mean_coupling = float(np.mean([e["coupling"] for e in tick_events]))
            self._coupling_history.append(mean_coupling)
            if len(self._coupling_history) > 64:
                self._coupling_history = self._coupling_history[-64:]
            self.coupling_accumulator = float(np.mean(self._coupling_history))

        return {
            "spin_phase":           round(self.spin_phase, 6),
            "spin_sign":            self.spin_sign,
            "fold_events_this_tick": len(tick_events),
            "fold_events":          tick_events,
            "coupling_accumulator": round(self.coupling_accumulator, 4),
            "total_fold_history":   len(self.fold_events),
        }

    # ── Field resolution awareness ────────────────────────────────────────────

    def update_field_state(
        self,
        persistence:  float,
        alignment:    float,
        named_count:  int,
        carry:        float,
    ) -> None:
        """
        Called by the language processor after each sentence is processed.

        Feeds field resolution quality back into the fold line so spin sign
        decisions are grounded in what the field actually resolved, not just
        where the phase clock is.

        persistence : waveform persistence [0, 1]
        alignment   : carry alignment with prior context [-1, +1]
        named_count : number of named invariants in truth library
        carry       : net carry magnitude (unsigned)
        """
        self._field_persistence = float(persistence)
        self._field_alignment   = float(alignment)
        self._field_named_count = int(named_count)
        self._field_carry       = abs(float(carry))

    def _compute_resolution_score(self) -> float:
        """
        Compute a scalar resolution quality score from current field state.

        Four signals, weighted by their reliability:

          persistence (0.35 weight)
            Direct measure of waveform sustain. Persistence=1.0 means the
            field fully resolved the input. Most reliable single signal.

          alignment (0.30 weight)
            How well this sentence fits prior context. Positive = coherent
            continuation. Negative = contradiction or context break.
            Scaled from [-1,+1] to [0,1].

          named_count (0.20 weight)
            Named invariants represent earned structural knowledge. More
            named words = richer geometric map = more confident resolution.
            Scaled logarithmically (diminishing returns above ~20).

          carry (0.15 weight)
            Net carry magnitude. High carry = field has accumulated context.
            Scaled so typical carry values (0.5-1.2) map to useful range.

        Returns float in [0, 1].
        """
        import math as _math

        # Persistence: already [0, 1]
        p_score = float(np.clip(self._field_persistence, 0.0, 1.0))

        # Alignment: [-1, +1] → [0, 1]
        a_score = float(np.clip((self._field_alignment + 1.0) / 2.0, 0.0, 1.0))

        # Named count: log-scaled, saturates at ~50 named words
        n = max(0, self._field_named_count)
        n_score = float(np.clip(_math.log1p(n) / _math.log1p(50), 0.0, 1.0))

        # Carry: scale so carry=0.5 → ~0.4, carry=1.0 → ~0.6, carry=1.5 → ~0.75
        c_score = float(np.clip(self._field_carry / 1.5, 0.0, 1.0))

        return (
            p_score * 0.35 +
            a_score * 0.30 +
            n_score * 0.20 +
            c_score * 0.15
        )

    def get_resolution_score(self) -> float:
        """Public access to current resolution score."""
        return round(self._compute_resolution_score(), 4)

    # ── Query interface ───────────────────────────────────────────────────────

    def get_imprinted_indices(self, threshold: float = 0.005) -> List[int]:
        """
        Return Fibonacci lattice indices that have received significant
        rotational imprints. These are the points where the two geometries
        have genuinely communicated.

        Used by symbol grouping semantics: symbols whose lattice positions
        fall near these indices share a geometric context.
        """
        return [int(i) for i in np.where(self.lattice_imprints > threshold)[0]]

    def get_active_fold_zone(self) -> Dict[str, Any]:
        """
        Return the current phase window where fold events are most likely.

        Looks back at recent fold events and returns the centroid phase
        and spread — the "active resonance zone" for this spin cycle.
        Useful for predicting where the next strong fold will fire.
        """
        recent = self.fold_events[-32:] if len(self.fold_events) >= 32 else self.fold_events
        if not recent:
            return {"centroid_phase": self.spin_phase, "spread": _TWO_PI, "strength": 0.0}

        phases    = [e["spin_phase"] for e in recent]
        couplings = [e["coupling"]   for e in recent]

        # Circular mean for phase centroid (handles wrap-around correctly)
        sin_mean  = float(np.mean(np.sin(phases)))
        cos_mean  = float(np.mean(np.cos(phases)))
        centroid  = math.atan2(sin_mean, cos_mean) % _TWO_PI
        spread    = float(np.std(phases))
        strength  = float(np.mean(couplings))

        return {
            "centroid_phase": round(centroid, 4),
            "spread":         round(spread, 4),
            "strength":       round(strength, 4),
        }

    def get_coherence_signal(self) -> float:
        """
        The security/resolution signal.

        When fold events cluster — many resonances in a short spin window —
        the two geometries are briefly coherent. This is the geometric
        correlate of "resolution felt from inside the process."

        Returns a value in [0, 1]:
          0.0 = no coherence, two processes are strangers
          1.0 = maximum coherence, full resonance window active
        """
        return round(self.coupling_accumulator, 4)

    def get_status(self) -> Dict[str, Any]:
        imprinted = len(self.get_imprinted_indices(threshold=0.005))
        return {
            "spin_phase":           round(self.spin_phase, 6),
            "spin_sign":            self.spin_sign,
            "total_fold_events":    len(self.fold_events),
            "imprinted_points":     imprinted,
            "coherence_signal":     self.get_coherence_signal(),
            "resolution_score":     self.get_resolution_score(),
            "active_fold_zone":     self.get_active_fold_zone(),
            "fold_tolerance":       _FOLD_TOLERANCE,
        }


# Singleton
fold_line_resonance = FoldLineResonance()
