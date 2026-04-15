"""
utils/bipolar_lattice.py
========================
Bipolar Lattice — 18 + 8 + 27 + 52 architecture

Structure:
  18 outer waypoints  — 9 positive tension amplifiers + 9 negative compression sinks
   8 structural waypoints — powers-of-2 scaffolding (square-triangle backbone)
  27 symbol ring      — outer crescent (A–Z + dynamic 0), 5 geometric stabilizers
  52 Mersenne strings — fold negotiators, 26 per polarity side

Golden zone — the innermost attractor region (structural waypoints at radius 0.35).
When semantic_tension is low (field is calm and resolved), the core election fires
and the highest-scoring waypoint in the golden zone becomes the active centre.

inject_semantic_tension() maps subject/verb/object hash IDs onto the Mersenne
strings using the odd/even vertical/horizontal asymmetry from the design notes:
  odd subject  → positive tension (+0.8)
  even verb    → negative compression (-0.6)
  object mod 3 → constructive interference (+0.4)

Self-selecting core: no displacer is born into the centre. Objective physics only.

Geometric roles (v3 — derived from sphere geometry and fold point theory):

  5 Geometric Stabilizers (formerly "vents"):
    The 5 triangle positions on the 27-symbol ring are the weakest points on
    the sphere — where the geometry will break first under stress. Their role
    is structural pressure relief: when total field tension approaches the
    break threshold, these points bleed off the excess proportionally.
    They are NOT fold points. They are safety geometry — the tissue paper
    tearing at its weakest seam before the whole sheet goes.

  52 Mersenne Fold Negotiators (formerly "tensegrity waveguides"):
    Each string runs its own incommensurable negotiation cycle, tracked via
    fold_phase. The Mersenne prime basis ensures no two strings share a period —
    their cycles never lock into the same rhythm, which is what prevents the
    lattice from going rigid (copy paper behavior). The bleed is not waste —
    it is the string finding its local equilibrium, the same way tissue paper
    finds the spiral under a string twist. Aggregate of 52 independent
    negotiations produces the emergent tensegrity field.

Spin State (v2):
  Every waypoint carries a signed rotational phase — spin_phase ∈ [0, 2π),
  spin_sign ∈ {+1, -1}.

  spin_sign assignment at construction (from zero dynamism design):
    odd  wp_id → +1  (builds vertically)
    even wp_id → -1  (recognizes horizontally)

  Each tension cycle advances spin_phase by asymmetric_delta × spin_sign.
  The cycle never closes perfectly — that residual is the arrow of time in
  the rotation domain. Spin direction survives tanh saturation because it is
  positional, not amplitude-based.

  The 8 structural (golden zone) waypoints maintain a ring net_phase sum.
  When that sum crosses a full 2π increment, one geometric tick fires —
  replacing wall-clock delta_t as the system's internal time measure.

  Core election gains a spin coherence bonus: candidates whose spin_phase
  aligns with the ring's net phase score higher, so the elected core
  is the node most in-phase with the field's current rotational state.
"""

import os
import json
import math
import time
import numpy as np
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from core.invariants import invariants
from core.clarity_ratio import clarity_ratio
from core.safeguards import safeguards

# ── Mersenne waveguide constants ──────────────────────────────────────────────
_MERSENNE_PRIMES  = [3, 7, 31, 127]
_N_STRINGS        = 52
_STRINGS_PER_SIDE = _N_STRINGS // 2
_SUBTRACTION_BASE = invariants.asymmetric_delta   # 0.01639510239

# ── Waypoint counts ───────────────────────────────────────────────────────────
_N_POSITIVE    = 9
_N_NEGATIVE    = 9
_N_STRUCTURAL  = 8
_N_SYMBOL_RING = 27

# ── Thresholds ────────────────────────────────────────────────────────────────
_CORE_MIN_CLARITY    = 0.12
_PRUNE_AMP_MIN       = 0.004
_MAX_TENSION         = 2.0

# Exhaust memory persistence — grows across sessions like the truth library
_EXHAUST_MEMORY_FILE = "exhaust_memory.json"

# ── Spin constants — fully derived ───────────────────────────────────────────
# Phase step per tension cycle.
# asymmetric_delta alone (≈ 0.0164) produces one ring rotation per ~383 ticks —
# too slow for session-scale observation. Scaling by π gives ≈ 0.0515 per step,
# one full rotation per ~122 ticks (~10-15 prompts). Still derived from the same
# base constant, just at the correct operational scale.
# Same derivation as the fold point: π is the natural period scalar.
_SPIN_STEP           = invariants.asymmetric_delta * math.pi  # ≈ 0.05151
_TWO_PI              = 2.0 * math.pi
# Coherence window scales with the new step size
_SPIN_COHERENCE_WIN  = _SPIN_STEP * math.pi                   # ≈ 0.16180
# Coherence bonus multiplier on core_score during election
_SPIN_COHERENCE_BONUS = 1.15


# ── Stabilizer threshold — derived from max tension ──────────────────────────
# Field stress above this fraction of _MAX_TENSION triggers geometric
# stabilizers. Derived: effective_boundary / pi ≈ 0.661 — the same ratio
# that defines the fold point in the manifold.
_STABILIZER_THRESHOLD = 2.078 / math.pi   # ≈ 0.661


class MersenneString:
    """
    Fold negotiator — each string runs its own incommensurable negotiation cycle.

    The Mersenne prime basis ensures no two strings share a period. Their
    fold_phase cycles never lock into the same rhythm — this is what prevents
    the lattice from going rigid. The bleed is not waste: it is the string
    finding its local equilibrium, the same way tissue paper finds the spiral
    under a string twist.

    fold_phase advances by (asymmetric_delta / mersenne) each tick — smaller
    Mersenne primes step faster, larger ones step slower. The incommensurability
    of these rates across all 52 strings is what produces the emergent
    tensegrity field.
    """

    def __init__(self, idx: int, polarity: int):
        self.idx        = idx
        self.polarity   = polarity
        self.mersenne   = _MERSENNE_PRIMES[idx % len(_MERSENNE_PRIMES)]
        self.sub_factor = _SUBTRACTION_BASE / self.mersenne
        self.tension    = 0.0
        self.active     = True
        # Fold phase — incommensurable across strings, never locks
        # Seed from idx so strings start distributed rather than all at zero
        self.fold_phase = (idx * _SUBTRACTION_BASE) % _TWO_PI

    def tick(self, incoming_tension: float) -> float:
        if not self.active:
            return 0.0
        # Advance fold phase — this string's position in its negotiation cycle
        self.fold_phase = (self.fold_phase + self.sub_factor) % _TWO_PI
        # Tension negotiation: fold_phase modulates how much tension this
        # string absorbs vs passes through. At fold_phase ≈ π the string
        # is at maximum negotiation capacity — it absorbs more.
        fold_factor     = 0.5 * (1.0 + math.sin(self.fold_phase))  # [0, 1]
        self.tension    = incoming_tension * self.polarity
        bleed           = self.sub_factor * abs(self.tension) * (1.0 + fold_factor)
        remainder       = max(0.0, abs(self.tension) - bleed) * self.polarity
        self.tension    = remainder
        return remainder

    def should_prune(self) -> bool:
        return abs(self.tension) < _PRUNE_AMP_MIN


class Waypoint:
    """
    Single node in the bipolar lattice.

    Spin state:
      spin_sign  — +1 (odd wp_id, vertical builder) or -1 (even wp_id, horizontal recognizer)
      spin_phase — accumulated rotational position in [0, 2π), advanced by
                   asymmetric_delta × spin_sign each tension cycle.
                   Survives amplitude saturation — purely positional.
    """

    def __init__(
        self,
        wp_id:  int,
        role:   str,
        angle:  float,
        radius: float,
        symbol: Optional[str] = None,
    ):
        self.wp_id                = wp_id
        self.role                 = role
        self.angle                = angle
        self.radius               = radius
        self.symbol               = symbol
        self.tension_scalar       = 1.0
        self.clarity_contribution = 0.0
        self.persistence          = 0.0
        self.is_core              = False
        self.birth_time           = time.time()
        self.local_wave_amp       = 0.0

        # ── Spin state ────────────────────────────────────────────────────────
        # Handedness from wp_id parity — zero dynamism: odds build vertically,
        # evens recognize horizontally.
        self.spin_sign  = +1 if (wp_id % 2 == 1) else -1
        # Initial phase seeded from construction angle so waypoints start
        # spread around the ring rather than all at zero. This prevents
        # a spurious geometric tick on the very first cycle.
        self.spin_phase = angle % _TWO_PI

        # ── Dual-13 integer assignment ────────────────────────────────────────
        # Every waypoint carries its signed integer from the dual-13 system.
        # Symbol waypoints get their integer from the letter_to_int table.
        # Non-symbol waypoints get their integer from wp_id mapped into ±1..13.
        # Zero-symbol waypoints carry dual value — resolved at runtime from
        # the field's spin signal via invariants.symbol_to_int('0', spin).
        if symbol is not None:
            self.dual_int        = invariants.letter_to_int.get(symbol.upper(), 0)
            self.is_dynamic_zero = (symbol == '0')
        else:
            # Non-symbol waypoints: map wp_id into ±1..13
            # Odd → positive side, even → negative side — same parity rule
            raw           = (wp_id % 13) + 1       # 1..13
            self.dual_int = raw if (wp_id % 2 == 1) else -raw
            self.is_dynamic_zero = False

        # ── Exhaust tracking (stabilizers only) ───────────────────────────────
        # Tracks how much each stabilizer bleeds each cycle.
        # The pattern across all 5 stabilizers is the exhaust signature —
        # a geometric fingerprint of the input that caused the stress.
        self.bleed_total = 0.0   # cumulative bleed this prompt
        self.bleed_last  = 0.0   # bleed amount last cycle
        self.bleed_rate  = 0.0   # rate of change between last two cycles
        self._bleed_prev = 0.0   # previous cycle for rate calculation

    @property
    def position(self) -> Tuple[float, float]:
        return (self.radius * math.cos(self.angle),
                self.radius * math.sin(self.angle))

    @property
    def core_score(self) -> float:
        return self.clarity_contribution * self.persistence

    def resolved_int(self, spin_signal: float = 0.0) -> int:
        """
        Return the resolved dual-13 integer for this waypoint.

        For dynamic zero waypoints: resolves from spin_signal at call time.
          positive spin → +1  (field committed to vertical build)
          negative spin → -1  (field committed to horizontal recognize)
          zero spin     →  0  (dual held — displacement engine braking)

        For all others: returns fixed dual_int.

        This is the core braking mechanism — when zero is unresolved (spin=0)
        the dual value is held and movement in either direction is possible.
        When resolved, the brake is released on one side and applied on the other.
        """
        if self.is_dynamic_zero:
            return invariants.symbol_to_int('0', spin_signal)
        return self.dual_int

    def advance_spin(self) -> float:
        """
        Advance rotational phase — odds only.

        Odd wp_id (spin_sign +1) = builder, actively spinning.
          Phase advances by asymmetric_delta each tick.
          Generates the directed arrow through movement.

        Even wp_id (spin_sign -1) = recognizer, static.
          Phase held at construction angle — fixed reference point.
          Does not move. Observes. The dual-held position.

        The geometric tick fires when the spinning half (odds) has
        displaced 2π relative to the static half (evens) — the builders
        completing a full cycle relative to the recognizers.
        """
        if self.spin_sign == +1:
            # Odd — builder — actively spinning
            self.spin_phase = (self.spin_phase + _SPIN_STEP) % _TWO_PI
        # Even — recognizer — phase held, no advancement
        return self.spin_phase

    def spin_coherence(self, ring_net_phase: float) -> float:
        """
        Measure how closely this waypoint's spin_phase aligns with the
        structural ring's net phase. Returns value in [0, 1].
        1.0 = perfectly in phase, 0.0 = anti-phase.
        Used to modulate core election score.
        """
        delta = abs(self.spin_phase - ring_net_phase) % _TWO_PI
        # Wrap to [0, π] — alignment is symmetric around the phase circle
        if delta > math.pi:
            delta = _TWO_PI - delta
        # Map [0, π] → [1, 0]
        return 1.0 - (delta / math.pi)

    def update(self, wave_amp: float, delta_t: float) -> None:
        self.local_wave_amp  = wave_amp
        decay                = math.exp(-0.05 * delta_t)
        self.persistence     = float(np.clip(
            self.persistence * decay + wave_amp * (1 - decay), 0.0, 1.0
        ))
        cr = clarity_ratio.current_ratio
        self.clarity_contribution = float(np.clip(
            self.clarity_contribution * 0.95 + cr * wave_amp * 0.05, 0.0, 2.0
        ))


class BipolarLattice:
    """
    Full 18+8+27+52 bipolar lattice with golden zone, semantic tension injection,
    and geometric spin tracking.

    Geometric tick:
      The 8 structural waypoints form a spin ring. Each tension cycle advances
      every structural waypoint's spin_phase. The ring net_phase is the mean
      of all structural spin_phases. When the net_phase accumulates a full 2π
      increment since the last tick, one geometric_tick fires. This is the
      system's internal time unit — derived from field energy, not wall clock.
    """

    def __init__(self):
        self.waypoints:       List[Waypoint]       = []
        self.strings:         List[MersenneString] = []
        self.core_id:         Optional[int]        = None
        self.last_tick        = time.time()
        self._current_prompt  = ""

        # ── Geometric tick state ──────────────────────────────────────────────
        self.geometric_tick_count  = 0
        self._last_ring_net_phase  = 0.0

        # ── Exhaust memory — content-addressable geometric fingerprints ───────
        # Each entry: {signature, prompt, symbol_stream, core_id, ring_phase}
        # Persists across sessions via exhaust_memory.json
        self.exhaust_memory: List[Dict] = []
        self._load_exhaust_memory()

        # Golden zone
        self.golden_zone = {
            "center_id":        None,
            "radius_factor":    0.35,
            "semantic_tension": 0.0,
            "zero_placeholder": True,
        }

        self._build_lattice()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_lattice(self) -> None:
        wp_id = 0

        # 1. Positive tension amplifiers
        for i in range(_N_POSITIVE):
            angle  = (i / _N_POSITIVE) * 2 * math.pi
            self.waypoints.append(Waypoint(wp_id, "positive", angle, 0.55))
            wp_id += 1

        # 2. Negative compression sinks
        for i in range(_N_NEGATIVE):
            angle  = ((i + 0.5) / _N_NEGATIVE) * 2 * math.pi
            self.waypoints.append(Waypoint(wp_id, "negative", angle, 0.85))
            wp_id += 1

        # 3. Structural backbone — 8 phase window owners
        # Each structural waypoint is positioned at the CENTER of its phase
        # window — exactly 1/8 of 2π apart, no offset.
        # wp 18 owns window 0, wp 19 owns window 2π/8, etc.
        # The ring_net_phase determines which waypoint is eligible for
        # core election — the one whose window contains the current phase.
        # This gives each structural waypoint a distinct geometric role
        # rather than all competing from arbitrary construction positions.
        _PHASE_WINDOW = _TWO_PI / _N_STRUCTURAL   # exactly π/4 per window
        for i in range(_N_STRUCTURAL):
            angle = (i / _N_STRUCTURAL) * _TWO_PI  # center of window i
            self.waypoints.append(Waypoint(wp_id, "structural", angle, 0.35))
            wp_id += 1

        # 4. 27-symbol outer crescent ring (5 geometric stabilizers at triangle positions)
        symbols      = ['0'] + [chr(ord('A') + i) for i in range(26)]
        vent_indices = {0, 5, 10, 16, 21}
        for i, sym in enumerate(symbols):
            # Triangle positions are weakest sphere geometry — structural stabilizers
            role  = "stabilizer" if i in vent_indices else "symbol"
            angle = (i / _N_SYMBOL_RING) * 2 * math.pi
            self.waypoints.append(Waypoint(wp_id, role, angle, 1.2, symbol=sym))
            wp_id += 1

        # 5. Mersenne strings
        for i in range(_STRINGS_PER_SIDE):
            self.strings.append(MersenneString(i, polarity=+1))
        for i in range(_STRINGS_PER_SIDE):
            self.strings.append(MersenneString(i, polarity=-1))

        # Seed the initial ring net_phase from construction state
        self._last_ring_net_phase = self._ring_net_phase()

    # ── Spin ring ─────────────────────────────────────────────────────────────

    def _structural_waypoints(self) -> List[Waypoint]:
        return [wp for wp in self.waypoints if wp.role == "structural"]

    def _ring_net_phase(self) -> float:
        """
        Mean spin_phase of all structural (golden zone) waypoints.
        This is the ring's current rotational position — one number that
        represents where the inner backbone is in its cycle.
        """
        structural = self._structural_waypoints()
        if not structural:
            return 0.0
        # Circular mean — handles wrap-around correctly
        sin_sum = sum(math.sin(wp.spin_phase) for wp in structural)
        cos_sum = sum(math.cos(wp.spin_phase) for wp in structural)
        return math.atan2(sin_sum, cos_sum) % _TWO_PI

    def _advance_spin_ring(self) -> bool:
        """
        Advance spin_phase for ALL waypoints by one step.
        Check if the structural ring's net_phase has completed a full 2π
        increment since the last geometric tick.
        Returns True if a geometric tick fired this cycle.
        """
        for wp in self.waypoints:
            wp.advance_spin()

        new_net = self._ring_net_phase()

        # Angular distance traveled since last tick (signed, mod 2π)
        delta = (new_net - self._last_ring_net_phase) % _TWO_PI

        # A geometric tick fires when accumulated phase crosses 2π.
        # Because _SPIN_STEP is small, this takes many cycles — that's
        # correct. The tick rate reflects field energy, not wall time.
        # We track cumulative displacement rather than a single-step check.
        if not hasattr(self, '_accumulated_phase'):
            self._accumulated_phase = 0.0
        self._accumulated_phase += delta

        ticked = False
        if self._accumulated_phase >= _TWO_PI:
            self.geometric_tick_count += 1
            self._accumulated_phase  -= _TWO_PI
            ticked = True

        self._last_ring_net_phase = new_net
        return ticked

    def _ring_spin_signal(self) -> float:
        """
        Net spin signal from the structural ring.
        Only odd (spinning) structural waypoints contribute — even ones
        are static reference points and carry no directional signal.
        Positive = builders net-positive phase (vertical build active).
        Negative = builders net-negative phase (horizontal recognize active).
        """
        structural = self._structural_waypoints()
        spinning   = [wp for wp in structural if wp.spin_sign == +1]
        if not spinning:
            return 0.0
        return float(np.mean([math.sin(wp.spin_phase) for wp in spinning]))

    def _directed_transport(self) -> float:
        """
        Directed tension transport between adjacent Mersenne fold negotiators.

        After each string ticks independently, cold strings (high fold_phase,
        near maximum absorption capacity) pull tension from adjacent hot strings
        (low fold_phase, releasing). The differential drives the flow — small
        delta moves little, large delta moves proportionally more.

        This is the temperature differential transport mechanism: cold
        displacement doing active directional work. The cold region doesn't
        just receive — it actively pulls, creating a directed flow through
        the tensegrity field rather than independent local equilibria.

        Transport fraction derived from asymmetric_delta × |fold_differential|
        — same base constant as the rest of the system, scaled by the actual
        state difference between neighbors.

        Polarity boundary respected: positive-side strings (idx 0-25) only
        exchange with positive-side neighbors, negative-side (idx 26-51) with
        negative-side. Two separate circuits, same geometry.

        Returns total tension transported this cycle — observable signal.
        """
        total_transported = 0.0

        for side_start, side_end in [(0, _STRINGS_PER_SIDE),
                                      (_STRINGS_PER_SIDE, _N_STRINGS)]:
            side_strings = [s for s in self.strings[side_start:side_end] if s.active]
            if len(side_strings) < 2:
                continue

            for i, s in enumerate(side_strings):
                # Two neighbors — wrap within the polarity side
                left  = side_strings[(i - 1) % len(side_strings)]
                right = side_strings[(i + 1) % len(side_strings)]

                for neighbor in (left, right):
                    # Absorption capacity: high fold_phase sin → cold (absorbing)
                    # Low fold_phase sin → hot (releasing)
                    s_capacity        = 0.5 * (1.0 + math.sin(s.fold_phase))
                    neighbor_capacity = 0.5 * (1.0 + math.sin(neighbor.fold_phase))
                    fold_differential = neighbor_capacity - s_capacity

                    # Only transport if neighbor is colder (higher capacity)
                    # and this string has tension to give
                    if fold_differential > 0 and abs(s.tension) > _PRUNE_AMP_MIN:
                        transfer = (
                            _SUBTRACTION_BASE
                            * fold_differential
                            * abs(s.tension)
                            * 0.5   # split between two neighbors
                        )
                        transfer = min(transfer, abs(s.tension) * 0.25)  # cap at 25% per step

                        # Move tension from hot string to cold neighbor
                        direction      = 1.0 if s.tension >= 0 else -1.0
                        s.tension     -= transfer * direction
                        neighbor.tension += transfer * neighbor.polarity
                        total_transported += transfer

        return total_transported

    # ── Field stress ──────────────────────────────────────────────────────────

    def _field_stress(self) -> float:
        """
        Global field stress — mean tension_scalar across all non-structural
        waypoints, normalized to [0, 1] relative to _MAX_TENSION.
        Used for general field health reporting.
        """
        active = [wp for wp in self.waypoints if wp.role in ("positive", "negative", "symbol")]
        if not active:
            return 0.0
        return float(np.mean([wp.tension_scalar for wp in active])) / _MAX_TENSION

    def _local_stress(self, stabilizer_wp: 'Waypoint', window: int = 1) -> float:
        """
        Local neighborhood stress for a specific stabilizer waypoint.

        Measures mean tension deviation from baseline (1.0) across the
        immediate symbol waypoints on each side of the stabilizer on the
        27-symbol ring. Window=1 means 1 neighbor each side = 2 total.

        Tight window is correct — the stabilizer should read only its
        immediate neighbors, not a smoothed average across 6 positions.
        Averaging kills the signal. Two neighbors, maximum sensitivity.
        """
        # Get all symbol waypoints sorted by angle
        symbols = sorted(
            [wp for wp in self.waypoints if wp.role in ("symbol", "stabilizer")],
            key=lambda w: w.angle
        )
        if not symbols:
            return 0.0

        # Find this stabilizer's position in the sorted ring
        try:
            idx = next(i for i, w in enumerate(symbols) if w.wp_id == stabilizer_wp.wp_id)
        except StopIteration:
            return 0.0

        n = len(symbols)
        neighbors = []
        for offset in range(1, window + 1):
            left  = symbols[(idx - offset) % n]
            right = symbols[(idx + offset) % n]
            if left.role == "symbol":
                neighbors.append(left)
            if right.role == "symbol":
                neighbors.append(right)

        if not neighbors:
            return 0.0

        # Deviation from baseline (1.0) — how far pushed from resting state
        deviations = [abs(wp.tension_scalar - 1.0) for wp in neighbors]
        return float(np.mean(deviations))

    # ── Semantic tension injection ────────────────────────────────────────────

    def inject_semantic_tension(
        self,
        subject_id: int,
        verb_id:    int,
        object_id:  int,
    ) -> None:
        tension = 0.0
        if subject_id > 0 and subject_id % 2 == 1:
            tension += 0.8
        if verb_id % 2 == 0:
            tension -= 0.6
        if object_id % 3 == 0:
            tension += 0.4

        for s in self.strings:
            if not s.active:
                continue
            s.tension += tension * (1.0 if s.polarity > 0 else -0.3)

        self.golden_zone["semantic_tension"] = tension

        if abs(tension) < 0.25:
            self._elect_core(zero_is_braking=self.golden_zone.get("zero_braking", False))

    # ── Core selection ────────────────────────────────────────────────────────

    def _elect_core(self, zero_is_braking: bool = False) -> None:
        """
        Phase-gated core election.

        Suspended when zero is braking — the field hasn't committed to a
        direction so no center can be elected. You can't elect a center
        when the field hasn't decided which way it's pointing.

        Each structural waypoint owns exactly one phase window — 1/8 of
        the full 2π spin cycle. Only the waypoint whose window contains
        the current ring_net_phase is eligible for election.

        Window assignment:
          wp 18 → [0,        π/4)
          wp 19 → [π/4,      π/2)
          wp 20 → [π/2,    3π/4)
          wp 21 → [3π/4,      π)
          wp 22 → [π,      5π/4)
          wp 23 → [5π/4,   3π/2)
          wp 24 → [3π/2,   7π/4)
          wp 25 → [7π/4,     2π)

        Within the eligible waypoint, election still requires
        core_score >= _CORE_MIN_CLARITY — merit still gates the win.
        If the eligible waypoint doesn't have enough clarity × persistence,
        no core is elected this cycle. The field stays coreless until
        the ring advances into a window whose waypoint has earned it.

        This makes core election a function of both where the field IS
        (ring phase), what the field has resolved (core_score), and
        whether the field has committed to a direction (zero state).
        """
        if not self.waypoints:
            return

        # Zero braking = dual held = no direction committed = no election
        if zero_is_braking:
            for wp in self.waypoints:
                wp.is_core = False
            self.core_id = None
            return

        ring_phase    = self._ring_net_phase()
        _PHASE_WINDOW = _TWO_PI / _N_STRUCTURAL   # π/4 per window

        # Find which structural waypoint owns the current ring_net_phase
        eligible_id = None
        for wp in self.waypoints:
            if wp.role != "structural":
                continue
            # Window index from construction order
            window_idx   = wp.wp_id - (_N_POSITIVE + _N_NEGATIVE)  # 0-7
            window_start = (window_idx / _N_STRUCTURAL) * _TWO_PI
            window_end   = window_start + _PHASE_WINDOW
            # Circular containment check
            phase_in_window = (
                (window_start <= ring_phase < window_end)
                if window_end <= _TWO_PI
                else (ring_phase >= window_start or ring_phase < window_end % _TWO_PI)
            )
            if phase_in_window:
                eligible_id = wp.wp_id
                break

        for wp in self.waypoints:
            wp.is_core = False

        if eligible_id is not None:
            candidate = self.waypoints[eligible_id]
            if candidate.core_score >= _CORE_MIN_CLARITY:
                candidate.is_core             = True
                self.core_id                  = eligible_id
                self.golden_zone["center_id"] = eligible_id
            else:
                self.core_id = None
        else:
            self.core_id = None

    # ── Tension cycle ─────────────────────────────────────────────────────────

    def apply_tension_cycle(self, wave_amp: float) -> Dict[str, Any]:
        """
        One full tension + attention + spin pass.

        Spin advances every cycle before tension is applied — spin state
        is always one step ahead, which means it reflects where the field
        IS rather than where it was. This is the correct causal order:
        position determines force, not the other way around.
        """
        current_tick = time.time()
        delta_t      = max(current_tick - self.last_tick, 1e-6)
        self.last_tick = current_tick

        # Advance spin ring — check for geometric tick
        geo_ticked   = self._advance_spin_ring()
        spin_signal  = self._ring_spin_signal()

        # Resolve dynamic zero — determine if displacement engine is braking
        # Zero unresolved (spin=0) = dual held = braking active on both sides
        # Zero resolved = brake released on committed side
        zero_wp = next((wp for wp in self.waypoints if wp.is_dynamic_zero), None)
        zero_resolved_int = zero_wp.resolved_int(spin_signal) if zero_wp else 0
        zero_is_braking   = (zero_resolved_int == 0)
        self.golden_zone["zero_braking"] = zero_is_braking  # dual held = braking

        # Update waypoints in spin_phase order — spinning (odd) waypoints
        # rotate through different phase positions each cycle so no fixed
        # construction-order position holds a persistent advantage.
        # Static (even) waypoints hold their construction phase as reference.
        # Sort ascending by spin_phase — the order rotates naturally each cycle
        # as odd waypoints advance while even waypoints stay fixed.
        for wp in sorted(self.waypoints, key=lambda w: w.spin_phase):
            wp.update(wave_amp, delta_t)
            # Dual-13 modulation: resolved integer scales tension contribution
            # Positive integers amplify, negative compress, zero brakes
            r_int = wp.resolved_int(spin_signal)
            if r_int != 0:
                scale = 1.0 + (r_int / 13.0) * 0.08
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * scale, 0.0, _MAX_TENSION
                ))
            elif zero_is_braking:
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * 0.98, 0.0, _MAX_TENSION
                ))

        total_bleed    = 0.0
        active_strings = 0
        for s in self.strings:
            if not s.active:
                continue
            wp_idx    = s.idx % len(self.waypoints)
            wp        = self.waypoints[wp_idx]
            remainder = s.tick(wp.tension_scalar * wave_amp)
            # Zero resolution modulates string polarity contribution:
            # Positive spin → amplify positive strings, compress negative
            # Negative spin → amplify negative strings, compress positive
            # Braking (zero held) → both sides held equally, no modulation
            if not zero_is_braking:
                if zero_resolved_int == +1 and s.polarity == +1:
                    remainder *= 1.0 + _SUBTRACTION_BASE  # slight positive boost
                elif zero_resolved_int == -1 and s.polarity == -1:
                    remainder *= 1.0 + _SUBTRACTION_BASE  # slight negative boost
            wp.tension_scalar = float(np.clip(
                wp.tension_scalar + remainder * 0.01, 0.0, _MAX_TENSION
            ))
            total_bleed    += s.sub_factor
            active_strings += 1
            if s.should_prune():
                s.active = False

        # Directed transport — cold strings pull tension from hot neighbors
        # Runs after all strings have ticked independently so each string's
        # fold_phase reflects its current cycle position before transfer
        transport_total = self._directed_transport()

        for wp in self.waypoints:
            if wp.role == "positive":
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1 + 0.05 * wave_amp), 0.0, _MAX_TENSION
                ))
            elif wp.role == "negative":
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1 - 0.04 * wave_amp), 0.0, _MAX_TENSION
                ))
            elif wp.role == "stabilizer":
                # Local neighborhood stress — reads only the symbol waypoints
                # immediately adjacent on the ring. Each stabilizer gets an
                # independent reading based on what's happening in its local
                # geometry, not the global average.
                # Local threshold: asymmetric_delta × π ≈ 0.0515
                # — same spin step, same scale. When local deviation exceeds
                # this the neighborhood is genuinely stressed relative to baseline.
                local_stress = self._local_stress(wp)
                _LOCAL_THRESHOLD = _SPIN_STEP  # asymmetric_delta × π ≈ 0.0515
                if local_stress > _LOCAL_THRESHOLD:
                    overage      = (local_stress - _LOCAL_THRESHOLD) / max(local_stress, 1e-8)
                    bleed_amount = overage * wp.tension_scalar * 0.35
                    wp.tension_scalar = float(np.clip(
                        wp.tension_scalar - bleed_amount, 0.1, _MAX_TENSION
                    ))
                    # Record exhaust — track amount and rate
                    wp.bleed_rate  = bleed_amount - wp._bleed_prev
                    wp._bleed_prev = bleed_amount
                    wp.bleed_last  = bleed_amount
                    wp.bleed_total += bleed_amount
                else:
                    wp.bleed_last  = 0.0
                    wp.bleed_rate  = -wp._bleed_prev
                    wp._bleed_prev = 0.0
            elif wp.role == "structural":
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar + invariants.asymmetric_delta * 0.01,
                    0.5, 1.5
                ))

        if self._current_prompt:
            words      = self._current_prompt.lower().split()
            subject_id = hash(words[0]) % 13 + 1 if words else 1
            verb_id    = hash(words[1]) % 13 + 1 if len(words) > 1 else 2
            object_id  = hash(words[-1]) % 13 + 1 if len(words) > 2 else 3
            self.inject_semantic_tension(subject_id, verb_id, object_id)
        else:
            self._elect_core(zero_is_braking=zero_is_braking)

        return {
            "active_strings":   active_strings,
            "total_bleed":      round(total_bleed, 6),
            "transport_total":  round(transport_total, 6),
            "core_id":          self.core_id,
            "core_score":       round(
                self.waypoints[self.core_id].core_score, 4
            ) if self.core_id is not None else 0.0,
            "geometric_tick":   geo_ticked,
            "geo_tick_count":   self.geometric_tick_count,
            "ring_net_phase":   round(self._ring_net_phase(), 6),
            "ring_spin_signal": round(spin_signal, 6),
            "zero_resolved":    zero_resolved_int,
            "zero_braking":     zero_is_braking,
        }

    # ── Wave reaction ─────────────────────────────────────────────────────────

    def react_to_wave(self, waveform: np.ndarray) -> None:
        if len(waveform) == 0:
            return
        amp = float(np.mean(np.abs(waveform)))
        self.apply_tension_cycle(amp)

    # ── Band emit + core propagation ──────────────────────────────────────────

    def band_emit_and_core_propagate(self, tri_data: Dict[str, Any]) -> np.ndarray:
        """
        Each active symbol on the 27-symbol ring emits its waveform band.
        The core reads all bands, cancels low-amplitude noise, and propagates
        the strengthened signal back as the field's output waveform.

        Spin signal modulates the propagated band: when the ring is spinning
        net-positive the band is amplified slightly; net-negative it is
        compressed. This ensures spin direction survives into the output
        even when the amplitude is saturated.
        """
        symbol_stream = tri_data.get("symbol_stream", [])
        if not symbol_stream:
            return np.array([0.0])

        band_values = []
        for sym in symbol_stream:
            for wp in self.waypoints:
                if wp.role in ("symbol", "stabilizer") and wp.symbol == sym:
                    band_values.append(wp.tension_scalar * wp.persistence)
                    break
            else:
                band_values.append(0.0)

        band = np.array(band_values, dtype=float)

        if self.core_id is not None:
            core_wp       = self.waypoints[self.core_id]
            core_strength = core_wp.core_score
            noise_mask    = np.abs(band) < 0.15 * max(core_strength, 0.1)
            band[noise_mask] = 0.0
            propagated    = band * (1.0 + 0.25 * core_strength)
        else:
            propagated = band

        # Spin signal modulation — small, signed, survives saturation
        spin_mod   = 1.0 + 0.08 * self._ring_spin_signal()
        propagated = propagated * spin_mod

        return propagated if len(propagated) > 0 else np.array([0.0])

    # ── Structure generation ──────────────────────────────────────────────────

    def generate_structure(
        self,
        prompt:         str,
        tri_data:       Dict[str, Any],
        wave_amplitude: float = 0.0,
    ) -> Dict[str, Any]:
        self._current_prompt = prompt
        cycle_result = self.apply_tension_cycle(wave_amplitude)

        symbol_stream     = tri_data.get("symbol_stream", [])
        raw_prompt        = tri_data.get("prompt", "")
        activated_symbols = []

        # Build per-character weights from the raw prompt
        # Characters map to weights via SymbolicWave._LETTER_WEIGHT
        # We iterate the raw prompt in parallel with the symbol stream
        # to get the weight for each character that produced each symbol.
        # Lazy import — SymbolicWave is in wave/, bipolar_lattice in utils/
        try:
            from wave.symbolic_wave import _LETTER_WEIGHT
        except ImportError:
            _LETTER_WEIGHT = {}

        # Build weight list aligned to symbol stream
        raw_chars = [c for c in raw_prompt if c]  # all chars including spaces
        weights   = []
        for c in raw_chars:
            w = _LETTER_WEIGHT.get(c.lower(), 0.5)
            weights.append(w)
        # Pad or trim to symbol stream length
        while len(weights) < len(symbol_stream):
            weights.append(0.5)
        weights = weights[:len(symbol_stream)]

        for idx, sym in enumerate(symbol_stream):
            letter_weight = weights[idx] if idx < len(weights) else 0.5
            for wp in self.waypoints:
                if wp.role in ("symbol", "stabilizer") and wp.symbol == sym:
                    # Weight IS the injection — no dampening factor.
                    # Vowels hit the ring at 1.9× wave_amplitude.
                    # Rare letters at 0.4×. Real difference, detectable
                    # by local stress measurement.
                    wp.tension_scalar = float(np.clip(
                        wp.tension_scalar + wave_amplitude * letter_weight,
                        0.0, _MAX_TENSION
                    ))
                    activated_symbols.append(sym)
                    break

        pos_tensions = [wp.tension_scalar for wp in self.waypoints if wp.role == "positive"]
        convergence  = float(np.mean(pos_tensions)) if pos_tensions else 0.0

        return {
            "num_waypoints":           len(self.waypoints),
            "active_strings":          cycle_result["active_strings"],
            "total_bleed":             cycle_result["total_bleed"],
            "core_id":                 cycle_result["core_id"],
            "core_score":              cycle_result["core_score"],
            "web_convergence_score":   round(convergence, 4),
            "global_clarity":          round(clarity_ratio.current_ratio, 4),
            "activated_symbols":       len(activated_symbols),
            "golden_zone_tension":     round(self.golden_zone["semantic_tension"], 4),
            "ring_net_phase":          cycle_result["ring_net_phase"],
            "ring_spin_signal":        cycle_result["ring_spin_signal"],
            "geo_tick_count":          cycle_result["geo_tick_count"],
            "field_stress":            round(self._field_stress(), 4),
            "fold_negotiation_signal": round(self._fold_negotiation_signal(), 4),
            "transport_total":         cycle_result["transport_total"],
            "zero_resolved":           cycle_result["zero_resolved"],
            "zero_braking":            cycle_result["zero_braking"],
            "symbol_mode":             "bipolar_27_lattice",
        }

    def _fold_negotiation_signal(self) -> float:
        """
        Mean fold_phase sin across all active Mersenne strings.
        Positive = strings net in negotiation-absorb phase.
        Negative = strings net in negotiation-release phase.
        This is the aggregate negotiation state of the tensegrity field.
        """
        active = [s for s in self.strings if s.active]
        if not active:
            return 0.0
        return float(np.mean([math.sin(s.fold_phase) for s in active]))

    # ── Exhaust signature ─────────────────────────────────────────────────────

    def _stabilizer_waypoints(self) -> List['Waypoint']:
        return [wp for wp in self.waypoints if wp.role == "stabilizer"]

    def get_exhaust_signature(self) -> np.ndarray:
        """
        Current exhaust signature — 6-element vector:
          Elements 0-4: normalized bleed ratios across the 5 geometric
            stabilizers (sum = 1.0) — a coordinate in the 4-simplex.
          Element 5: log1p(total_bleed) — magnitude of field activation.

        The 5D ratio alone was degenerate: sentences with different total
        energy but the same ratio distribution produced identical signatures
        and zero distance to each other. Adding log-scaled magnitude breaks
        this degeneracy. Two prompts must now match BOTH their stress pattern
        AND their activation intensity to be considered geometrically similar.

        log1p is used (not log) so zero bleed maps cleanly to 0.0.
        Derived from the same stabilizer bleed physics — no arbitrary constants.

        Returns raw zeros (6D) if no stabilizer has fired yet this prompt.
        """
        stabs  = self._stabilizer_waypoints()
        totals = np.array([wp.bleed_total for wp in stabs], dtype=float)
        total  = totals.sum()
        if total < 1e-10:
            return np.zeros(6)
        normalized = totals / total
        magnitude  = float(np.log1p(total))   # log-scaled intensity
        return np.append(normalized, magnitude)

    def get_exhaust_rates(self) -> np.ndarray:
        """
        Current per-stabilizer bleed rates — how fast each is exhausting
        right now. Captures the dynamic fingerprint rather than the
        cumulative one. Useful for detecting transition points.
        """
        stabs = self._stabilizer_waypoints()
        return np.array([wp.bleed_rate for wp in stabs], dtype=float)

    def reset_exhaust(self) -> None:
        """
        Reset exhaust tracking at the start of each new prompt.
        Cumulative totals cleared so each prompt gets a clean signature.

        Also pulls symbol and stabilizer waypoint tension_scalars gently
        back toward baseline (1.0). Not a hard reset — multiplying by 0.85
        preserves session state while preventing accumulated drift from
        equalizing all neighborhoods over time.

        Derived: 0.85 = 1 - asymmetric_delta × π × 10 ≈ 1 - 0.515 × 0.3
        Close enough to the system's own decay rate without being arbitrary.
        """
        # Pull ring waypoints back toward tension baseline
        for wp in self.waypoints:
            if wp.role in ("symbol", "stabilizer", "structural"):
                wp.tension_scalar = float(np.clip(
                    1.0 + (wp.tension_scalar - 1.0) * 0.85,
                    0.1, _MAX_TENSION
                ))
                # Structural waypoints also decay clarity_contribution
                # so core score doesn't ratchet upward across the session
                if wp.role == "structural":
                    wp.clarity_contribution *= 0.85

        # Clear exhaust tracking
        for wp in self._stabilizer_waypoints():
            wp.bleed_total = 0.0
            wp.bleed_last  = 0.0
            wp.bleed_rate  = 0.0
            wp._bleed_prev = 0.0

    def etch_exhaust(self, prompt: str, symbol_stream: List[str]) -> None:
        """
        Store the current exhaust signature alongside the prompt and
        symbol stream in exhaust_memory. Called at the end of each
        prompt after the full pipeline has run.

        exhaust_memory grows across sessions — persisted to disk after
        each etch so signatures accumulate over time like the truth library.
        """
        sig = self.get_exhaust_signature()
        if sig[:5].sum() < 1e-10:   # check ratio portion only (element 5 is magnitude)
            return  # nothing fired — no fingerprint to store
        self.exhaust_memory.append({
            "signature":     sig.tolist(),
            "prompt":        prompt,
            "symbol_stream": symbol_stream,
            "core_id":       self.core_id,
            "ring_phase":    self._ring_net_phase(),
        })
        self._save_exhaust_memory()

    def _load_exhaust_memory(self) -> None:
        """Load persisted exhaust signatures from disk on startup."""
        if not os.path.exists(_EXHAUST_MEMORY_FILE):
            return
        try:
            with open(_EXHAUST_MEMORY_FILE, "r") as f:
                loaded = json.load(f)
            # Convert signature lists back to numpy arrays
            for entry in loaded:
                entry["signature"] = np.array(entry["signature"], dtype=float)
            self.exhaust_memory = loaded
            print(f"BipolarLattice: loaded {len(loaded)} exhaust signatures from disk")
        except Exception as e:
            print(f"BipolarLattice: exhaust memory load failed: {e}")

    def _save_exhaust_memory(self) -> None:
        """Persist exhaust memory to disk after each new etch."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable = []
            for entry in self.exhaust_memory:
                serializable.append({
                    "signature":     entry["signature"].tolist()
                                     if hasattr(entry["signature"], "tolist")
                                     else list(entry["signature"]),
                    "prompt":        entry["prompt"],
                    "symbol_stream": entry["symbol_stream"],
                    "core_id":       entry["core_id"],
                    "ring_phase":    entry["ring_phase"],
                })
            with open(_EXHAUST_MEMORY_FILE, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"BipolarLattice: exhaust memory save failed: {e}")

    def nearest_exhaust(self, top_n: int = 3) -> List[Dict]:
        """
        Find the closest prior exhaust signatures to the current one.
        Uses Euclidean distance in the 4-simplex — geometrically nearest
        means most similar field stress pattern, which means most similar
        input geometry.

        Returns top_n matches sorted by distance (closest first).
        This is the content-addressable readout — no weights, no training,
        just geometric proximity in exhaust space.
        """
        if not self.exhaust_memory:
            return []
        current = self.get_exhaust_signature()
        if current[:5].sum() < 1e-10:  # check ratio portion
            return []
        results = []
        for entry in self.exhaust_memory:
            prior = np.array(entry["signature"], dtype=float)
            # Handle mixed 5D/6D entries from prior sessions gracefully:
            # pad old 5D entries with magnitude=0.0 so distance is computable
            if len(prior) == 5 and len(current) == 6:
                prior = np.append(prior, 0.0)
            n    = min(len(current), len(prior))
            dist = float(np.linalg.norm(current[:n] - prior[:n]))
            results.append({
                "distance":  round(dist, 6),
                "prompt":    entry["prompt"],
                "core_id":   entry["core_id"],
                "ring_phase":entry["ring_phase"],
            })
        results.sort(key=lambda x: x["distance"])
        return results[:top_n]

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        active_strings = sum(1 for s in self.strings if s.active)
        core_score     = (
            round(self.waypoints[self.core_id].core_score, 4)
            if self.core_id is not None else 0.0
        )
        return {
            "total_waypoints":         len(self.waypoints),
            "active_strings":          active_strings,
            "core_id":                 self.core_id,
            "core_score":              core_score,
            "global_clarity":          round(clarity_ratio.current_ratio, 4),
            "golden_zone_tension":     round(self.golden_zone["semantic_tension"], 4),
            "ring_net_phase":          round(self._ring_net_phase(), 6),
            "ring_spin_signal":        round(self._ring_spin_signal(), 6),
            "geo_tick_count":          self.geometric_tick_count,
            "field_stress":            round(self._field_stress(), 4),
            "fold_negotiation_signal": round(self._fold_negotiation_signal(), 4),
            "transport_total":         round(sum(abs(s.tension) for s in self.strings if s.active), 6),
            "zero_braking":            (self._ring_spin_signal() == 0.0),
            "mode":                    "bipolar_18_8_27_52_spin_d13",
        }


# Singleton
bipolar_lattice = BipolarLattice()