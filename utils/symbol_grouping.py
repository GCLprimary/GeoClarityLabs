"""
utils/symbol_grouping.py
========================
Symbol Grouping Semantics — Dual-13 Signed Integer Mapping

DUAL-13 SYSTEM
──────────────
The 27-symbol alphabet is a signed integer system, not a linear sequence:

  0          →  central regulator  (governs +1/-1 spin tick, not a content symbol)
  A=+1 … M=+13  →  positive side  (A through M, 13 symbols)
  N=-1 … Z=-13  →  negative side  (N through Z, 13 symbols)

This means:
  - A and N are mirror images at ±1
  - M and Z are the positive and negative ceilings at ±13
  - 0 is the fulcrum — it regulates transition, not content
  - The alphabet folds in half at the 0/M-N boundary

LATTICE MAPPING (REBUILT)
──────────────────────────
Positive symbols (A–M, values +1 to +13) map to the first half of
the Fibonacci lattice [0, π) — expanding outward from 0.

Negative symbols (N–Z, values -1 to -13) map to the second half
[π, 2π) — expanding outward from π in the negative direction.

0 maps exactly to π — the boundary between positive and negative,
which is also why it shares a lattice coordinate with N (-1).
N is the first step into negative territory and 0 is the boundary
they both sit on. This is geometrically correct, not a bug.

ODD/EVEN ASYMMETRY (SIGNED)
────────────────────────────
Odd absolute values (±1,±3,±5,±7,±9,±11,±13) build vertically.
Even absolute values (±2,±4,±6,±8,±10,±12) recognize horizontally.

Positive odd  → vertical upward   (+)
Negative odd  → vertical downward (-)
Positive even → horizontal right  (+, compressed)
Negative even → horizontal left   (-, compressed)

GROUP TENSION
─────────────
Each group's base_tension is now computed from the signed integer
values of its members, not their raw alphabet positions. This means
a group containing A(+1) and N(-1) has near-zero net tension —
they cancel. A group containing M(+13) and L(+12) has strong
positive tension. The geometry encodes magnitude and direction.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set

from core.invariants import invariants
from utils.fold_line_resonance import fold_line_resonance

# ── Constants ─────────────────────────────────────────────────────────────────
_GOLDEN_ANGLE     = math.pi * (3 - math.sqrt(5))
_TWO_PI           = 2 * math.pi
_LATTICE_POINTS   = 512
_SYMBOLS          = ['0'] + [chr(ord('A') + i) for i in range(26)]  # 27 symbols

# Neighbourhood radius — just above mean symbol spacing in the new mapping
_GROUP_RADIUS     = 20

# Minimum imprint to be considered an active resonance point
_IMPRINT_THRESHOLD = 0.005

# Odd/even tension asymmetry — now applied to signed integer values
# Positive odd  → vertical upward   (amplification)
# Negative odd  → vertical downward (inversion)
# Positive even → horizontal right  (compressed)
# Negative even → horizontal left   (compressed)
_ODD_SCALE        = 0.8
_EVEN_SCALE       = 0.6
_MOD3_INTERFERENCE = 0.4


# ── Dual-13 signed integer helpers ───────────────────────────────────────────

def symbol_to_signed(sym: str) -> int:
    """
    Map symbol to its signed integer value in the dual-13 system.

      '0'    →   0  (central regulator)
      A=+1 … M=+13  (positive side)
      N=-1 … Z=-13  (negative side)
    """
    if sym == '0':
        return 0
    idx = ord(sym.upper()) - ord('A')   # A=0 … Z=25
    if idx <= 12:                        # A–M → +1 to +13
        return idx + 1
    else:                                # N–Z → -1 to -13
        return -(idx - 12)              # N=idx13→-1, Z=idx25→-13


def signed_to_lattice_angle(signed_val: int) -> float:
    """
    Map a signed dual-13 integer to a Fibonacci lattice target angle.

    Positive values (+1 to +13) expand into [0, π):
        angle = (val / 13) * π
        +1  → small positive angle near 0
        +13 → angle approaching π

    Negative values (-1 to -13) expand into (π, 2π):
        angle = π + (|val| / 13) * π
        -1  → small step past π
        -13 → angle approaching 2π

    0 → exactly π (the boundary / fulcrum)

    This means the positive side and negative side are mirror images
    folded at π — the same structure reflected, opposite signed.
    """
    if signed_val == 0:
        return math.pi
    elif signed_val > 0:
        return (signed_val / 14.0) * math.pi
    else:
        return math.pi + (abs(signed_val) / 14.0) * math.pi


class SymbolGroup:
    """
    A cluster of symbols sharing a fold-line geometric context.

    Members share proximity in the Fibonacci lattice AND have both
    been touched by resonance events. Tension is now computed from
    the signed dual-13 integer values of members — magnitude and
    direction both matter.
    """

    def __init__(self, group_id: int, seed_symbol: str, seed_lattice_idx: int):
        self.group_id             = group_id
        self.members:   List[str] = [seed_symbol]
        self.lattice_indices: List[int] = [seed_lattice_idx]
        self.tension_centroid     = 0.0
        self.signed_values:  List[int] = [symbol_to_signed(seed_symbol)]

    def add(self, symbol: str, lattice_idx: int) -> None:
        if symbol not in self.members:
            self.members.append(symbol)
            self.lattice_indices.append(lattice_idx)
            self.signed_values.append(symbol_to_signed(symbol))

    def update_centroid(self, imprints: np.ndarray) -> None:
        if not self.lattice_indices:
            return
        vals = [float(imprints[i]) for i in self.lattice_indices
                if i < len(imprints)]
        self.tension_centroid = float(np.mean(vals)) if vals else 0.0

    @property
    def net_signed_value(self) -> int:
        """
        Sum of all member signed values.
        Positive = group leans toward positive side.
        Negative = group leans toward negative side.
        Near zero = balanced cross-polarity group (e.g. M+F seen earlier
        is +13 + +6 = +19, but K+D is +11 + +4 = +15 — both positive).
        A true balanced group would be something like A+N = +1 + -1 = 0.
        """
        return sum(self.signed_values)

    @property
    def dominant_polarity(self) -> str:
        """Direction derived from net signed value and odd/even majority."""
        net = self.net_signed_value
        # Count odd and even absolute values
        odd_count  = sum(1 for v in self.signed_values if v != 0 and abs(v) % 2 == 1)
        even_count = sum(1 for v in self.signed_values if v != 0 and abs(v) % 2 == 0)
        zero_count = sum(1 for v in self.signed_values if v == 0)

        if zero_count == len(self.signed_values):
            return 'regulator'
        if net == 0:
            return 'balanced'
        if odd_count > even_count:
            return 'vertical_up' if net > 0 else 'vertical_down'
        else:
            return 'horizontal_right' if net > 0 else 'horizontal_left'

    @property
    def odd_count(self) -> int:
        return sum(1 for v in self.signed_values if v != 0 and abs(v) % 2 == 1)

    @property
    def even_count(self) -> int:
        return sum(1 for v in self.signed_values if v != 0 and abs(v) % 2 == 0)

    @property
    def base_tension(self) -> float:
        """
        Base tension vector — signed integer magnitude × imprint centroid.

        Each member contributes its signed value scaled by odd/even factor:
          odd  member → value × _ODD_SCALE   (vertical driver)
          even member → value × _EVEN_SCALE  (horizontal recognizer)
          zero        → no contribution (regulator role)

        Net tension is the sum across all members, weighted by centroid.
        This means:
          - A group of all positive odds has strong positive tension
          - A group spanning + and - sides has tension near zero
          - Magnitude encodes how far from centre the group sits
        """
        raw = 0.0
        for v in self.signed_values:
            if v == 0:
                continue
            scale = _ODD_SCALE if abs(v) % 2 == 1 else _EVEN_SCALE
            raw  += (v / 13.0) * scale   # normalise by max value (13)
        return float(np.clip(raw * self.tension_centroid, -2.0, 2.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id":          self.group_id,
            "members":           self.members,
            "signed_values":     self.signed_values,
            "net_signed_value":  self.net_signed_value,
            "size":              len(self.members),
            "tension_centroid":  round(self.tension_centroid, 4),
            "base_tension":      round(self.base_tension, 4),
            "dominant_polarity": self.dominant_polarity,
            "odd_count":         self.odd_count,
            "even_count":        self.even_count,
        }


class SymbolGrouping:
    """
    Dynamic symbol grouping driven by fold-line imprint geometry.

    Groups are recomputed whenever the fold-line imprint pattern has
    changed significantly — tracked via a change counter. Between
    recomputations, cached groups are used so the pipeline doesn't
    pay the grouping cost on every tick.

    Primary interface:
      group_for(symbol)        → which group does this symbol belong to?
      pair_tension(s1, s2)     → combined tension for a symbol pair
      stream_context(stream)   → tension profile across a symbol stream
    """

    def __init__(self):
        self.groups:       List[SymbolGroup]       = []
        self.symbol_map:   Dict[str, int]          = {}  # symbol → group_id
        self.lattice_map:  Dict[str, int]          = {}  # symbol → lattice_idx
        self._last_imprint_sum = 0.0
        self._recompute_threshold = 0.02   # recompute when imprint sum shifts by this much

        # Precompute symbol → lattice index mapping
        # Uses same golden_angle as Fibonacci lattice so mapping is native
        self._symbol_lattice_indices = self._map_symbols_to_lattice()

    # ── Symbol → lattice mapping ──────────────────────────────────────────────

    def _map_symbols_to_lattice(self) -> Dict[str, int]:
        """
        Map each symbol to its nearest Fibonacci lattice index using
        the dual-13 signed integer system.

        Positive symbols (A–M, +1 to +13) → target angles in [0, π)
        Negative symbols (N–Z, -1 to -13) → target angles in (π, 2π)
        Zero                               → exactly π (the boundary)

        This places positive and negative sides as mirror images folded
        at π — the same geometric structure, opposite signed direction.
        The 0/N collision at π is now structurally correct: N=-1 is the
        first step into negative territory and 0 is the boundary itself.
        """
        fib_angles = np.array([
            (_GOLDEN_ANGLE * i) % _TWO_PI
            for i in range(_LATTICE_POINTS)
        ])

        mapping = {}
        for sym in _SYMBOLS:
            signed_val   = symbol_to_signed(sym)
            target_angle = signed_to_lattice_angle(signed_val)

            diffs   = np.abs(fib_angles - target_angle)
            diffs   = np.minimum(diffs, _TWO_PI - diffs)
            nearest = int(np.argmin(diffs))
            mapping[sym] = nearest

        return mapping

    # ── Group computation ─────────────────────────────────────────────────────

    def _compute_groups(self) -> None:
        """
        Cluster symbols by shared fold-line imprint proximity.

        Algorithm:
          1. Get current imprint array from fold_line_resonance
          2. For each symbol, check if its lattice index is imprinted
             above threshold
          3. Build groups by proximity: two symbols join the same group
             if their lattice indices are within _GROUP_RADIUS of each
             other AND both are imprinted
          4. Unimprinted symbols each get their own singleton group
             (they haven't been touched by resonance yet — no context)
          5. Update tension centroids from imprint values
        """
        imprints = fold_line_resonance.lattice_imprints

        # Find imprinted symbols
        imprinted: List[Tuple[str, int, float]] = []
        unimprinted: List[str] = []

        for sym in _SYMBOLS:
            lidx   = self._symbol_lattice_indices[sym]
            imp    = float(imprints[lidx]) if lidx < len(imprints) else 0.0
            if imp >= _IMPRINT_THRESHOLD:
                imprinted.append((sym, lidx, imp))
            else:
                unimprinted.append(sym)

        # Sort by lattice index for proximity sweep
        imprinted.sort(key=lambda x: x[1])

        new_groups: List[SymbolGroup] = []
        assigned:   Set[str]          = set()
        group_id = 0

        for sym, lidx, imp in imprinted:
            if sym in assigned:
                continue

            # Start a new group with this symbol as seed
            grp = SymbolGroup(group_id, sym, lidx)
            assigned.add(sym)

            # Sweep for neighbours within _GROUP_RADIUS
            for other_sym, other_lidx, other_imp in imprinted:
                if other_sym in assigned:
                    continue
                if abs(other_lidx - lidx) <= _GROUP_RADIUS:
                    grp.add(other_sym, other_lidx)
                    assigned.add(other_sym)

            grp.update_centroid(imprints)
            new_groups.append(grp)
            group_id += 1

        # Singleton groups for unimprinted symbols
        for sym in unimprinted:
            lidx = self._symbol_lattice_indices[sym]
            grp  = SymbolGroup(group_id, sym, lidx)
            grp.update_centroid(imprints)
            new_groups.append(grp)
            group_id += 1

        self.groups = new_groups

        # Rebuild symbol → group_id map
        self.symbol_map = {}
        for grp in self.groups:
            for sym in grp.members:
                self.symbol_map[sym] = grp.group_id

        self._last_imprint_sum = float(np.sum(imprints))

    def _should_recompute(self) -> bool:
        current_sum = float(np.sum(fold_line_resonance.lattice_imprints))
        return abs(current_sum - self._last_imprint_sum) > _RECOMPUTE_THRESHOLD if \
               hasattr(self, '_recompute_threshold') else True

    def _ensure_groups(self) -> None:
        """Recompute groups if imprint pattern has shifted enough."""
        current_sum = float(np.sum(fold_line_resonance.lattice_imprints))
        if (not self.groups or
                abs(current_sum - self._last_imprint_sum) > self._recompute_threshold):
            self._compute_groups()

    # ── Public interface ──────────────────────────────────────────────────────

    def group_for(self, symbol: str) -> Optional[SymbolGroup]:
        """Return the group this symbol belongs to."""
        self._ensure_groups()
        gid = self.symbol_map.get(symbol)
        if gid is None:
            return None
        for grp in self.groups:
            if grp.group_id == gid:
                return grp
        return None

    def pair_tension(self, s1: str, s2: str) -> Dict[str, Any]:
        """
        Directed tension for a symbol pair — source→target model.

        s1 is the SOURCE: sets direction (sign of v1)
        s2 is the TARGET: sets displacement magnitude (|v2 - v1|)

        This makes the system non-commutative:
          A(+1)→M(+13): direction=+, displacement=12/13 → strong positive
          M(+13)→A(+1): direction=+, displacement=12/13 BUT source is at
                         ceiling moving toward centre → actually contracts
                         so we invert when source > target on same side.

        Full directional rule:
          direction  = sign(v1) if v1 != 0 else sign(v2)
          raw_disp   = (v2 - v1) / 13.0          (signed displacement)
          base       = direction * |raw_disp| * scale

          When source and target are on the same side:
            moving away from centre (|v2| > |v1|) → expansive, same sign
            moving toward centre   (|v2| < |v1|) → contractive, opposite sign

          Cross-polarity (different sides):
            full displacement across boundary — raw_disp carries sign

        Scale uses the source symbol's odd/even property — the source
        drives the character of the interaction.

        Same group → constructive boost (interference boost if odd+even)
        Different group → competitive (same-side compressed, cross full)
        """
        self._ensure_groups()

        g1 = self.group_for(s1)
        g2 = self.group_for(s2)

        if g1 is None or g2 is None:
            return {"tension": 0.0, "relationship": "unresolved",
                    "same_group": False}

        v1         = symbol_to_signed(s1)
        v2         = symbol_to_signed(s2)
        same_group = (g1.group_id == g2.group_id)

        # Source scale — odd drives vertically, even recognizes horizontally
        scale = _ODD_SCALE if (v1 != 0 and abs(v1) % 2 == 1) else _EVEN_SCALE

        # Direction from source
        if v1 != 0:
            direction = math.copysign(1.0, v1)
        elif v2 != 0:
            direction = math.copysign(1.0, v2)
        else:
            direction = 0.0

        # Signed displacement: how far and in what direction target is from source
        raw_disp = (v2 - v1) / 13.0

        # Base tension: source direction × displacement magnitude × scale
        # For same-side pairs, sign of raw_disp tells us expansion vs contraction
        if (v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0):
            # Same side — direction preserved, magnitude is displacement
            base = direction * abs(raw_disp) * scale
            # Moving away from centre = expansive (amplify slightly)
            # Moving toward centre = contractive (compress slightly)
            if abs(v2) > abs(v1):
                base *= 1.1    # expansive
            else:
                base *= 0.9    # contractive
            relationship = "constructive" if same_group else "compressed"
        else:
            # Cross-polarity or one is zero — full signed displacement
            base = raw_disp * scale
            relationship = "constructive" if same_group else "cross_polarity"

        # Same-group interference boost for odd+even pairing
        if same_group and v1 != 0 and v2 != 0 and abs(v1) % 2 != abs(v2) % 2:
            base *= (1.0 + invariants.asymmetric_delta)

        # Mod-3 interference
        if (abs(v1) + abs(v2)) % 3 == 0:
            base += math.copysign(_MOD3_INTERFERENCE * 0.5, base)

        # Centroid weight — floor-based so imprinted symbols contribute fully
        centroid_weight = (g1.tension_centroid + g2.tension_centroid) / 2.0
        weight = max(0.1, 1.0 - (1.0 - centroid_weight) ** 2)  # floor: unimprinted pairs retain raw geometric identity
        base  *= weight

        return {
            "tension":       round(float(np.clip(base, -2.0, 2.0)), 4),
            "relationship":  relationship,
            "same_group":    same_group,
            "group_ids":     (g1.group_id, g2.group_id),
            "polarities":    (g1.dominant_polarity, g2.dominant_polarity),
            "signed_values": (v1, v2),
            "direction":     direction,
            "displacement":  round(raw_disp, 4),
        }

    def stream_context(self, symbol_stream: List[str]) -> Dict[str, Any]:
        """
        Compute the tension profile across an entire symbol stream.

        For each adjacent pair in the stream, compute pair_tension.
        The resulting tension curve is the stream's geometric
        interpretation — how meaning accumulates and shifts across
        the symbol sequence.

        Zero boundary behaviour:
          - 0 marks a phase boundary and resets the running tension window
          - A non-zero symbol adjacent to 0 contributes its solo signed
            value (scaled by its group centroid weight) rather than
            being silenced. A symbol next to a boundary isn't silent —
            it's unpartnered, which is different from zero tension.
          - A 0→0 adjacency produces 0.0 (pure boundary, no content)
        """
        self._ensure_groups()

        if not symbol_stream:
            return {"tensions": [], "mean_tension": 0.0, "zero_boundaries": []}

        tensions:        List[float] = []
        zero_boundaries: List[int]   = []
        window = []   # running accumulator — reset at each boundary

        for i in range(len(symbol_stream) - 1):
            s1 = symbol_stream[i]
            s2 = symbol_stream[i + 1]

            # Both are boundaries — pure separator, no content
            if s1 == '0' and s2 == '0':
                tensions.append(0.0)
                zero_boundaries.append(i)
                window = []   # reset accumulator
                continue

            # One side is a boundary — solo reading for the content symbol
            if s1 == '0' or s2 == '0':
                zero_boundaries.append(i)
                window = []   # reset accumulator at boundary

                # Get the content symbol
                content_sym = s2 if s1 == '0' else s1
                v = symbol_to_signed(content_sym)
                if v == 0:
                    tensions.append(0.0)
                    continue

                # Solo tension — signed value scaled by odd/even.
                # Use floor weight: unimprinted symbols contribute their
                # raw geometric identity at minimum weight (0.1) rather
                # than zero. An unimprinted symbol next to a boundary
                # still has a signed value — it's just unconfirmed.
                scale    = _ODD_SCALE if abs(v) % 2 == 1 else _EVEN_SCALE
                grp      = self.group_for(content_sym)
                centroid = grp.tension_centroid if grp else 0.0
                # Floor weight: minimum 0.1 so raw identity always shows
                weight   = max(0.1, 1.0 - (1.0 - centroid) ** 2)
                solo_t   = (v / 13.0) * scale * weight
                tensions.append(round(float(np.clip(solo_t, -2.0, 2.0)), 4))
                continue

            # Normal pair — both non-zero
            pt = self.pair_tension(s1, s2)
            tensions.append(pt["tension"])

        mean_t = float(np.mean(tensions)) if tensions else 0.0

        # Profile: running mean for smoothed tension curve
        profile = []
        win     = []
        for t in tensions:
            win.append(t)
            if len(win) > 4:
                win.pop(0)
            profile.append(round(float(np.mean(win)), 4))

        return {
            "tensions":        [round(t, 4) for t in tensions],
            "tension_profile": profile,
            "mean_tension":    round(mean_t, 4),
            "zero_boundaries": zero_boundaries,
            "coherence_used":  round(fold_line_resonance.get_coherence_signal(), 4),
            "stream_length":   len(symbol_stream),
        }

    def get_group_summary(self) -> List[Dict[str, Any]]:
        """Return all current groups as dicts for inspection/logging."""
        self._ensure_groups()
        return [g.to_dict() for g in self.groups]

    def get_collisions(self) -> List[Dict[str, Any]]:
        """
        Return symbols that map to the same lattice index.
        These are genuine geometric collisions — the Fibonacci lattice
        treats them as identical coordinates. Most significant: 0 and N
        both map to index 305, meaning the even/odd separator and N
        share a lattice position. This is structurally meaningful.
        """
        from collections import defaultdict
        reverse: dict = defaultdict(list)
        for sym, idx in self._symbol_lattice_indices.items():
            reverse[idx].append(sym)
        return [
            {"lattice_idx": idx, "symbols": syms}
            for idx, syms in reverse.items()
            if len(syms) > 1
        ]

    def get_status(self) -> Dict[str, Any]:
        self._ensure_groups()
        # Count any group whose centroid is above threshold — singletons included.
        # A symbol touched by resonance is geometrically active even without neighbours.
        imprinted_groups = sum(
            1 for g in self.groups
            if g.tension_centroid >= _IMPRINT_THRESHOLD
        )
        return {
            "total_groups":     len(self.groups),
            "imprinted_groups": imprinted_groups,
            "symbols_mapped":   len(self.symbol_map),
            "coherence":        fold_line_resonance.get_coherence_signal(),
            "last_imprint_sum": round(self._last_imprint_sum, 4),
            "collisions":       self.get_collisions(),
        }


# Singleton
symbol_grouping = SymbolGrouping()
