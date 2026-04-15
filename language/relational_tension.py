"""
language/relational_tension.py
================================
Relational Tension — Cross-Sentence Field Carry

After each sentence is processed, the named invariants and stable words
from that sentence inject a small residual tension into the 52 Mersenne
bands. The next sentence processes against a field already shaped by
what came before.

THE MECHANISM
─────────────
Each named invariant has a signed geometric value derived from its
word fingerprint — net_signed captures how far into positive or negative
territory that word sits in the dual-13 space.

After a sentence resolves, we compute a CARRY VECTOR from its named
invariant hits:

  carry = sum(net_signed(word) * familiarity(word) * carry_scale)
          for each named invariant in the sentence

This carry vector is distributed across the 52 Mersenne strings using
the same odd/even polarity asymmetry as inject_semantic_tension:
  positive carry → positive strings amplified (+1 polarity)
  negative carry → negative strings amplified (-1 polarity)

The carry bleeds away at the Mersenne subtraction rate between sentences,
so it doesn't accumulate indefinitely. What persists is what was strong
enough to survive the bleed — exactly the radioactive decay principle.

CORRECT CONTINUATION
────────────────────
When the next sentence contains words that are geometrically aligned
with the residual carry, the field resolves faster — the incoming
tension reinforces the existing field state. This produces:
  - Higher persistence
  - Lower field stress
  - More settled direction (positive or negative stays consistent)

When the next sentence contradicts the carry (opposite direction, high
net_signed words from the other side), the field has to work harder.
Field stress increases, persistence may drop, and the reconstruction
output will show higher peak tensions.

This is conversation emerging as field resolution — not lookup.

CARRY SCALE
───────────
The carry injection is intentionally small — asymmetric_delta * 2 as
the base scale. This keeps it as a nudge, not a takeover. The field's
response to the actual new sentence still dominates. The carry is
the whisper of prior context, not a constraint.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

from core.invariants import invariants
from utils.bipolar_lattice import bipolar_lattice
from utils.symbol_grouping import symbol_grouping, symbol_to_signed

# ── Constants ─────────────────────────────────────────────────────────────────
# Carry scale — how strongly prior sentences influence the current field
# Small by design: the new sentence dominates, prior context nudges
_CARRY_SCALE       = invariants.asymmetric_delta * 2.0   # ≈ 0.0328

# Maximum carry magnitude — caps so even very strong sentences don't overpower
_CARRY_MAX         = 0.25

# Decay between sentences — how much carry bleeds before next sentence
# Uses same Mersenne subtraction base as the strings themselves
_INTER_CARRY_DECAY = 1.0 - (invariants.asymmetric_delta / 3)   # ≈ 0.9945

# How many prior sentences to carry forward (rolling window)
_CARRY_WINDOW      = 5


class SentenceCarry:
    """
    Single sentence's contribution to the relational tension field.
    Stores the carry vector and its current strength after decay.
    """

    def __init__(
        self,
        sentence:      str,
        carry_value:   float,
        named_anchors: List[str],
        direction:     str,
    ):
        self.sentence      = sentence
        self.carry_value   = carry_value   # signed — positive or negative field lean
        self.named_anchors = named_anchors
        self.direction     = direction
        self.age           = 0             # sentences since this carry was injected

    def decay(self) -> None:
        """Bleed carry between sentences."""
        self.carry_value *= _INTER_CARRY_DECAY
        self.age         += 1

    @property
    def is_active(self) -> bool:
        """Carry is active if its magnitude is still meaningful."""
        return abs(self.carry_value) > 1e-4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence":      self.sentence[:40] + "..." if len(self.sentence) > 40
                             else self.sentence,
            "carry_value":   round(self.carry_value, 4),
            "named_anchors": self.named_anchors,
            "direction":     self.direction,
            "age":           self.age,
        }


class RelationalTension:
    """
    Cross-sentence field carry manager.

    After each sentence:
      1. Extract carry from named invariants and stable words
      2. Inject into 52 Mersenne bands
      3. Decay all prior carries

    Before each sentence:
      - The accumulated carry from the rolling window is already
        sitting in the Mersenne bands, shaping how the field responds
    """

    def __init__(self):
        self._window: deque = deque(maxlen=_CARRY_WINDOW)
        self._total_injected = 0
        self._net_carry      = 0.0

    # ── Carry extraction ──────────────────────────────────────────────────────

    def _extract_carry(
        self,
        fingerprint:  Dict[str, Any],
        vocab_hits:   List[Dict[str, Any]],
        invariant_engine: Any,
    ) -> float:
        """
        Compute the carry value for a sentence.

        Carry = weighted sum of named invariant net_signed values,
                modulated by familiarity and field direction.

        Named invariants contribute more than merely stable words
        because they've been etched into the truth library —
        they've earned structural significance.
        """
        carry = 0.0

        per_word = {w["word"]: w for w in fingerprint.get("per_word", [])}

        for hit in vocab_hits:
            word       = hit["word"]
            familiarity = hit["familiarity"]
            is_named   = hit.get("named", False)

            # Get net_signed for this word from per_word data
            word_data  = per_word.get(word, {})
            net_signed = word_data.get("net_signed", 0.0)

            # Named invariants carry more weight
            weight = familiarity * (2.0 if is_named else 0.5)
            carry += net_signed * weight * _CARRY_SCALE

        # Modulate by sentence direction
        direction = fingerprint.get("direction", "boundary")
        if direction == "positive":
            carry *= 1.1
        elif direction == "negative":
            carry *= 1.1   # same scale, sign is already in net_signed
        # boundary: no additional modulation

        # Cap magnitude
        carry = float(np.clip(carry, -_CARRY_MAX, _CARRY_MAX))
        return carry

    # ── Mersenne band injection ───────────────────────────────────────────────

    def _inject_into_bands(self, carry: float) -> None:
        """
        Distribute carry across the 52 Mersenne strings.

        Follows the same odd/even polarity pattern as inject_semantic_tension:
          positive carry → positive strings (+1 polarity) amplified
          negative carry → negative strings (-1 polarity) amplified

        The carry is small by design so it nudges rather than overrides.
        The asymmetric_delta ensures it's derived from the system's own
        fundamental constant — not an arbitrary injection value.
        """
        if abs(carry) < 1e-6:
            return

        for s in bipolar_lattice.strings:
            if not s.active:
                continue
            # Positive carry → reinforce positive strings, dampen negative
            # Negative carry → reinforce negative strings, dampen positive
            if carry > 0:
                injection = carry * (1.0 if s.polarity > 0 else -0.2)
            else:
                injection = carry * (1.0 if s.polarity < 0 else -0.2)

            s.tension = float(np.clip(
                s.tension + injection,
                -2.0, 2.0
            ))

        self._total_injected += 1
        self._net_carry = float(np.clip(
            self._net_carry + carry, -_CARRY_MAX * 2, _CARRY_MAX * 2
        ))

    # ── Public interface ──────────────────────────────────────────────────────

    def after_sentence(
        self,
        fingerprint:      Dict[str, Any],
        vocab_hits:       List[Dict[str, Any]],
        invariant_engine: Any,
    ) -> float:
        """
        Call this after each sentence is processed.

        1. Decay all existing carries
        2. Extract carry from this sentence
        3. Inject into Mersenne bands
        4. Store in rolling window

        Returns the carry value injected.
        """
        # Decay existing carries first
        for carry_entry in self._window:
            carry_entry.decay()

        # Remove exhausted carries
        active = [c for c in self._window if c.is_active]
        self._window.clear()
        self._window.extend(active)

        # Extract carry for this sentence
        carry = self._extract_carry(fingerprint, vocab_hits, invariant_engine)

        if abs(carry) > 1e-6:
            # Inject into Mersenne bands
            self._inject_into_bands(carry)

            # Store in window
            named_anchors = [h["word"] for h in vocab_hits if h.get("named")]
            entry = SentenceCarry(
                sentence      = fingerprint.get("sentence", ""),
                carry_value   = carry,
                named_anchors = named_anchors,
                direction     = fingerprint.get("direction", "boundary"),
            )
            self._window.append(entry)

        return carry

    def get_current_carry(self) -> float:
        """
        Total active carry currently sitting in the field.
        Capped to ±_CARRY_MAX so window accumulation cannot saturate
        the field and force alignment=+1.0 on every subsequent prompt.
        """
        if not self._window:
            return 0.0
        total = float(sum(c.carry_value for c in self._window))
        return float(np.clip(total, -_CARRY_MAX, _CARRY_MAX))

    def get_carry_direction(self) -> str:
        """Current direction of accumulated carry."""
        carry = self.get_current_carry()
        if carry > 0.01:
            return "positive"
        elif carry < -0.01:
            return "negative"
        return "neutral"

    def get_window(self) -> List[Dict[str, Any]]:
        """Return the current carry window for inspection."""
        return [c.to_dict() for c in self._window]

    def measure_alignment(
        self,
        fingerprint: Dict[str, Any],
    ) -> float:
        """
        Measure how well a sentence's fingerprint aligns with current carry.

        Returns [-1, 1]:
          +1.0 = perfect alignment (sentence direction matches carry direction)
          -1.0 = full opposition (sentence contradicts prior context)
           0.0 = neutral / no carry

        This is the "correct continuation" signal — used to modulate
        answer confidence and report field coherence across turns.
        """
        carry = self.get_current_carry()
        if abs(carry) < 1e-4:
            return 0.0

        net_tension = fingerprint.get("net_tension", 0.0)
        if abs(net_tension) < 1e-4:
            return 0.0

        # Alignment = normalised dot product of carry and sentence direction
        carry_sign   = math.copysign(1.0, carry)
        tension_sign = math.copysign(1.0, net_tension)
        raw          = carry_sign * tension_sign  # +1 same direction, -1 opposite

        # Scale by magnitude of carry relative to max
        magnitude_factor = min(1.0, abs(carry) / _CARRY_MAX)
        return round(float(raw * magnitude_factor), 4)

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_carries":   len(self._window),
            "net_carry":        round(self.get_current_carry(), 4),
            "carry_direction":  self.get_carry_direction(),
            "total_injected":   self._total_injected,
            "window":           self.get_window(),
        }


# Singleton
relational_tension = RelationalTension()
