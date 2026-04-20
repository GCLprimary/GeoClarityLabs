import math
from typing import Dict, Any, Optional, Tuple

class Invariants:
    """
    Central Invariants Module - Clarity Ratio Laboratory

    Contains shared geometric and mathematical invariants used across the lab.

    Dual-13 Integer System:
      +1 to +13 on the positive side (odd vertical builders)
      -1 to -13 on the negative side (even horizontal recognizers)
      Zero sits at center — dual value (+1 AND -1 simultaneously).
      Zero is the only position that has not committed to a direction.
      When zero resolves it becomes either +1 or -1 — that resolution
      is what controls displacement engine braking. Zero deciding its
      direction is what starts or stops movement.

      Letter assignment (A–Z across ±1 to ±13):
        A=+1  B=+2  C=+3  D=+4  E=+5  F=+6  G=+7
        H=+8  I=+9  J=+10 K=+11 L=+12 M=+13
        N=-1  O=-2  P=-3  Q=-4  R=-5  S=-6  T=-7
        U=-8  V=-9  W=-10 X=-11 Y=-12 Z=-13
        '0' symbol → dual value: resolves dynamically per cycle

      Dynamic zero facilitation:
        Zero holds two integers (+1, -1) and maps to two letters (A, N)
        simultaneously until the field resolves its direction.
        The resolved value is determined by ring_spin_signal sign:
          positive spin → zero resolves to +1 (A side)
          negative spin → zero resolves to -1 (N side)
          zero spin     → zero remains unresolved (dual, both held)
    """

    def __init__(self):
        self.pi               = 3.141592653589793
        self.asymmetric_delta = 0.01639510239
        self.golden_ratio     = (1 + math.sqrt(5)) / 2

        # ── Polarization constants (from coupling experiments) ─────────────
        # Confirmed across 3 sessions, 14 controlled prompts.
        # P0_COLD: geometric floor of cold field = sqrt(phi)/phi²
        # P_MAX:   dielectric saturation ceiling = 3/phi²
        # Note: parity threshold 1/phi² = P_MAX/3 exactly.
        _phi = self.golden_ratio
        self.P0_cold          = math.sqrt(_phi) / (_phi ** 2)  # 0.4859
        self.P_max            = 3.0 / (_phi ** 2)              # 1.1459

        # ── Dual-13 integer lookup tables ─────────────────────────────────────
        # Letter → signed integer
        self.letter_to_int: Dict[str, int] = {}
        positive_letters = list('ABCDEFGHIJKLM')   # +1 to +13
        negative_letters = list('NOPQRSTUVWXYZ')   # -1 to -13
        for i, ch in enumerate(positive_letters):
            self.letter_to_int[ch] = i + 1
        for i, ch in enumerate(negative_letters):
            self.letter_to_int[ch] = -(i + 1)

        # Signed integer → letter (reverse lookup)
        self.int_to_letter: Dict[int, str] = {
            v: k for k, v in self.letter_to_int.items()
        }

        # Zero symbol — dual value, unresolved until field commits
        # Represented as a tuple: (positive_potential, negative_potential)
        self.zero_dual = (+1, -1)

    # ── Dual-13 interface ─────────────────────────────────────────────────────

    def symbol_to_int(self, symbol: str, spin_signal: float = 0.0) -> int:
        """
        Map a symbol to its signed dual-13 integer.

        For '0': resolves dynamically from spin_signal.
          positive spin → +1  (vertical build direction)
          negative spin → -1  (horizontal recognize direction)
          zero spin     →  0  (unresolved — dual held, no commitment)

        For A-Z: returns fixed signed integer from lookup.
        For unknown: returns 0 (unresolved).
        """
        if symbol == '0':
            if spin_signal > 0.0:
                return +1
            elif spin_signal < 0.0:
                return -1
            else:
                return 0  # truly unresolved — dual held
        return self.letter_to_int.get(symbol.upper(), 0)

    def int_to_sym(self, value: int) -> str:
        """
        Map a signed integer back to its letter symbol.
        Zero returns '0' (dynamic placeholder).
        Out of range (|value| > 13) clamps to ±13.
        """
        if value == 0:
            return '0'
        clamped = max(-13, min(13, value))
        return self.int_to_letter.get(clamped, '0')

    def dual_zero_state(self, spin_signal: float) -> Tuple[int, int, bool]:
        """
        Return the current state of dynamic zero given the field's spin signal.

        Returns (positive_val, negative_val, is_resolved):
          is_resolved = True  → field has committed to a direction
          is_resolved = False → field is at zero spin, dual held simultaneously

        Positive spin → committed positive (+1 active, brake released on + side)
        Negative spin → committed negative (-1 active, brake released on - side)
        Zero spin     → dual held, neither committed, braking both sides
        """
        if spin_signal > 0.0:
            return (+1, -1, True)    # committed positive
        elif spin_signal < 0.0:
            return (+1, -1, True)    # committed negative
        else:
            return (+1, -1, False)   # dual held, not committed

    def odd_even_bias(self, value: float, layer: int) -> float:
        """Apply odd-vertical / even-horizontal bias."""
        if layer % 2 == 0:
            return value * 0.92
        else:
            return value * 1.08

    def get_pi_gradient(self, scale: float = 1.0) -> float:
        """Return asymmetric π-gradient for directed persistence/zoom."""
        return (self.pi + self.asymmetric_delta) * scale

    def get_status(self) -> Dict[str, Any]:
        return {
            "pi":               self.pi,
            "asymmetric_delta": self.asymmetric_delta,
            "golden_ratio":     round(self.golden_ratio, 6),
            "dual_13_range":    "A=+1..M=+13, N=-1..Z=-13, 0=dynamic",
        }


# Singleton instance for easy import
invariants = Invariants()