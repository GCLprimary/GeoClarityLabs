import math
from typing import Dict, Any


class Invariants:
    """
    Central Invariants Module - Clarity Ratio Laboratory

    All derived constants + dual-13 lookup tables.

    Dual-13 Integer System:
      +1 to +13 on the positive side (odd vertical builders)
      -1 to -13 on the negative side (even horizontal recognizers)
      Zero = dual value (+1 AND -1 simultaneously) until field commits.

      Letter assignment (A-Z across ±1 to ±13):
        A=+1  B=+2  C=+3  D=+4  E=+5  F=+6  G=+7
        H=+8  I=+9  J=+10 K=+11 L=+12 M=+13
        N=-1  O=-2  P=-3  Q=-4  R=-5  S=-6  T=-7
        U=-8  V=-9  W=-10 X=-11 Y=-12 Z=-13
        0 = dynamic dual value, resolves from spin_signal at runtime

    int_to_sym removed — never called.
    dual_zero_state removed — zero state accessed directly via symbol_to_int().
    """

    def __init__(self):
        self.pi               = 3.141592653589793
        self.asymmetric_delta = 0.01639510239
        self.golden_ratio     = (1 + math.sqrt(5)) / 2

        # Letter -> signed integer
        self.letter_to_int: Dict[str, int] = {}
        for i, ch in enumerate(list("ABCDEFGHIJKLM")):
            self.letter_to_int[ch] = i + 1
        for i, ch in enumerate(list("NOPQRSTUVWXYZ")):
            self.letter_to_int[ch] = -(i + 1)

        # Signed integer -> letter (reverse lookup)
        self.int_to_letter: Dict[int, str] = {
            v: k for k, v in self.letter_to_int.items()
        }

        # Zero dual potential — unresolved until field commits
        self.zero_dual = (+1, -1)

    def symbol_to_int(self, symbol: str, spin_signal: float = 0.0) -> int:
        """Map symbol to signed dual-13 integer, resolving 0 via spin_signal."""
        if symbol == "0":
            if spin_signal > 0.0:
                return +1
            elif spin_signal < 0.0:
                return -1
            return 0
        return self.letter_to_int.get(symbol.upper(), 0)

    def odd_even_bias(self, value: float, layer: int) -> float:
        """Apply odd-vertical / even-horizontal bias."""
        return value * (1.08 if layer % 2 == 1 else 0.92)

    def get_pi_gradient(self, scale: float = 1.0) -> float:
        """Return asymmetric pi-gradient for directed persistence/zoom."""
        return (self.pi + self.asymmetric_delta) * scale

    def get_status(self) -> Dict[str, Any]:
        return {
            "pi":               self.pi,
            "asymmetric_delta": self.asymmetric_delta,
            "golden_ratio":     round(self.golden_ratio, 6),
            "dual_13_range":    "A=+1..M=+13, N=-1..Z=-13, 0=dynamic",
        }


invariants = Invariants()
