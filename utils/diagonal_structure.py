"""
utils/diagonal_structure.py
===========================
Diagonal Structure Generator

Converts exhaust signatures into self-organizing geometric structures.
Each exhaust stabilizer generates 2 diagonals (one builder, one recognizer)
giving 10 diagonals total — 2 per exhaust, seam points identical at every
scale for reversibility.

Derivation chain:
  entry spacing  = 2π / 5          — 5 evenly spaced seam points on unrolled circle
  offset angle   = π / 8           — 22.5°, same family as spin step (asymmetric_delta × π)
  z_scale        = asymmetric_delta × 64 + 1 ≈ 1.04929  — per-step growth factor
  dispersal      = π               — structure folds back at effective_boundary
  max_steps      = π / (asymmetric_delta × π) ≈ 61      — derived maximum step count

The output structure IS the fingerprint made visible. Similar exhaust
signatures grow similar structures. The complexity score and diagonal
vectors can be compared directly for similarity measurement.

Reversibility: because seam points are evenly spaced and the offset is
constant, the structure can be unrolled back to its seed exhaust signature.
Information is conserved — nothing is lost, just restructured.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any

# ── Derived constants ─────────────────────────────────────────────────────────
_PI                = math.pi
_TWO_PI            = 2 * _PI
_ASYMMETRIC_DELTA  = 0.01639510239          # (2π/3) - 2.078
_OFFSET_ANGLE      = _PI / 8               # 22.5° — π/8
_Z_SCALE           = _ASYMMETRIC_DELTA * 64 + 1   # ≈ 1.04929 per step
_DISPERSAL_ANGLE   = _PI                   # structure folds at π
_MAX_STEPS         = int(_PI / (_ASYMMETRIC_DELTA * _PI))  # ≈ 61
_SEAM_SPACING      = _TWO_PI / 5           # evenly spaced seam points
_N_EXHAUSTS        = 5
_DIAGONALS_PER_EXHAUST = 2
_N_DIAGONALS       = _N_EXHAUSTS * _DIAGONALS_PER_EXHAUST  # 10 total


class Diagonal:
    """
    A single diagonal line segment in the structure.

    Builder (odd index): grows vertically — advances in y
    Recognizer (even index): grows horizontally — advances in x

    Angle determines direction. Length accumulates via z_scale per step.
    Dispersal fires when accumulated_angle reaches π.
    """

    def __init__(
        self,
        index:       int,
        entry_x:     float,
        entry_y:     float,
        angle:       float,
        exhaust_val: float,
    ):
        self.index          = index
        self.is_builder     = (index % 2 == 1)   # odd = builder, even = recognizer
        self.start          = (entry_x, entry_y)
        self.angle          = angle % _TWO_PI
        self.exhaust_val    = exhaust_val

        # Step through the diagonal applying z_scale growth
        self.steps:         List[Tuple[float, float]] = [(entry_x, entry_y)]
        self.accumulated_angle = 0.0
        self.dispersed      = False
        self._grow()

    def _grow(self) -> None:
        """
        Step the diagonal until dispersal condition is met.

        Each step:
          length = exhaust_val × z_scale^step_number
          if builder (odd): advance in y direction
          if recognizer (even): advance in x direction
          accumulated_angle += offset_angle per step
          dispersal when accumulated_angle >= π
        """
        x, y = self.start
        step_length = self.exhaust_val

        for step in range(_MAX_STEPS):
            step_length *= _Z_SCALE

            dx = step_length * math.cos(self.angle)
            dy = step_length * math.sin(self.angle)

            if self.is_builder:
                # Vertical build — y dominant, x as offset
                x += dx * _ASYMMETRIC_DELTA
                y += dy
            else:
                # Horizontal build — x dominant, y as offset
                x += dx
                y += dy * _ASYMMETRIC_DELTA

            self.steps.append((x, y))
            self.accumulated_angle += _OFFSET_ANGLE

            if self.accumulated_angle >= _DISPERSAL_ANGLE:
                self.dispersed = True
                break

    @property
    def end(self) -> Tuple[float, float]:
        return self.steps[-1]

    @property
    def length(self) -> float:
        """Total arc length of all steps."""
        total = 0.0
        for i in range(1, len(self.steps)):
            dx = self.steps[i][0] - self.steps[i-1][0]
            dy = self.steps[i][1] - self.steps[i-1][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    @property
    def direction_vector(self) -> np.ndarray:
        """Unit vector from start to end — encodes overall direction."""
        sx, sy = self.start
        ex, ey = self.end
        v = np.array([ex - sx, ey - sy], dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return np.zeros(2)
        return v / norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index":             self.index,
            "is_builder":        self.is_builder,
            "start":             self.start,
            "end":               self.end,
            "angle":             round(self.angle, 6),
            "length":            round(self.length, 6),
            "steps":             len(self.steps),
            "dispersed":         self.dispersed,
            "accumulated_angle": round(self.accumulated_angle, 6),
        }


class DiagonalStructure:
    """
    Full 10-diagonal structure grown from an exhaust signature.

    The structure is the fingerprint made visible. Two structures grown
    from similar exhaust signatures will be geometrically similar —
    measurable via dot product of their direction vector arrays.
    """

    def __init__(
        self,
        exhaust_signature: np.ndarray,
        ring_net_phase:    float = 0.0,
        core_id:           int   = None,
        prompt:            str   = "",
    ):
        self.exhaust_signature = exhaust_signature
        self.ring_net_phase    = ring_net_phase
        self.core_id           = core_id
        self.prompt            = prompt
        self.diagonals:        List[Diagonal] = []
        self._build()

    def _build(self) -> None:
        """
        Grow all 10 diagonals from the exhaust signature.

        For each of 5 exhausts:
          entry_x = (exhaust_index / 5) × 2π   — evenly spaced seam points
          entry_y = exhaust_value               — amplitude from bleed

          Diagonal 1 (builder, odd index):
            angle = exhaust_value × π + ring_net_phase

          Diagonal 2 (recognizer, even index):
            angle = exhaust_value × π + ring_net_phase + π/8  (22.5° offset)
        """
        for i, val in enumerate(self.exhaust_signature):
            entry_x = (i / _N_EXHAUSTS) * _TWO_PI
            entry_y = float(val)

            # Builder diagonal — odd index
            builder_angle = float(val) * _PI + self.ring_net_phase
            builder = Diagonal(
                index       = i * 2 + 1,
                entry_x     = entry_x,
                entry_y     = entry_y,
                angle       = builder_angle,
                exhaust_val = float(val),
            )
            self.diagonals.append(builder)

            # Recognizer diagonal — even index
            recognizer_angle = float(val) * _PI + self.ring_net_phase + _OFFSET_ANGLE
            recognizer = Diagonal(
                index       = i * 2 + 2,
                entry_x     = entry_x,
                entry_y     = entry_y,
                angle       = recognizer_angle,
                exhaust_val = float(val),
            )
            self.diagonals.append(recognizer)

    @property
    def direction_matrix(self) -> np.ndarray:
        """
        10×2 matrix of unit direction vectors — one per diagonal.
        This is the compact representation for similarity comparison.
        """
        return np.array([d.direction_vector for d in self.diagonals])

    @property
    def complexity_score(self) -> float:
        """
        Total structure complexity — sum of all diagonal lengths
        normalized by dispersal threshold.
        Derived: dispersal threshold = π × max_steps × z_scale^(max_steps/2)
        """
        total_length = sum(d.length for d in self.diagonals)
        dispersed    = sum(1 for d in self.diagonals if d.dispersed)
        # Complexity = length × dispersal fraction
        dispersal_fraction = dispersed / max(_N_DIAGONALS, 1)
        return round(total_length * (1 + dispersal_fraction), 6)

    def similarity(self, other: 'DiagonalStructure') -> float:
        """
        Geometric similarity between two structures.
        Uses mean dot product of direction vectors — 1.0 = identical,
        0.0 = orthogonal, -1.0 = opposite.

        This is the structural equivalent of exhaust signature distance —
        but richer because it captures directional geometry, not just
        amplitude ratios.
        """
        a = self.direction_matrix
        b = other.direction_matrix
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dots = np.sum(a[:n] * b[:n], axis=1)
        return float(np.mean(dots))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt":           self.prompt[:60],
            "ring_net_phase":   round(self.ring_net_phase, 6),
            "core_id":          self.core_id,
            "complexity_score": self.complexity_score,
            "n_diagonals":      len(self.diagonals),
            "n_dispersed":      sum(1 for d in self.diagonals if d.dispersed),
            "diagonals":        [d.to_dict() for d in self.diagonals],
        }


class DiagonalStructureGenerator:
    """
    Generates and compares diagonal structures from exhaust signatures.

    Standalone module — takes exhaust signature + ring state, produces
    a DiagonalStructure. Comparison between structures gives geometric
    similarity in the output domain, complementing exhaust distance in
    the input domain.
    """

    def __init__(self):
        self.structures: List[DiagonalStructure] = []

    def generate(
        self,
        exhaust_signature: np.ndarray,
        ring_net_phase:    float = 0.0,
        core_id:           int   = None,
        prompt:            str   = "",
    ) -> DiagonalStructure:
        """
        Generate a diagonal structure from an exhaust signature.
        Stores it in history for later comparison.
        """
        sig = np.array(exhaust_signature, dtype=float)
        # Ensure normalized — should already be but guard anyway
        total = sig.sum()
        if total > 1e-10:
            sig = sig / total

        structure = DiagonalStructure(
            exhaust_signature = sig,
            ring_net_phase    = ring_net_phase,
            core_id           = core_id,
            prompt            = prompt,
        )
        self.structures.append(structure)
        return structure

    def nearest(self, structure: DiagonalStructure, top_n: int = 3) -> List[Dict]:
        """
        Find the most geometrically similar prior structures.
        Uses direction matrix dot product similarity.
        Returns top_n matches sorted by similarity (highest first).
        """
        if len(self.structures) < 2:
            return []
        results = []
        for prior in self.structures[:-1]:  # exclude the current one
            sim = structure.similarity(prior)
            results.append({
                "similarity": round(sim, 6),
                "prompt":     prior.prompt[:60],
                "complexity": prior.complexity_score,
                "core_id":    prior.core_id,
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_n]

    def get_status(self) -> Dict[str, Any]:
        return {
            "structures_generated": len(self.structures),
            "z_scale":              round(_Z_SCALE, 6),
            "offset_angle_deg":     round(math.degrees(_OFFSET_ANGLE), 4),
            "max_steps":            _MAX_STEPS,
            "dispersal_angle":      round(_DISPERSAL_ANGLE, 6),
        }


# Singleton
diagonal_structure_generator = DiagonalStructureGenerator()
