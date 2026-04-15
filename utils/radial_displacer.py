import math
import numpy as np
import time
import hashlib
from typing import Dict, Any
from core.invariants import invariants
from core.clarity_ratio import clarity_ratio
from core.safeguards import safeguards

SYMBOLS = ['0'] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Tension prune thresholds — derived from asymmetric π-gradient
_MAX_DISPLACERS     = 10     # hard cap: web never exceeds this
_MIN_WAVE_AMP       = 0.005  # displacers below this amplitude are low-tension
_MIN_AGE_TO_PRUNE   = 2.0    # don't prune brand-new displacers (seconds)


class DynamicRadialDisplacer:
    """
    Reliable spawning + organic growth. Respects real symbol_stream + true
    0-phase-shift barrier. Now includes tension-based pruning so the web
    stays clean instead of accumulating noise across runs.
    """
    def __init__(self):
        self.asymmetric_delta = 0.01639510239
        self.web = []
        self.next_id = 0
        self.last_tick = time.time()

    def _symbol_to_radial_position(self, symbol: str, layer: int = 0, tick: float = 0.0) -> tuple:
        """Deterministic position with HARD phase-shift for '0'."""
        if symbol == '0':
            angle  = (layer % 2) * math.pi + invariants.get_pi_gradient(scale=1.0) * tick
            radius = 2.2
            return (radius * math.cos(angle), radius * math.sin(angle))
        idx    = (ord(symbol.upper()) - ord('A')) % 26
        radius = 1.0 if layer % 2 == 0 else 0.6
        angle  = (idx / 26.0) * 2 * math.pi + (layer * 0.3)
        return (radius * math.cos(angle), radius * math.sin(angle))

    def _create_displacer(self, prompt: str, tri_data: Dict[str, Any], wave_amp: float = 0.0) -> Dict[str, Any]:
        """Use REAL symbol_stream from SymbolicWave (with proper 0s)."""
        symbol_seq = tri_data.get("symbol_stream", [])
        if not symbol_seq:
            hash_val   = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
            symbol_seq = [SYMBOLS[((hash_val >> (i*5)) & 0x1F) % 27] for i in range(8)]

        positions = []
        for i, sym in enumerate(symbol_seq):
            pos = self._symbol_to_radial_position(sym, layer=i, tick=time.time())
            positions.append(pos)

        birth_radius = 0.3 + 0.08 * len(symbol_seq) * (1 + 0.3 * symbol_seq.count('0'))

        displacer = {
            "id":               self.next_id,
            "age":              0.0,
            "growth_rate":      invariants.get_pi_gradient(scale=0.4),
            "positions":        positions,
            "base_radius":      round(birth_radius, 4),
            "symbol_sequence":  symbol_seq,
            "zero_breaks":      [i for i, s in enumerate(symbol_seq) if s == '0'],
            "band_length":      max(4, tri_data.get("width", 4) // 2),
            "active":           True,
            "local_wave_amp":   wave_amp,
            "convergence_score": 0.0,
            "prompt_length":    tri_data.get("n_original", len(prompt)),
            "box_dimensions":   (tri_data.get("width", 4), tri_data.get("height", 4)),
            "birth_time":       time.time(),
        }
        self.next_id += 1
        return displacer

    def _tension_prune(self) -> None:
        """
        Ouroboros-inspired tension pruning.

        Two passes:
          1. Mark low-tension displacers inactive:
             - wave amplitude below _MIN_WAVE_AMP (no energy → no attention)
             - AND old enough that they had a chance to grow
          2. Hard-cap: if web still exceeds _MAX_DISPLACERS after marking,
             evict the lowest-amplitude ones until we're within the cap.

        This mirrors the Ouroboros principle: what doesn't sustain long
        enough to matter gets recycled. The manifold stays clean rather
        than accumulating noise across runs.
        """
        now = time.time()

        # Pass 1 — mark low-tension as inactive
        for d in self.web:
            if not d["active"]:
                continue
            age = now - d.get("birth_time", now)
            if age > _MIN_AGE_TO_PRUNE and d["local_wave_amp"] < _MIN_WAVE_AMP:
                d["active"] = False

        # Remove inactive
        self.web = [d for d in self.web if d["active"]]

        # Pass 2 — hard cap: evict lowest amplitude if over limit
        if len(self.web) > _MAX_DISPLACERS:
            self.web.sort(key=lambda d: d["local_wave_amp"], reverse=True)
            self.web = self.web[:_MAX_DISPLACERS]

    def generate_structure(self, prompt: str, tri_data: Dict[str, Any], wave_amplitude: float = 0.0) -> Dict[str, Any]:
        current_tick = time.time()
        delta_t      = current_tick - self.last_tick
        self.last_tick = current_tick

        # Age and grow existing displacers
        for d in self.web:
            if d["active"]:
                d["age"]           += delta_t * d["growth_rate"]
                growth_factor       = 1 / (1 + math.exp(-0.25 * (d["age"] - 4)))
                symbol_boost        = 1 + 0.12 * (len(d["symbol_sequence"]) / 8.0)
                d["base_radius"]    = round(
                    d["base_radius"] + self.asymmetric_delta * growth_factor * symbol_boost, 4
                )
                d["local_wave_amp"] = wave_amplitude
                d["band_length"]    = max(4, int(
                    d["box_dimensions"][0] * (1 + 0.1 * clarity_ratio.current_ratio)
                ))

        # Tension prune before spawning — keeps the web clean
        self._tension_prune()

        # Spawn new displacer only if web has room and clarity warrants it
        current_clarity = clarity_ratio.current_ratio
        if len(self.web) == 0 or (len(self.web) < _MAX_DISPLACERS and current_clarity > 0.9):
            new_d = self._create_displacer(prompt, tri_data, wave_amplitude)
            self.web.append(new_d)

        convergence = self._compute_web_convergence()

        return {
            "num_displacers":       len(self.web),
            "total_waypoints":      sum(len(d["positions"]) for d in self.web),
            "web_convergence_score": round(convergence, 4),
            "global_clarity":       round(current_clarity, 4),
            "displacers":           self.web,
            "prompt_length":        tri_data.get("n_original", len(prompt)),
            "box_dimensions":       (tri_data.get("width", 4), tri_data.get("height", 4)),
            "total_triangles":      tri_data.get("total_triangles", 8),
            "symbol_mode":          "27-symbol_lattice_active"
        }

    def _compute_web_convergence(self) -> float:
        if len(self.web) < 2:
            return 0.0
        scores = []
        for i, d1 in enumerate(self.web):
            for j, d2 in enumerate(self.web):
                if i >= j:
                    continue
                c1   = np.mean(d1["positions"], axis=0)
                c2   = np.mean(d2["positions"], axis=0)
                dist = np.linalg.norm(c1 - c2)
                overlap = max(0.0, (
                    d1["base_radius"] + d2["base_radius"] - dist
                ) / (d1["base_radius"] + d2["base_radius"]))
                score = (
                    overlap
                    * (d1["local_wave_amp"] + d2["local_wave_amp"]) / 2.0
                    * invariants.get_pi_gradient(scale=1.0)
                )
                scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def react_to_wave(self, waveform: np.ndarray) -> None:
        if len(self.web) == 0 or len(waveform) == 0:
            return
        amp = float(np.mean(np.abs(waveform)))
        for d in self.web:
            if d["active"]:
                d["local_wave_amp"] = amp * (1 + 0.1 * (d["age"] / 10.0))
                if d["band_length"] > 8:
                    d["convergence_score"] = float(
                        np.clip(d["convergence_score"] + 0.05 * amp, 0.0, 1.0)
                    )

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_displacers": len(self.web),
            "next_id":          self.next_id,
            "global_clarity":   round(clarity_ratio.current_ratio, 4),
            "last_tick_age":    round(time.time() - self.last_tick, 2),
            "mode":             "27-symbol_lattice"
        }


# Singleton
radial_displacer = DynamicRadialDisplacer()