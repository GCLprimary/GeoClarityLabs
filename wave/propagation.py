import math
import numpy as np
from typing import Dict, Any
from core.ouroboros_engine import ouroboros_engine
from utils.bipolar_lattice import bipolar_lattice


class WavePropagator:
    def __init__(self, pi_factor: float = 3.141592653589793):
        self.pi_factor        = pi_factor
        self.asymmetric_delta = 0.01639510239

    def propagate(
        self,
        triangulation_data: Dict[str, Any],
        steps:      int   = 100,
        modulation: float = 1.0,
    ) -> Dict[str, Any]:
        if "error" in triangulation_data:
            return triangulation_data
        width           = triangulation_data["width"]
        height          = triangulation_data["height"]
        total_triangles = triangulation_data["total_triangles"]
        n               = triangulation_data.get("n_original", 1)
        base_amplitude  = height * 0.085
        base_frequency  = width / 12.0
        triangle_density = total_triangles / max(n, 1)
        phase_increment = (self.pi_factor + self.asymmetric_delta) * modulation
        waveform = []
        for i in range(steps):
            angle = i * base_frequency * phase_increment
            value = base_amplitude * math.sin(angle) * (1 + 0.1 * triangle_density)
            value += self.asymmetric_delta * (i / steps)
            waveform.append(round(value, 6))
        clarity_ratio_score = (width * height) / max(n, 1)
        persistence = sum(abs(x) for x in waveform) / len(waveform)
        return {
            "status":             "propagated",
            "steps":              steps,
            "width_used":         width,
            "height_used":        height,
            "total_triangles":    total_triangles,
            "base_amplitude":     round(base_amplitude, 6),
            "base_frequency":     round(base_frequency, 6),
            "phase_increment":    round(phase_increment, 8),
            "waveform_sample":    waveform[:8] + ["..."] if len(waveform) > 8 else waveform,
            "waveform_full":      waveform,
            "clarity_ratio_score": round(clarity_ratio_score, 4),
            "persistence":        round(persistence, 6),
            "modulation_applied": modulation,
            "mode":               "direct",
        }

    def propagate_generative(
        self,
        prop_result:        Dict[str, Any],
        triangulation_data: Dict[str, Any],
        recall_triggered:   bool = False,
        pass_depth:         int  = 2,
    ) -> Dict[str, Any]:
        waveform_full = prop_result.get("waveform_full", [])
        if not waveform_full:
            prop_result["mode"] = "direct_fallback"
            return prop_result
        waveform_np = np.array(
            [x for x in waveform_full if isinstance(x, (int, float))], dtype=float
        )
        if len(waveform_np) == 0:
            prop_result["mode"] = "direct_fallback"
            return prop_result
        gen      = ouroboros_engine.run_generative(waveform_np, triangulation_data, pass_depth)
        out_wave = gen["output_waveform"]
        result   = dict(prop_result)
        result.update({
            "waveform_sample": list(out_wave[:8]) + ["..."] if len(out_wave) > 8 else list(out_wave),
            "waveform_full":   out_wave.tolist(),
            "persistence":     round(gen["consensus_pers"], 6),
            "phys_pers":       round(gen["phys_pers"], 4),
            "wave_pers":       round(gen["wave_pers"], 4),
            "data_pers":       round(gen["data_pers"], 4),
            "consensus_pers":  round(gen["consensus_pers"], 4),
            "pass_weights":    [round(w, 4) for w in gen["weights"]],
            "mode":            "generative",
        })
        if hasattr(bipolar_lattice, "golden_zone") and bipolar_lattice.golden_zone["semantic_tension"] < 0.25:
            result["consensus_pers"] = min(1.0, result["consensus_pers"] + 0.12)
        return result
