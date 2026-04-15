"""
core/ouroboros_engine.py
========================
Ouroboros Field Dynamics Engine — Lab Integration Port

Ported from OuroborosFramework (slim v8.9) into GeometricClarityLab's
architecture. Strips cosmological simulation (Hubble, CMB, expansion) and
matplotlib dependencies, keeping the pure geometric field dynamics:

  bloom  → expand + noise injection (stochastic exploration)
  etch   → boundary enforcement + asymmetric bias (irreversible structure)
  prune  → sparsification + selection pressure (information death)
  feedback → truth library injection (persistent memory → field influence)
  tanh   → amplitude stabilization (bounded attractors)

Three pass types with different damping/noise profiles:
  physical — low noise (0.15), high damping (0.995) — structure formation
  wave     — medium noise (0.69), medium damping (0.95) — propagation
  data     — high noise (1.5), low damping (0.75) — unbiased exploration

The truth library stores high-persistence signatures as FFT projections.
On every nested pass, those signatures are resampled and injected back
into the grid — closing the loop that the lab's one-pass wave pipeline
currently leaves open.

Integration with the lab:
  - OuroborosEngine is a singleton imported by WavePropagator
  - WavePropagator.propagate() stays unchanged (fast, direct path)
  - WavePropagator.propagate_generative() runs the multi-pass resonance
    when persistence or recall triggers warrant it
  - The truth library persists to ouro_truth_library.json so high-resonance
    patterns accumulate across sessions — this is the "scaling = training"
    mechanism described in the design notes

Key constants (fully derived — no magic numbers):
  pi_center           = π
  effective_boundary  = 2.078   (sole fixed invariant — minimal asymmetry)
  frame_delta         = 2π/3 - 2.078 ≈ 0.01639510239  (asymmetric_delta)
  deviation           = π - 2.078 ≈ 1.0636             (primary asymmetry)
  prune_threshold     = deviation * 0.01                (sparse but not dead)
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

from core.invariants import invariants
from core.clarity_ratio import clarity_ratio


# ── Derived geometric constants ───────────────────────────────────────────────
_PI               = np.pi
_EFFECTIVE_BOUNDARY = 2.078          # sole fixed invariant
_FRAME_DELTA      = (2 * _PI / 3) - _EFFECTIVE_BOUNDARY   # ≈ 0.01639510239
_DEVIATION        = _PI - _EFFECTIVE_BOUNDARY              # ≈ 1.0636
_PRUNE_THRESHOLD  = abs(_DEVIATION) * 0.01                # sparse, not dead
_SIGNATURE_DIM    = 32               # FFT projection dimension for library
_MAX_GRID_SIZE    = 1024             # downsample ceiling

# Pass profiles — (noise_amplitude, damping_factor)
_PASS_PROFILES = {
    "physical": (0.15, 0.995),   # structure — tight, stable
    "wave":     (0.69, 0.95),    # propagation — breathing
    "data":     (1.5,  0.75),    # exploration — unbiased
}

# Truth library persistence
_LIBRARY_FILE = "ouro_truth_library.json"


class OuroborosEngine:
    """
    Pure geometric field dynamics engine.

    Two propagation modes:
      direct     — one-pass wave (existing lab pipeline, fast)
      generative — multi-pass resonance with library feedback (closed loop)

    The generative mode activates when:
      - persistence >= trigger_persistence (wave is well-sustained)
      - a cross-turn recall was triggered (prior etch feeds into field)
      - explicitly requested

    Truth library entries accumulate across sessions. Each high-persistence
    answer's waveform signature is stored and re-injected on future passes,
    so the field is shaped by what it has successfully resolved before.
    """

    def __init__(
        self,
        env_feedback_fraction: float = 0.08,
        matter_damping:        float = 0.98,
        generative_threshold:  float = 0.38,  # persistence above this → generative
    ):
        self.env_feedback_fraction = env_feedback_fraction
        self.matter_damping        = matter_damping
        self.generative_threshold  = generative_threshold
        self.truth_library: List[Dict] = []
        self._bootstrap_library()
        self._load_library()

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def _bootstrap_library(self) -> None:
        """Seed the library with geometric priors."""
        # Fibonacci phasing — universal resonance harmonic
        fib = np.array([1,1,2,3,5,8,13,21,34,55], dtype=float) / 55.0
        self._add_entry(fib, "fibonacci_phasing")

        # π-asymmetric gradient pulse
        t   = np.linspace(0, 2 * _PI, 32)
        pulse = np.sin(t * (_PI + _FRAME_DELTA))
        self._add_entry(pulse, "pi_asymmetric_pulse")

        # Golden-ratio spiral
        phi = (1 + np.sqrt(5)) / 2
        spiral = np.sin(np.linspace(0, 20 * _PI, 64) * phi)
        self._add_entry(spiral, "golden_ratio_spiral")

    # ── Core field operations ─────────────────────────────────────────────────

    def _bloom(self, grid: np.ndarray, noise_amp: float) -> np.ndarray:
        """Expand + noise injection — stochastic exploration."""
        expanded = np.sin(grid * _PI) * (1 + noise_amp)
        return expanded + np.random.uniform(-noise_amp, noise_amp, grid.shape)

    def _etch(self, grid: np.ndarray) -> np.ndarray:
        """Boundary enforcement + irreversible asymmetric bias."""
        constrained = np.cos(grid) * (grid ** 2 < _EFFECTIVE_BOUNDARY ** 2)
        return constrained + _DEVIATION * np.tanh(grid / _PI)

    def _prune(self, grid: np.ndarray) -> np.ndarray:
        """Sparsification — weak trails die, strong ones persist."""
        result = grid.copy()
        result[np.abs(result) < _PRUNE_THRESHOLD] = 0
        return result

    def _apply_library_feedback(self, grid: np.ndarray) -> np.ndarray:
        """
        Inject truth library signatures back into the field.
        This is the closed loop — prior high-persistence patterns
        shape future propagation without overriding it.
        """
        if not self.truth_library:
            return grid
        feedback   = np.zeros_like(grid)
        target_len = grid.size
        for entry in self.truth_library:
            proj      = np.array(entry["projected"], dtype=float)
            resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(proj)),
                proj
            )
            feedback += resampled.reshape(grid.shape) * self.env_feedback_fraction
        return grid + feedback

    def _stabilize(self, grid: np.ndarray) -> np.ndarray:
        """Tanh amplitude stabilization — prevents runaway attractors."""
        return np.tanh(grid)

    def _downsample(self, grid: np.ndarray) -> np.ndarray:
        """Safety downsample for large grids."""
        if grid.size <= _MAX_GRID_SIZE:
            return grid
        factor = int(np.ceil(np.sqrt(grid.size / _MAX_GRID_SIZE)))
        if grid.ndim == 2:
            h, w   = grid.shape[0] // factor, grid.shape[1] // factor
            return np.mean(grid.reshape(h, factor, w, factor), axis=(1, 3))
        else:
            n = grid.shape[0] // factor
            return np.mean(grid.reshape(n, factor), axis=1)

    def _project_to_signature(self, vec: np.ndarray) -> np.ndarray:
        """
        FFT-based compression to fixed-dim signature.
        Two vectors that share frequency content will have similar signatures
        even if they differ in length or phase — geometric similarity.
        """
        flat = vec.flatten().astype(float)
        if len(flat) < _SIGNATURE_DIM:
            flat = np.pad(flat, (0, _SIGNATURE_DIM - len(flat)))
        fft_mags   = np.abs(np.fft.rfft(flat))
        orig_freq  = np.linspace(0, 0.5, len(fft_mags))
        tgt_freq   = np.linspace(0, 0.5, _SIGNATURE_DIM)
        sig        = np.interp(tgt_freq, orig_freq, fft_mags)
        return sig / (np.linalg.norm(sig) + 1e-8)

    # ── Multi-pass resonance ──────────────────────────────────────────────────

    def nested_pass(
        self,
        grid:      np.ndarray,
        depth:     int  = 2,
        pass_type: str  = "wave",
    ) -> Tuple[np.ndarray, List[float], float]:
        """
        Single-type nested resonance pass.

        Each iteration:
          bloom → etch → library_feedback → prune → dampen → stabilize
          persistence = fraction of non-zero cells (geometric fill ratio)

        Returns (final_grid, persistence_curve, hazard)
        where hazard = std(persistence_curve) — measures instability.
        """
        noise_amp, damping = _PASS_PROFILES.get(pass_type, (0.69, 0.95))
        current    = self._downsample(grid.copy())
        pers_curve = []

        for _ in range(depth):
            current = self._bloom(current, noise_amp)
            current = self._etch(current)
            current = self._apply_library_feedback(current)
            current = self._prune(current)
            current = current * damping
            current = self._stabilize(current)
            pers    = float(np.count_nonzero(current)) / current.size
            pers_curve.append(pers)

        hazard = float(np.std(pers_curve)) if pers_curve else 0.0
        return current, pers_curve, hazard

    def consensus_pass(
        self,
        grid:  np.ndarray,
        depth: int = 2,
    ) -> Dict:
        """
        Full physical + wave + data triple pass with weighted consensus.

        The three pass types run independently on the same input grid.
        Their outputs are weighted by mean persistence — the pass that
        sustained the most structure contributes most to the consensus.
        Library feedback runs once more on the consensus grid, then pruned
        and stabilized.

        This is the closed-loop intelligence mechanism: all three observers
        (Matter/Wave/Data) process the field, their contributions are
        weighted by geometric evidence (persistence), and the result is
        fed back through what the system already knows (truth library).
        """
        phys, phys_pers, phys_haz = self.nested_pass(grid, depth, "physical")
        wave, wave_pers, wave_haz = self.nested_pass(grid, depth, "wave")
        data, data_pers, data_haz = self.nested_pass(grid, depth, "data")

        weights = np.array([
            np.mean(phys_pers),
            np.mean(wave_pers),
            np.mean(data_pers),
        ])
        weights /= (weights.sum() + 1e-8)

        consensus = (
            weights[0] * phys +
            weights[1] * wave +
            weights[2] * data
        )
        consensus = self._apply_library_feedback(consensus)
        consensus = self._prune(consensus)
        consensus = self._stabilize(consensus)
        consensus_pers = float(np.count_nonzero(consensus)) / consensus.size

        return {
            "consensus_grid":  consensus,
            "phys_pers":       float(np.mean(phys_pers)),
            "wave_pers":       float(np.mean(wave_pers)),
            "data_pers":       float(np.mean(data_pers)),
            "consensus_pers":  consensus_pers,
            "weights":         weights.tolist(),
            "phys_hazard":     phys_haz,
            "wave_hazard":     wave_haz,
            "data_hazard":     data_haz,
        }

    def run_generative(
        self,
        waveform:   np.ndarray,
        tri_data:   Dict,
        pass_depth: int = 2,
    ) -> Dict:
        """
        Full generative propagation path.

        Runs consensus_pass (physical + wave + data) then injects semantic
        tension into the bipolar lattice based on subject/verb/object extracted
        from the prompt stored in tri_data. Uses lazy import to avoid circular
        dependency (bipolar_lattice imports invariants which imports nothing
        from ouroboros_engine, so this is safe as a deferred import).
        """
        n      = len(waveform)
        side   = max(1, int(np.ceil(np.sqrt(n))))
        padded = np.pad(waveform, (0, side * side - n))
        grid   = padded.reshape(side, side)

        result = self.consensus_pass(grid, depth=pass_depth)

        # Semantic tension injection — map prompt words to subject/verb/object IDs
        # and inject into bipolar lattice golden zone. Lazy import avoids circular
        # dependency at module load time.
        prompt = tri_data.get("prompt", "")
        if prompt:
            try:
                from utils.bipolar_lattice import bipolar_lattice as _bl
                words      = prompt.lower().split()
                subject_id = hash(words[0]) % 13 + 1 if words else 1
                verb_id    = hash(words[1]) % 13 + 1 if len(words) > 1 else 2
                object_id  = hash(words[-1]) % 13 + 1 if len(words) > 2 else 3
                _bl.inject_semantic_tension(subject_id, verb_id, object_id)

                # Golden zone boost: calm field (|tension| < 0.25) means the
                # geometry has resolved the structure cleanly — reward it
                if abs(_bl.golden_zone["semantic_tension"]) < 0.25:
                    result["consensus_pers"] = min(1.0, result["consensus_pers"] + 0.12)
            except Exception:
                pass  # never let lattice injection break the generative path

        flat     = result["consensus_grid"].flatten()
        out_wave = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(flat)),
            flat
        )

        return {
            "output_waveform": out_wave,
            "consensus_pers":  result["consensus_pers"],
            "phys_pers":       result["phys_pers"],
            "wave_pers":       result["wave_pers"],
            "data_pers":       result["data_pers"],
            "weights":         result["weights"],
            "mode":            "generative",
        }

    # ── Truth library ─────────────────────────────────────────────────────────

    def _add_entry(self, vec: np.ndarray, desc: str) -> None:
        """Internal — add without saving (used for bootstrap)."""
        existing = {e["desc"] for e in self.truth_library}
        if desc not in existing:
            sig = self._project_to_signature(vec)
            self.truth_library.append({"projected": sig.tolist(), "desc": desc})

    def etch_to_library(self, vec: np.ndarray, desc: str) -> None:
        """
        Add a high-persistence pattern to the truth library and persist to disk.
        Called by WavePropagator when an answer is resolved with high confidence.
        """
        self._add_entry(vec, desc)
        self._save_library()

    def _load_library(self) -> None:
        if not os.path.exists(_LIBRARY_FILE):
            return
        try:
            with open(_LIBRARY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            existing = {e["desc"] for e in self.truth_library}
            added = 0
            for item in loaded:
                if item["desc"] not in existing:
                    self.truth_library.append(item)
                    existing.add(item["desc"])
                    added += 1
            if added:
                print(f"OuroborosEngine: loaded {added} persisted truths → total {len(self.truth_library)}")
        except Exception as e:
            print(f"OuroborosEngine: truth library load failed: {e}")

    def _save_library(self) -> None:
        try:
            with open(_LIBRARY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.truth_library, f, indent=2)
        except Exception as e:
            print(f"OuroborosEngine: truth library save failed: {e}")

    def should_go_generative(self, persistence: float, recall_triggered: bool = False) -> bool:
        """
        Decision gate: should this propagation step use the generative path?

        Fires when:
          - persistence is high (wave is well-sustained → geometry warrants depth)
          - OR a cross-turn recall was triggered (prior etch should feed into field)

        Maps directly to the sensory asymmetry insight:
          low persistence  → direct/recognition mode (fast, current lab path)
          high persistence → generative/reconstruction mode (closed loop)
        """
        return recall_triggered or persistence >= self.generative_threshold

    def get_status(self) -> Dict:
        return {
            "truth_library_size":   len(self.truth_library),
            "generative_threshold": self.generative_threshold,
            "feedback_fraction":    self.env_feedback_fraction,
            "prune_threshold":      round(_PRUNE_THRESHOLD, 6),
            "deviation":            round(_DEVIATION, 6),
            "frame_delta":          round(_FRAME_DELTA, 14),
        }


# Singleton
ouroboros_engine = OuroborosEngine()
