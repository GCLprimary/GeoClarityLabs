import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from core.clarity_ratio import clarity_ratio
from utils.radial_displacer import radial_displacer


class Observer:
    def __init__(self):
        pass

    def blend(self, data: np.ndarray, band_mean: float = 21.0) -> np.ndarray:
        # Guard: if signal is near-constant (saturated tanh output from generative
        # path), std ≈ 0 and normalisation explodes. Detect and handle gracefully.
        data_std = float(np.std(data))
        if data_std < 1e-4:
            direction = float(np.mean(data))
            return np.full_like(data, np.clip(direction * 0.5, -1.0, 1.0))

        eq = data - np.mean(data)
        eq = eq / (data_std + 1e-8)

        signal_health = min(1.0, data_std / 0.5)
        boost = (band_mean ** 2) / np.pi * signal_health
        return np.clip(eq * (1 + boost * 0.45), -1.0, 1.0)


class MultiObserver:
    """
    Radicalized asymmetry: Matter / Wave / Data observers.

    Two consensus paths:

    WAVEFORM PATH (original)
    ─────────────────────────
    Used when prop_result has no language fingerprint data.
    Operates on raw holographic linkage waveform amplitude.
    Band means: Matter=21, Wave=10, Data=65.

    LANGUAGE PATH (new)
    ────────────────────
    Used when prop_result contains language fingerprint signals:
      field_direction, field_stress, carry_alignment, fold_coherence,
      stream_mean_tension, vocab_stable, vocab_hits.

    The language path computes consensus from four geometric signals:

      1. DIRECTION SIGNAL — field_direction ('positive'/'negative'/'boundary')
         Maps directly to a signed base: positive=+0.6, negative=-0.6,
         boundary=0.0. This is the primary signal — the field's own
         assessment of which way it's leaning.

      2. ALIGNMENT SIGNAL — carry_alignment [-1, +1]
         How well this sentence resonates with prior context.
         High positive alignment = coherent continuation = boosts consensus.
         High negative alignment = contradiction = suppresses consensus.

      3. STRESS SIGNAL — field_stress [0, ~0.15]
         Low stress = clean, settled structure = higher confidence.
         High stress = contested, complex field = lower confidence.
         Scaled so typical stress (0.03) contributes positively.

      4. PERSISTENCE SIGNAL — persistence [0, 1]
         How well the waveform sustained. High persistence = real structure.
         At persistence 1.0 the field is fully resolved.

    The four signals are combined with weights derived from the
    odd/even asymmetry principle:
      odd signals (direction, stress)   → vertical build (+)
      even signals (alignment, persist) → horizontal recognition (×)

    Final consensus is clipped to [-1, 1] and the sign is preserved
    so the generation layer can still distinguish affirmative/contested.
    """

    def __init__(self, num_observers: int = 3):
        self.num_observers      = num_observers
        self.observers          = [Observer() for _ in range(num_observers)]
        self.bands              = [21.0, 10.0, 65.0]   # Matter / Wave / Data
        self.cumulative_perturb = 0.0
        self.last_consensus_time = time.time()

    # ── Language-aware consensus ──────────────────────────────────────────────

    def _language_consensus(self, prop_result: Dict[str, Any]) -> Optional[float]:
        """
        Compute consensus from language fingerprint signals.

        Returns None if required signals are not present
        (falls through to waveform path).
        """
        direction   = prop_result.get("field_direction")
        stress      = prop_result.get("field_stress")
        alignment   = prop_result.get("carry_alignment")   # may be absent early
        persistence = prop_result.get("persistence", 0.0)
        coherence   = prop_result.get("fold_coherence", 0.5)
        vocab_stable = prop_result.get("vocab_stable", 0)
        vocab_hits   = prop_result.get("vocab_hits", 0)

        # Need at least direction and stress to use language path
        if direction is None or stress is None:
            return None

        # ── Signal 1: Direction (odd/vertical — primary driver) ───────────────
        direction_map = {"positive": +0.6, "negative": -0.6, "boundary": 0.0}
        dir_signal = direction_map.get(direction, 0.0)

        # ── Signal 2: Alignment (even/horizontal — recogniser) ───────────────
        # Alignment may be 0.0 on first sentence (no carry yet) — treat as neutral
        align_signal = float(alignment) if alignment is not None else 0.0

        # ── Signal 3: Stress (odd/vertical — confidence modifier) ────────────
        # Low stress = high confidence. Typical range 0.02–0.05.
        # Map so that stress=0.03 → +0.3, stress=0.10 → ~0, stress>0.15 → negative
        stress_signal = float(np.clip(0.3 - (stress * 10.0), -0.5, 0.5))

        # ── Signal 4: Persistence (even/horizontal — resolution gate) ────────
        # Persistence 1.0 = fully resolved = full weight
        # Persistence < 0.3 = not resolved = suppress
        persist_signal = float(np.clip((persistence - 0.3) / 0.7, 0.0, 1.0))

        # ── Vocabulary boost (structural memory contribution) ─────────────────
        # Named invariants and stable vocab contribute a small positive signal
        # — they represent resolved structure the field already knows
        vocab_boost = float(np.clip(
            (vocab_stable * 0.005) + (vocab_hits * 0.003),
            0.0, 0.15
        ))

        # ── Coherence gate ────────────────────────────────────────────────────
        # Low fold-line coherence means the geometry is still forming.
        # Scale the whole result by coherence so early-session outputs
        # are appropriately uncertain.
        coherence_gate = float(np.clip(0.4 + 0.6 * coherence, 0.4, 1.0))

        # ── Combine: odd signals build, even signals recognise ────────────────
        # Odd (direction + stress): additive vertical contribution
        odd_component  = dir_signal + stress_signal

        # Even (alignment + persistence): multiplicative horizontal gate
        # When alignment is 0 (neutral), even_component = persist_signal
        # When alignment is +1 (fully aligned), even_component amplified
        even_component = persist_signal * (1.0 + 0.5 * align_signal)

        # Net consensus before gating
        raw = (odd_component * 0.6 + even_component * 0.4) * coherence_gate

        # Apply vocab boost in the direction of the field
        if dir_signal != 0.0:
            raw += vocab_boost * np.sign(dir_signal)

        return float(np.clip(raw, -1.0, 1.0))

    # ── Primary interact ──────────────────────────────────────────────────────

    def interact(
        self,
        data:        np.ndarray,
        prompt:      str = "",
        iterations:  int = 10,
        prop_result: Dict[str, Any] = None,
    ) -> Tuple[float, float]:

        if len(data) == 0:
            return 0.0, 0.0

        # ── Try language path first ───────────────────────────────────────────
        if prop_result is not None:
            lang_consensus = self._language_consensus(prop_result)
            if lang_consensus is not None:
                # Language path succeeded — use it as primary consensus
                # Still run the waveform path as a secondary reality-check
                # and blend: language path dominates (0.8), waveform corrects (0.2)
                wave_consensus = self._waveform_consensus(
                    data, prompt, iterations, prop_result
                )
                blended = float(np.clip(
                    lang_consensus * 0.8 + wave_consensus * 0.2,
                    -1.0, 1.0
                ))

                perturb = abs(lang_consensus - wave_consensus)
                self.last_consensus_time  = time.time()
                self.cumulative_perturb   = float(np.clip(
                    self.cumulative_perturb + perturb * 0.3, -1.0, 1.0
                ))
                return blended, self.cumulative_perturb

        # ── Fallback: pure waveform path ──────────────────────────────────────
        wave_consensus = self._waveform_consensus(
            data, prompt, iterations, prop_result
        )
        self.last_consensus_time = time.time()
        return wave_consensus, self.cumulative_perturb

    # ── Waveform path (original logic, extracted) ─────────────────────────────

    def _waveform_consensus(
        self,
        data:        np.ndarray,
        prompt:      str,
        iterations:  int,
        prop_result: Optional[Dict[str, Any]],
    ) -> float:
        prompt_lower = prompt.lower()
        is_question  = (
            prompt_lower.endswith("?") or
            any(w in prompt_lower for w in
                ["does", "do", "is", "are", "what", "who", "how"])
        )
        has_negation = any(
            w in prompt_lower for w in ["not", "no", "never"]
        )

        radial_status    = radial_displacer.get_status()
        convergence_boost = max(0.3, radial_status.get("global_clarity", 1.0) * 0.8)

        is_generative = False
        phys_w = wave_w = data_w = 1.0
        if prop_result is not None and prop_result.get("mode") == "generative":
            is_generative = True
            phys_w = prop_result.get("phys_pers", 1.0)
            wave_w = prop_result.get("wave_pers", 1.0)
            data_w = prop_result.get("data_pers", 1.0)
            total  = phys_w + wave_w + data_w + 1e-8
            phys_w = (phys_w / total) * 3
            wave_w = (wave_w / total) * 3
            data_w = (data_w / total) * 3

        perceptions  = []
        role_weights = [phys_w, wave_w, data_w]

        for i, obs in enumerate(self.observers):
            band_mean = self.bands[i % len(self.bands)]
            perc      = obs.blend(data, band_mean=band_mean)
            amp       = float(np.mean(perc))

            if is_generative:
                amp *= role_weights[i % 3] * convergence_boost
            else:
                if i == 0:
                    amp *= 1.8 * convergence_boost
                elif i == 1:
                    amp *= 1.2 * convergence_boost
                else:
                    amp *= 0.8 * convergence_boost

            if is_question:
                amp *= 1.2 if i % 2 == 0 else 0.85
            if has_negation:
                amp *= -1.2 if i % 2 == 1 else 0.9

            perceptions.append(amp)

        consensus = np.mean(perceptions)
        for _ in range(iterations):
            props     = [p * (1.05 if i % 2 == 0 else 0.95)
                         for i, p in enumerate(perceptions)]
            consensus = float(np.mean(props))

        final_consensus = float(np.clip(consensus, -1.0, 1.0))
        perturb         = float(np.std(perceptions))
        self.cumulative_perturb = float(np.clip(
            self.cumulative_perturb + perturb * 0.6, -1.0, 1.0
        ))
        return final_consensus

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_observers":       self.num_observers,
            "last_consensus_time": self.last_consensus_time,
            "cumulative_perturb":  round(self.cumulative_perturb, 6),
        }
