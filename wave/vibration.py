import numpy as np
from typing import Optional, Dict, Any


class VibrationPropagator:
    """
    Vibration Propagation Module.

    Only holographic_linkage() is active in the pipeline.
    propagate_vibration() and refract() removed — never called.
    """

    def __init__(self, asymmetric_delta: float = 0.01639510239):
        self.asymmetric_delta = asymmetric_delta
        self.base_range       = (-1.0, 1.0)

    def holographic_linkage(
        self,
        data:           np.ndarray,
        position_ratio: float          = 0.5,
        real_freq:      Optional[float] = None,
    ) -> np.ndarray:
        """FFT-based wave binding — called from processor.py."""
        if len(data) == 0:
            return np.array([0.0])
        freq  = real_freq / np.pi if real_freq is not None else 1.0
        chain = np.fft.fft(data)
        realized = np.real(chain * (freq ** 2))
        eq = realized - np.mean(realized)
        eq = eq / (np.std(eq) + 1e-8)
        return np.clip(eq, *self.base_range)

    def get_status(self) -> Dict[str, Any]:
        return {
            "asymmetric_delta": self.asymmetric_delta,
            "base_range":       self.base_range,
        }
