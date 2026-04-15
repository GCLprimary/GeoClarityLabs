from typing import Dict, Any
import numpy as np

class ClarityRatio:
    """
    Clarity Ratio Tracker - Clarity Ratio Laboratory
    
    Measures and tracks increases in perceptual resolution vs unclarity.
    """

    def __init__(self):
        self.history = []
        self.current_ratio = 0.0

    def measure(self, width: int, height: int, total_triangles: int, n: int) -> float:
        """
        Compute clarity ratio score based on box dimensions and triangle density.
        Higher = better resolved structure.
        """
        if n == 0:
            return 0.0
            
        resolution = (width * height) / max(n, 1)
        density_factor = total_triangles / max(n, 1)
        
        # Combined clarity ratio score
        score = resolution * (1 + 0.3 * density_factor)
        
        self.history.append(score)
        self.current_ratio = score
        
        return round(score, 4)

    def get_trend(self) -> Dict[str, Any]:
        """Return trend information about clarity ratio over time."""
        if len(self.history) < 2:
            return {"trend": "insufficient_data", "current": self.current_ratio}
        
        recent = self.history[-5:]
        trend = "increasing" if recent[-1] > np.mean(recent[:-1]) else "stable_or_decreasing"
        
        return {
            "current": round(self.current_ratio, 4),
            "trend": trend,
            "improvement_rate": round((self.current_ratio - self.history[0]) / len(self.history), 4)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_ratio": round(self.current_ratio, 4),
            "history_length": len(self.history)
        }


# Singleton instance
clarity_ratio = ClarityRatio()