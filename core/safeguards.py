from typing import Dict, Any


class Safeguards:
    """
    Anti-grandiosity safeguard — tracks reversal events and clarity ceiling.

    apply_reversal_trigger and should_force_reversal removed (only called
    from triad.py which is no longer part of the pipeline).
    check_re_derivation and is_mimic removed (never called anywhere).
    """

    def __init__(self):
        self.reversal_count    = 0
        self.max_local_clarity = 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "reversal_count":    self.reversal_count,
            "max_local_clarity": round(self.max_local_clarity, 4),
        }


safeguards = Safeguards()
