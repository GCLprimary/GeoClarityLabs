import math
from typing import Dict, Any, List
from core.invariants import invariants

# ── Letter → ring position mapping ───────────────────────────────────────────
# 27 positions: '0' at position 0, A-Z at positions 1-26.
# Ring symbol at position n = chr(ord('A') + n - 1) for n > 0, else '0'.
#
# Vowel placement — one vowel anchored near each of the 5 stabilizer zones:
#   Stabilizer positions on ring: 0, 5, 10, 16, 21
#   Vowels placed at: 1(a), 6(e), 11(i), 17(o), 22(u)
#   One vowel per stabilizer neighborhood so every vowel-containing word
#   fires at least one stabilizer. Coverage before discrimination.
#
# Consonant placement — distributed across remaining positions.
# Ordered by natural language frequency (high→low) filling remaining slots
# so high-frequency consonants are evenly spread around the ring.
#
# Weights — derived from dual-13 scale (1.0 = neutral baseline):
#   Vowels: 1.5-1.9 (sustained tones, builders)
#   High-freq consonants: 1.1-1.4
#   Mid-freq consonants: 0.9-1.0
#   Low-freq consonants: 0.6-0.8
#   Space/'0': 0.0 (zero dynamism, no weight)
#
# Weight scale factor: golden_ratio / pi ≈ 0.515 per unit step
# — same ratio as spin step to full rotation. Keeps weights
# within [0, 2] matching _MAX_TENSION bounds.

_LETTER_TO_RING_POS: Dict[str, int] = {
    # ── Vowels — stabilizer zone anchors ──────────────────────────────────
    'a': 1,   # near stabilizer 0 (position 0)
    'e': 6,   # near stabilizer 5 (position 5)
    'i': 11,  # near stabilizer 10 (position 10)
    'o': 17,  # near stabilizer 16 (position 16)
    'u': 22,  # near stabilizer 21 (position 21)
    # ── High-frequency consonants — evenly distributed ────────────────────
    't': 2,
    'n': 3,
    's': 4,
    'r': 7,
    'h': 8,
    'l': 9,
    'd': 12,
    'c': 13,
    'm': 14,
    'f': 15,
    'p': 18,
    'g': 19,
    'b': 20,
    'w': 23,
    'y': 24,
    # ── Low-frequency consonants ───────────────────────────────────────────
    'v': 25,
    'k': 26,
    'j': 16,  # near stabilizer 16 — rare, won't dominate
    'x': 21,  # near stabilizer 21 — rare, won't dominate
    'q': 10,  # near stabilizer 10 — very rare
    'z': 5,   # near stabilizer 5  — very rare
}

# Ring position → symbol character
_RING_POS_TO_SYM: Dict[int, str] = {
    0: '0',
    **{pos: chr(ord('A') + pos - 1) for pos in range(1, 27)}
}

# ── Letter weights — derived from dual-13 scale ───────────────────────────
# Base unit = asymmetric_delta × π × 10 ≈ 0.515
# Vowels at 1.5-1.9, consonants scaled by frequency, rare letters 0.4-0.7
# All weights in [0, 2] matching _MAX_TENSION bounds.
_LETTER_WEIGHT: Dict[str, float] = {
    # Vowels — sustained tones, maximum energy
    'a': 1.9, 'e': 1.8, 'i': 1.7, 'o': 1.6, 'u': 1.5,
    # High-frequency consonants
    't': 1.4, 'n': 1.3, 's': 1.3, 'r': 1.2, 'h': 1.2,
    'l': 1.1, 'd': 1.1, 'c': 1.0, 'm': 1.0, 'f': 1.0,
    # Mid-frequency consonants
    'p': 0.9, 'g': 0.9, 'b': 0.8, 'w': 0.8, 'y': 0.8,
    # Low-frequency consonants
    'v': 0.7, 'k': 0.7, 'j': 0.6, 'x': 0.5, 'q': 0.4, 'z': 0.4,
    # Space / zero — no weight
    '0': 0.0,
}


class SymbolicWave:
    # Tick padding constants — derived from system geometry
    # N_WORD_PAD:   '0' ticks between each word segment (φ rounded = 2)
    #               Gives the lattice a zero-crossing breath between words.
    # N_POCKET_PAD: DYNAMIC — computed per prompt at triangulation time.
    #               Base = round(n_pkt0_words / φ) scaled by charge density.
    #               Formula: round((n_words / φ) × (mean_charge / 4.0))
    #               where mean_charge = mean |ns| of pkt0 symbols (0-13 scale).
    #               Low-charge prompt (mean=1.5): fewer zeros needed.
    #               High-charge prompt (mean=4.0): more zeros needed.
    #               Self-scales: field uses own signal density to calibrate drain.
    N_WORD_PAD   = 2    # φ rounded — per-word boundary breathing room
    # N_POCKET_PAD is computed dynamically in triangulate() — see _pocket_pad()
    _PHI         = 1.61803398
    _AD          = invariants.asymmetric_delta

    """
    Overhauled 27-Symbol Embedder - Tighter, more powerful front-end.
    0 is now a true dynamic reset/pocket/delimiter for long contexts.

    Letter → ring position mapping is intentional:
      Vowels anchored near the 5 stabilizer zones (positions 1,6,11,17,22)
      so every vowel-containing word loads at least one stabilizer neighborhood.
      Consonants distributed across remaining positions by frequency.

    Letter weights derived from dual-13 scale — vowels highest (sustained
    tones), consonants scaled by natural frequency, rare letters lowest.
    Weights applied at ring activation via generate_structure().
    """
    def __init__(self):
        self.name = "SymbolicWave - 27-Symbol Intentional Mapping"

    def _token_to_27_symbol(self, c: str) -> str:
        """
        Map a character to its intentional ring position symbol.
        Lowercase lookup — case-insensitive. Unknown characters map to '0'.
        """
        if not c or c.isspace():
            return '0'
        lower = c.lower()
        pos   = _LETTER_TO_RING_POS.get(lower)
        if pos is None:
            # Digits, punctuation, unknown — treat as zero dynamism
            return '0'
        return _RING_POS_TO_SYM[pos]

    def get_weight(self, c: str) -> float:
        """Return the geometric weight for a character."""
        return _LETTER_WEIGHT.get(c.lower(), 0.5)

    def _insert_pockets(self, text: str) -> List[str]:
        """FORCE hard 0 phase-shift between context and query."""
        text = text.strip()
        if '?' in text:
            parts = text.rsplit('?', 1)
            context = parts[0].strip()
            query = (parts[1].strip() + '?') if parts[1].strip() else '?'
            # Always force 0 after last sentence or at end of context
            if '.' in context or '!' in context:
                last_end = max(context.rfind('.'), context.rfind('!'))
                if last_end != -1:
                    context = context[:last_end + 1] + '0'
                else:
                    context += '0'
            else:
                context += '0'
            text = context + query
        else:
            # For statements, still add a trailing 0 so pockets exist
            if '.' in text or '!' in text:
                last_end = max(text.rfind('.'), text.rfind('!'))
                if last_end != -1:
                    text = text[:last_end + 1] + '0'
                else:
                    text += '0'
            else:
                text += '0'

        # Natural breaks + character-level splitting for longer streams
        segments = []
        current = ""
        for char in text + " ":
            current += char
            if char in ".!?;0" or len(current) > 30 or (char.isspace() and len(current) > 10):
                segments.append(current.strip())
                current = ""
        if current:
            segments.append(current.strip())
        return segments

    def _pocket_pad(self, pkt0_symbols: list) -> int:
        """
        Compute dynamic pocket boundary padding from pkt0 symbol charge density.

        Formula: round((n_words / φ) × (mean_charge / 4.0))
          n_words     = rough word count (pkt0 symbol count / avg_word_len≈5)
          mean_charge = mean |signed value| of pkt0 symbols on 0-13 scale
          φ           = golden ratio (scaling constant)
          4.0         = neutral charge midpoint (mid of 0-13 range)

        This means:
          Low-charge input  (mean≈1.5) → fewer zeros, less drain needed
          High-charge input (mean≈4.0) → more zeros, stronger imprint needs more settling
          Scales with length so short prompts aren't over-drained.

        Minimum: 2 (always some settling)
        Maximum: 20 (cap to prevent over-drain on very long dense prompts)
        """
        if not pkt0_symbols:
            return 2
        # Symbol to signed magnitude (0=0, A-M=1-13, N-Z=1-13)
        _charges = []
        for s in pkt0_symbols:
            if s == '0':
                continue
            if 'A' <= s <= 'M':
                _charges.append(ord(s) - ord('A') + 1)
            elif 'N' <= s <= 'Z':
                _charges.append(ord(s) - ord('N') + 1)
        if not _charges:
            return 2
        mean_charge = sum(_charges) / len(_charges)
        n_words = max(1, len(pkt0_symbols) // 5)  # rough word count
        n_pad = round((n_words / self._PHI) * (mean_charge / 4.0))
        return max(2, min(n_pad, 20))

    def triangulate(self, sequence: List[int] or str) -> Dict[str, Any]:
        if isinstance(sequence, str):
            text = sequence
        else:
            text = "".join(chr(c) for c in sequence if 32 <= c <= 126)

        pockets = self._insert_pockets(text)
        symbol_stream = []
        zero_breaks = []

        # Identify the pocket boundary (context/question split) — it's the pocket
        # that contains the '0' character inserted by _insert_pockets at the '?'
        # boundary. Everything before it is pkt=0, everything after is pkt=1.
        _boundary_idx = None
        for _i, _p in enumerate(pockets):
            if '0' in _p and _i < len(pockets) - 1:
                _boundary_idx = _i
                break

        for i, pocket in enumerate(pockets):
            pocket_symbols = [self._token_to_27_symbol(c) for c in pocket if c]

            # Per-word padding: insert N_WORD_PAD '0' ticks after each word-length
            # segment within pkt=0. Words are separated by spaces — each pocket
            # segment already corresponds roughly to a word or short phrase.
            # We insert the breathing room at segment boundaries.
            if self.N_WORD_PAD > 0 and i < (_boundary_idx or len(pockets)):
                # Add word-boundary zeros after each non-boundary pocket
                symbol_stream.extend(pocket_symbols)
                if i < len(pockets) - 1 and i != _boundary_idx:
                    symbol_stream.extend(['0'] * self.N_WORD_PAD)
            else:
                symbol_stream.extend(pocket_symbols)

            if i < len(pockets) - 1:
                # At the pocket boundary: insert dynamic padding zeros
                if i == _boundary_idx:
                    # Collect pkt0 symbols seen so far for charge calculation
                    _pkt0_syms = [s for s in symbol_stream if s != '0']
                    _n_pad = self._pocket_pad(_pkt0_syms)
                    if _n_pad > 0:
                        symbol_stream.extend(['0'] * _n_pad)
                symbol_stream.append('0')  # original single separator
                zero_breaks.append(len(symbol_stream) - 1)

        n = len(symbol_stream)
        n_adjusted = n + (4 - n % 4) if n % 4 != 0 else n
        width = math.ceil(math.sqrt(n_adjusted))

        if width == 0:
            width = 1
            height = 1
            total_triangles = 0
        else:
            height = math.ceil(n_adjusted / width)
            total_triangles = 2 * (n_adjusted // 4)

        result = {
            "sequence": symbol_stream,
            "n_original": n,
            "n_adjusted": n_adjusted,
            "is_padded": n != n_adjusted,
            "width": width,
            "height": height,
            "total_triangles": total_triangles,
            "square_count": n_adjusted // 4,
            "box_area": width * height,
            "triangulation_type": "27-symbol_pocket_embedder",
            "pockets": pockets,
            "zero_breaks": zero_breaks,
            "symbol_stream": symbol_stream
        }
        return result

    def triangulate_raw(self, sequence: str) -> Dict[str, Any]:
        """
        Raw symbol triangulation — no pocket insertion, no zero-breaks.

        Maps every character directly to the 27-symbol alphabet and
        builds the box geometry from the flat stream. No boundary
        detection, no context/query splitting, no injected zeros.

        Use this for pure symbol testing — measuring what individual
        letters and sequences do to the field without structural
        boundary interference. Pocket confidence will read 0.0
        (correct — no boundary geometry exists in raw input).

        Spaces map to '0' per the standard character mapping.
        """
        symbol_stream = [self._token_to_27_symbol(c) for c in sequence if c]

        n          = len(symbol_stream)
        n_adjusted = n + (4 - n % 4) if n % 4 != 0 else n
        width      = max(1, math.ceil(math.sqrt(n_adjusted)))
        height     = math.ceil(n_adjusted / width)
        total_triangles = 2 * (n_adjusted // 4)

        return {
            "sequence":          symbol_stream,
            "n_original":        n,
            "n_adjusted":        n_adjusted,
            "is_padded":         n != n_adjusted,
            "width":             width,
            "height":            height,
            "total_triangles":   total_triangles,
            "square_count":      n_adjusted // 4,
            "box_area":          width * height,
            "triangulation_type":"raw_flat",
            "pockets":           [sequence],   # whole input as single pocket
            "zero_breaks":       [],           # none — no boundary geometry
            "symbol_stream":     symbol_stream,
        }

    def get_box_summary(self, sequence: List[int] or str) -> str:
        data = self.triangulate(sequence)
        return (f"Box: {data['width']}×{data['height']} | Triangles: {data['total_triangles']} | "
                f"Pockets: {len(data['pockets'])} | Zero breaks: {len(data['zero_breaks'])}")

# Quick self-test
if __name__ == "__main__":
    sw = SymbolicWave()
    test_prompt = "Adam likes to eat oranges. What does Adam like to eat?"
    result = sw.triangulate(test_prompt)
    print(sw.get_box_summary(test_prompt))
    print("Symbol stream:", result["symbol_stream"])
    print("Zero breaks:", result["zero_breaks"])