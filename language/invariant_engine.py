"""
language/invariant_engine.py
=============================
Invariant Formation — Three Integrated Stages

STAGE 1 — STABLE GROUP NAMING
──────────────────────────────
When a word crosses the stability threshold in SessionVocabulary, its
geometric fingerprint (symbol stream tension profile projected to a
numpy vector) is etched into the Ouroboros truth library as a named
invariant. Format: "word::{word}" e.g. "word::the", "word::are".

Named invariants become fixed attractors. On every subsequent generative
pass, _apply_library_feedback injects their FFT signatures back into the
field. Words the system has genuinely resolved now shape how it processes
new input — without any explicit lookup. The geometry carries the memory.

This is "appropriate perceptual clarity reduces avoidable suffering"
made concrete: stable structure reinforces future resolution. Unstable
patterns don't get etched, so noise doesn't accumulate.

STAGE 2 — NON-LOCAL DECAY
──────────────────────────
All active groups decay by the same ratio on each pipeline call.
The ratio is derived from asymmetric_delta / mersenne_prime — the same
subtraction base the 52 Mersenne strings use.

Effect:
  - Named invariants (high centroid, many appearances) survive longer
    because their centroid is continuously reinforced by new matches
  - Novel words start with full weight but decay toward zero if they
    never stabilise
  - The system self-prunes noise without any threshold tuning

This is the radioactive decay insight implemented: the system doesn't
choose what decays. It applies the same ratio across all phases. What
sustains meaning survives. What doesn't, fades.

STAGE 3 — SPIN-DRIVEN GENERATION
──────────────────────────────────
spin_phase and spin_sign from fold_line_resonance drive generation mode:

  spin_sign = +1  (vertical build, odds dominant)
    → direct recognition mode
    → system reports what it has resolved with high confidence
    → answers are definitive when field supports it

  spin_sign = -1  (horizontal observer, evens dominant)
    → reconstruction mode
    → system reports the field geometry explicitly
    → answers surface the structural evidence rather than conclusions

  spin_phase near 0 or π  (boundary zone)
    → transitional mode
    → system reports the transition explicitly
    → neither definitive nor pure geometry — acknowledges the boundary

This maps the odd/even asymmetry from the design notes directly to
generation behaviour. The system's "personality" at any moment is
determined by where the spin is in its cycle — not by a prompt template.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional

from core.invariants import invariants
from core.ouroboros_engine import ouroboros_engine
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping, symbol_to_signed

# ── Decay constants ───────────────────────────────────────────────────────────
# Non-local decay ratio — same Mersenne subtraction base as the 52 strings
# Using mersenne prime 7 as the denominator (middle of the four primes)
_DECAY_MERSENNE    = 7
_DECAY_RATIO       = 1.0 - (invariants.asymmetric_delta / _DECAY_MERSENNE)
# ≈ 0.99766 — very slow decay so stable words persist across many sentences

# Minimum centroid before a word is considered for naming
_NAMING_THRESHOLD  = 0.15

# Spin phase boundary tolerance — within this of 0 or π = boundary zone
_BOUNDARY_TOLERANCE = math.pi / 8   # 22.5 degrees


class InvariantEngine:
    """
    Manages naming, decay, and spin-driven generation.
    Singleton — shared across the language processor and pipeline.
    """

    _NO_NAME = {
        # length < 3
        "a", "an", "as", "at", "be", "by", "do", "if", "in",
        "is", "it", "of", "on", "or", "so", "to", "up",
        # 3-letter structural/connective
        "the", "and", "are", "but", "can", "did", "for", "had",
        "has", "how", "its", "may", "not", "now", "was", "will",
        "yet", "did", "got", "let", "put", "set", "got", "nor",
        # 4-letter structural
        "also", "been", "both", "does", "done", "each", "else",
        "even", "from", "have", "here", "just", "like", "made",
        "many", "more", "most", "much", "must", "once", "only",
        "over", "same", "such", "than", "that", "them", "then",
        "they", "this", "thus", "very", "what", "when", "whom",
        "with", "been", "whom", "were", "whom", "said", "whom",
        # 4-letter prepositions/connectives observed being named
        "into", "onto", "upon", "over", "also", "even", "just",
        "once", "then", "than", "been", "will", "does", "your",
        "they", "them", "each", "such", "both",
        # Structural adverbs — high frequency, zero domain content
        "always", "never", "often", "usually", "mostly", "rarely",
        "already", "simply", "merely", "really", "quite", "rather",
        "nearly", "almost", "barely", "surely", "truly", "fully",
        # 5-letter
        "their", "there", "these", "those", "which", "while",
        "would", "could", "shall", "might", "about", "after",
        "again", "below", "every", "first", "other", "where",
        "still", "until", "under",
        # Common verbs/nouns that repeat but carry no domain geometry
        # These were being named from Q-only outputs contaminating the library
        "things", "causes", "happen", "happens", "form", "forms",
        "need", "needs", "learn", "learns", "make", "makes",
        "use", "uses", "used", "show", "shows", "work", "works",
        "come", "goes", "give", "gives", "take", "takes", "keep",
        "thing", "cause", "place", "point", "part", "area", "case",
        # Question-output artifacts that shouldn't be named
        "during", "approach", "important", "around", "between",
    }

    def __init__(self):
        self.named_invariants: Dict[str, Dict[str, Any]] = {}
        # Load any previously named invariants from the truth library
        self._load_from_library()

    # ── Stage 1: Stable Group Naming ─────────────────────────────────────────

    def _word_to_vector(self, word: str, symbol_stream: List[str]) -> np.ndarray:
        """
        Project a word's symbol stream tension profile to a numpy vector
        suitable for FFT signature projection in the truth library.

        The vector encodes:
          - signed values of each symbol (geometric address)
          - pair tensions across adjacent symbols
          - position weights (earlier symbols weighted higher — source drives)

        Length is padded/truncated to 32 for consistent FFT projection.
        """
        _VEC_LEN = 32
        vec      = np.zeros(_VEC_LEN, dtype=float)

        # Remove zero breaks from the stream for this word
        syms = [s for s in symbol_stream if s != '0']
        if not syms:
            return vec

        for i, sym in enumerate(syms[:_VEC_LEN]):
            v            = symbol_to_signed(sym)
            scale        = 0.8 if abs(v) % 2 == 1 else 0.6
            pos_weight   = 1.0 / (1.0 + 0.1 * i)   # source emphasis
            vec[i]       = (v / 13.0) * scale * pos_weight

        # Add pair tensions as second-order features
        for i in range(min(len(syms) - 1, _VEC_LEN // 2)):
            pt = symbol_grouping.pair_tension(syms[i], syms[i+1])
            idx = _VEC_LEN // 2 + i
            if idx < _VEC_LEN:
                vec[idx] = pt["tension"]

        # Normalise
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        return vec

    def try_name_word(
        self,
        word:          str,
        symbol_stream: List[str],
        appearances:   int,
        familiarity:   float,
        centroid:      float,
    ) -> bool:
        """
        Attempt to name a word as a stable invariant.

        Conditions:
          1. Word not already named
          2. Appearances >= 2 (seen it more than once)
          3. Familiarity >= 0.65 (consistent geometric response)
          4. Centroid >= _NAMING_THRESHOLD (fold line has genuinely imprinted it)

        On success: etches to Ouroboros truth library, stores in named_invariants.
        Returns True if named, False otherwise.
        """
        word_key = f"word::{word.lower()}"

        if word_key in self.named_invariants:
            return False   # already named

        # Structural word guard — these words must never become named invariants.
        # They are connective tissue or high-frequency function words that appear
        # in every prompt and have no domain-specific geometric meaning.
        if len(word.strip()) < 3 or word.lower().strip() in self._NO_NAME:
            return False

        if appearances < 2 or familiarity < 0.65 or centroid < _NAMING_THRESHOLD:
            return False

        vec = self._word_to_vector(word, symbol_stream)
        ouroboros_engine.etch_to_library(vec, word_key)

        self.named_invariants[word_key] = {
            "word":        word.lower(),
            "appearances": appearances,
            "familiarity": familiarity,
            "centroid":    centroid,
            "vector_norm": float(np.linalg.norm(vec)),
        }

        print(f"InvariantEngine: named '{word.lower()}' → truth library "
              f"(appearances={appearances}, familiarity={familiarity:.3f})")
        return True

    def _load_from_library(self) -> None:
        """Restore previously named invariants from truth library entries."""
        for entry in ouroboros_engine.truth_library:
            desc = entry.get("desc", "")
            if desc.startswith("word::"):
                word = desc[6:]
                # Skip words in _NO_NAME — they may have been named before
                # the blocklist was updated. Purge them on load.
                if word in self._NO_NAME:
                    continue
                if desc not in self.named_invariants:
                    self.named_invariants[desc] = {
                        "word":        word,
                        "appearances": 0,
                        "familiarity": 1.0,   # already stable — fully trusted
                        "centroid":    1.0,
                        "vector_norm": 0.0,
                    }

    def is_named(self, word: str) -> bool:
        return f"word::{word.lower()}" in self.named_invariants

    def get_named_words(self) -> List[str]:
        return [v["word"] for v in self.named_invariants.values()]

    # ── Stage 2: Non-Local Decay ──────────────────────────────────────────────

    def apply_decay(self, groups: List[Any]) -> None:
        """
        Apply non-local decay to all group tension centroids.

        Every group decays by _DECAY_RATIO each pipeline call.
        Named invariants are re-reinforced before decay — they get a
        small centroid boost proportional to their familiarity score,
        which counteracts the decay and keeps them stable.

        Unnamed groups simply decay. If their centroid falls below
        _IMPRINT_THRESHOLD they will eventually fall out of the active
        group set on the next recompute — natural pruning.
        """
        for grp in groups:
            # Named invariants get a stability boost before decay
            is_named_grp = any(
                self.is_named(m) for m in grp.members
            )
            if is_named_grp:
                # Boost: small additive reinforcement
                boost = invariants.asymmetric_delta * 0.5
                grp.tension_centroid = float(min(
                    1.0, grp.tension_centroid + boost
                ))

            # Apply non-local decay — same ratio for all
            grp.tension_centroid = float(
                grp.tension_centroid * _DECAY_RATIO
            )

    # ── Stage 3: Spin-Driven Generation ──────────────────────────────────────

    def get_generation_mode(self) -> Dict[str, Any]:
        """
        Determine current generation mode from fold line resolution state.

        Uses the resolution score — not raw phase proximity — as the primary
        signal. Phase proximity only contributes to boundary detection when
        the resolution score is genuinely near the threshold.

        Returns a dict with:
          mode:             'recognition' | 'reconstruction' | 'boundary'
          spin_sign:        +1 or -1
          spin_phase:       current phase in [0, 2π)
          resolution_score: field resolution quality [0, 1]
          description:      human-readable mode description
          confidence_scale: how much to trust field output (0–1)
        """
        phase      = fold_line_resonance.spin_phase
        sign       = fold_line_resonance.spin_sign
        coh        = fold_line_resonance.get_coherence_signal()
        resolution = fold_line_resonance.get_resolution_score()

        _RESOLUTION_THRESHOLD = 0.45
        _BOUNDARY_BAND        = 0.10   # ±0.10 around threshold = genuine boundary

        near_threshold = abs(resolution - _RESOLUTION_THRESHOLD) < _BOUNDARY_BAND

        if near_threshold:
            return {
                "mode":             "boundary",
                "spin_sign":        sign,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"field in transition (resolution {resolution:.3f})",
                "confidence_scale": 0.5,
            }

        if resolution >= _RESOLUTION_THRESHOLD:
            return {
                "mode":             "recognition",
                "spin_sign":        +1,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"vertical build — direct recognition (resolution {resolution:.3f})",
                "confidence_scale": min(1.0, 0.6 + 0.4 * coh),
            }
        else:
            return {
                "mode":             "reconstruction",
                "spin_sign":        -1,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"horizontal observer — reconstruction (resolution {resolution:.3f})",
                "confidence_scale": min(1.0, 0.4 + 0.6 * coh),
            }

    def generate_response(
        self,
        fingerprint:    Dict[str, Any],
        base_answer:    str,
        consensus:      float,
        persistence:    float,
        vocab_hits:     List[Dict[str, Any]],
    ) -> str:
        """
        Spin-driven response generation.

        Takes the base answer from the existing generator and modulates
        it based on spin mode, fingerprint direction, named invariant
        hits, and field state.

        Recognition mode  → direct, confident when field supports it
        Reconstruction    → surfaces the geometric evidence explicitly
        Boundary mode     → acknowledges the transition honestly

        This replaces the fallback phrases with geometrically honest
        statements about what the field actually found.
        """
        gen_mode = self.get_generation_mode()
        mode     = gen_mode["mode"]
        conf     = gen_mode["confidence_scale"]
        direction = fingerprint.get("direction", "boundary")
        stress    = fingerprint.get("field_stress", 0.0)
        named_hits = [h["word"] for h in vocab_hits if self.is_named(h["word"])]

        # Skip modulation if base answer already has real content
        # (i.e. it came from context extraction, not geometric fallback)
        geometric_fallbacks = [
            "radial", "manifold", "geometric pattern", "field has not yet",
            "field processing", "boundary detected", "field resolved",
            "geometry stable", "convergence"
        ]
        has_real_content = not any(
            f.lower() in base_answer.lower() for f in geometric_fallbacks
        )
        if has_real_content:
            # Real content — just append named invariant context if relevant
            if named_hits and len(named_hits) >= 2:
                return (f"{base_answer} "
                        f"[anchored: {', '.join(named_hits[:3])}]")
            return base_answer

        # ── Recognition mode ──────────────────────────────────────────────────
        if mode == "recognition":
            if persistence >= 0.8 and abs(consensus) > 0.15:
                polarity = "affirmative" if consensus > 0 else "contested"
                stress_note = " High structural complexity." if stress > 0.04 else ""
                named_note  = (f" Known anchors: {', '.join(named_hits)}."
                               if named_hits else "")
                return (f"Field resolved ({polarity}). "
                        f"Direction: {direction}. "
                        f"Persistence {persistence:.2f}, "
                        f"consensus {consensus:+.3f}.{stress_note}{named_note}")
            elif persistence >= 0.5:
                return (f"Field partially resolved. "
                        f"Direction: {direction}, stress {stress:.4f}. "
                        f"Confidence: {conf:.2f}.")
            else:
                return (f"Field forming. "
                        f"Insufficient persistence ({persistence:.2f}) "
                        f"for resolution in recognition mode.")

        # ── Reconstruction mode ───────────────────────────────────────────────
        elif mode == "reconstruction":
            per_word = fingerprint.get("per_word", [])
            # Surface the highest net_signed words — these are the structural
            # load-bearers the field found
            load_bearers = sorted(
                [w for w in per_word if abs(w.get("net_signed", 0)) > 0.5],
                key=lambda w: abs(w.get("net_signed", 0)),
                reverse=True
            )[:3]
            bearer_str = ", ".join(
                f"{w['word']}({w['net_signed']:+.2f})"
                for w in load_bearers
            ) if load_bearers else "none identified"

            peak = fingerprint.get("peak_pair", ("?", "?"))
            peak_t = fingerprint.get("peak_tension", 0.0)

            return (f"Reconstruction pass. "
                    f"Structural load-bearers: {bearer_str}. "
                    f"Peak interaction: {peak[0]}→{peak[1]} "
                    f"({peak_t:+.4f}). "
                    f"Net field: {direction} "
                    f"(tension {fingerprint.get('net_tension', 0.0):+.4f}).")

        # ── Boundary mode ─────────────────────────────────────────────────────
        else:
            phase = gen_mode["spin_phase"]
            near  = "zero" if phase < math.pi / 2 or phase > 3 * math.pi / 2 else "π"
            return (f"Field at phase boundary (near {near}, "
                    f"phase={phase:.3f}). "
                    f"Direction: {direction}. "
                    f"Transition state — resolution pending next spin cycle.")

    def get_status(self) -> Dict[str, Any]:
        gen_mode = self.get_generation_mode()
        return {
            "named_invariants":   len(self.named_invariants),
            "named_words":        self.get_named_words(),
            "decay_ratio":        round(_DECAY_RATIO, 6),
            "generation_mode":    gen_mode["mode"],
            "spin_description":   gen_mode["description"],
            "resolution_score":   gen_mode["resolution_score"],
            "confidence_scale":   gen_mode["confidence_scale"],
        }


# Singleton
invariant_engine = InvariantEngine()
