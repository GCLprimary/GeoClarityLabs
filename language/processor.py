"""
language/processor.py  (v4 — exhaust-diagonal recall wired)
=============================================================
Changes from v3:

  WIRED: exhaust -> diagonal_structure -> nearest -> generation

  After apply_tension_cycle runs (and before the answer generator),
  the processor now:

  1. Calls diagonal_structure_generator.generate() with the current
     exhaust signature and ring phase, producing a DiagonalStructure
     for this sentence and storing it in session history.

  2. Calls diagonal_structure_generator.nearest() to find the most
     geometrically similar prior structure in session + cross-session
     history (loaded from exhaust_memory.json on startup via
     bipolar_lattice._load_exhaust_memory).

  3. Cross-references the nearest diagonal match against
     bipolar_lattice.nearest_exhaust() (Euclidean distance in exhaust
     space) to get the prior prompt text and distance.

  4. Passes exhaust_recall dict into generator.generate() so step 0
     of _resolve() can use it.

  The diagonal structure generator maintains its own session history
  in-memory. The bipolar lattice maintains cross-session history on
  disk (exhaust_memory.json). Together they cover both short-term
  structural memory (session diagonals) and long-term geometric
  fingerprint memory (persisted exhaust signatures).
"""

import math
import re
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import hashlib

from wave.symbolic_wave import SymbolicWave
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping, symbol_to_signed
from utils.bipolar_lattice import bipolar_lattice
from utils.diagonal_structure import diagonal_structure_generator
from core.clarity_ratio import clarity_ratio
from core.invariants import invariants
from core.ouroboros_engine import ouroboros_engine
from observer.observer import MultiObserver
from wave.generation import generator
from language.invariant_engine import invariant_engine
from language.relational_tension import relational_tension
from language.geometric_output import geometric_output

_VOCAB_STABILITY_THRESHOLD  = 2
_FAMILIARITY_THRESHOLD      = 0.65
_ETCH_PERSISTENCE_THRESHOLD = 0.7

def context_similarity(ctx_a, ctx_b):
    words_a = set(ctx_a.split())
    words_b = set(ctx_b.split())

    def bigrams(s):
        words = s.split()
        return set(zip(words, words[1:]))

    b1 = bigrams(ctx_a)
    b2 = bigrams(ctx_b)

    # word overlap
    word_overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

    # structure overlap
    if not b1 or not b2:
        bigram_overlap = 1.0
    else:
        bigram_overlap = len(b1 & b2) / max(len(b1 | b2), 1)

    # combine them
    return 0.7 * word_overlap + 0.3 * bigram_overlap

def make_context_key(sentence: str) -> str:
    normalized = " ".join(sentence.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]

def is_nonsense(text: str) -> bool:
    words = text.split()

    if len(words) == 0:
        return True

    lowered = text.lower()
    vowel_count = sum(c in "aeiou" for c in lowered if c.isalpha())
    alpha_count = sum(c.isalpha() for c in lowered)
    vowel_ratio = vowel_count / max(alpha_count, 1)

    no_vowel_words = sum(
        1 for w in words
        if any(ch.isalpha() for ch in w) and not any(v in w.lower() for v in "aeiou")
    )

    short_alpha_words = [
        w for w in words
        if sum(ch.isalpha() for ch in w) >= 3
    ]
    weird_cluster_words = sum(
        1 for w in short_alpha_words
        if not any(v in w.lower() for v in "aeiou")
    )

    if vowel_ratio < 0.20:
        return True

    if words and no_vowel_words > len(words) * 0.60:
        return True

    if short_alpha_words and weird_cluster_words > len(short_alpha_words) * 0.60:
        return True

    return False

class WordFingerprint:
    def __init__(
        self,
        word: str,
        symbol_stream: List[str],
        tensions: List[float],
        group_ids: List[int],
        net_signed: float,
        session_epoch: int = 0,
    ):
        self.word          = word.lower()
        self.symbol_stream = symbol_stream
        self.tensions      = tensions
        self.mean_tension  = float(np.mean(tensions)) if tensions else 0.0
        self.group_ids     = group_ids
        self.dominant_group = max(set(group_ids), key=group_ids.count) if group_ids else -1
        self.net_signed    = net_signed
        self.session_epoch = session_epoch   # which session this word was first seen in
        self.timestamp     = time.time()
        self.appearances   = 1
        self.context_keys  = set()

    def similarity(self, other: "WordFingerprint") -> float:
        s1 = set(self.symbol_stream)
        s2 = set(other.symbol_stream)
        sym_overlap  = len(s1 & s2) / max(len(s1 | s2), 1)
        tension_diff = abs(self.mean_tension - other.mean_tension)
        tension_sim  = max(0.0, 1.0 - (tension_diff / 2.0))
        group_match  = 1.0 if self.dominant_group == other.dominant_group else 0.0
        return round(0.4 * sym_overlap + 0.4 * tension_sim + 0.2 * group_match, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word":           self.word,
            "mean_tension":   round(self.mean_tension, 4),
            "dominant_group": self.dominant_group,
            "net_signed":     round(self.net_signed, 4),
            "appearances":    self.appearances,
            "session_epoch":  self.session_epoch,
        }


class SessionVocabulary:
    def __init__(self):
        self._store: Dict[str, WordFingerprint] = {}

    def lookup(self, word: str) -> Optional[WordFingerprint]:
        return self._store.get(word.lower())

    def update(self, fp: WordFingerprint, context_key: str) -> Tuple[float, bool, int]:
        word     = fp.word
        existing = self._store.get(word)

        if existing is None:
            fp.context_keys.add(context_key)
            self._store[word] = fp
            return 0.0, False, 1

        similarity = existing.similarity(fp)
        existing.appearances += 1

        if not hasattr(existing, "context_keys"):
            existing.context_keys = set()

        existing.context_keys.add(context_key)

        contexts = list(existing.context_keys)
        distinct_contexts = len(contexts)

        # --- NEW: context similarity check ---
        similarities = []

        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                sim = context_similarity(contexts[i], contexts[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # If contexts are too similar, collapse them
        if avg_similarity > 0.65:
            distinct_contexts = 1

        # Original behavior
        if distinct_contexts == 1:
            similarity *= 0.5

        n = existing.appearances
        existing.mean_tension = (existing.mean_tension * (n - 1) + fp.mean_tension) / n

        if fp.dominant_group != existing.dominant_group and existing.appearances <= 2:
            existing.dominant_group = fp.dominant_group

        is_stable = (
            existing.appearances >= _VOCAB_STABILITY_THRESHOLD
            and similarity >= _FAMILIARITY_THRESHOLD
            and distinct_contexts >= 2
        )

        return similarity, is_stable, distinct_contexts

    def get_stable_words(self) -> List[Dict[str, Any]]:
        return [
            fp.to_dict() for fp in self._store.values()
            if fp.appearances >= _VOCAB_STABILITY_THRESHOLD
        ]

    def size(self) -> int:
        return len(self._store)

    def stable_count(self) -> int:
        return sum(
            1 for fp in self._store.values()
            if fp.appearances >= _VOCAB_STABILITY_THRESHOLD
        )


# Session epoch — incremented at each startup, stored in every WordFingerprint.
# Words from a different epoch are cross-session; same epoch = current session.
import os as _os
_SESSION_EPOCH_FILE = "session_epoch.txt"
def _load_session_epoch() -> int:
    try:
        with open(_SESSION_EPOCH_FILE) as f:
            epoch = int(f.read().strip()) + 1
    except Exception:
        epoch = 0
    with open(_SESSION_EPOCH_FILE, "w") as f:
        f.write(str(epoch))
    return epoch

_CURRENT_SESSION_EPOCH: int = _load_session_epoch()


class LanguageProcessor:
    def __init__(self):
        self.vocabulary     = SessionVocabulary()
        self.sw             = SymbolicWave()
        self._process_count = 0
        self._session_epoch = _CURRENT_SESSION_EPOCH

    def _fingerprint_word(self, word: str) -> WordFingerprint:
        # Strip punctuation before geometric processing
        # Prevents 'insects?' 'temperatures?' being named as invariants
        word = word.strip().rstrip('?!.,;:"\'').lstrip('"\'(')
        if not word:
            word = '_'
        stream = [self.sw._token_to_27_symbol(c) for c in word if c and not c.isspace()]
        stream = [s for s in stream if s != chr(48)]
        if not stream:
            return WordFingerprint(word, [], [], [], 0.0)
        tensions  = []
        group_ids = []
        net_sv    = 0.0
        for i in range(len(stream) - 1):
            s1, s2 = stream[i], stream[i + 1]
            pt = symbol_grouping.pair_tension(s1, s2)
            tensions.append(pt["tension"])
            if pt.get("group_ids"):
                group_ids.extend(pt["group_ids"])
        if not tensions and stream:
            v     = symbol_to_signed(stream[0])
            scale = 0.8 if abs(v) % 2 == 1 else 0.6
            grp   = symbol_grouping.group_for(stream[0])
            c     = grp.tension_centroid if grp else 0.1
            weight = max(0.1, 1.0 - (1.0 - c) ** 2)
            tensions.append((v / 13.0) * scale * weight)
            if grp:
                group_ids.append(grp.group_id)
        for sym in stream:
            net_sv += symbol_to_signed(sym) / 13.0
        return WordFingerprint(word, stream, tensions, group_ids, net_sv,
                               session_epoch=self._session_epoch)

    def _fingerprint_sentence(
        self,
        sentence:      str,
        symbol_stream: List[str],
        stream_ctx:    Dict[str, Any],
        word_fps:      List[WordFingerprint],
    ) -> Dict[str, Any]:
        tensions = stream_ctx.get("tensions", [])
        profile  = stream_ctx.get("tension_profile", [])

        net_tension = float(np.sum(tensions)) if tensions else 0.0
        direction   = (
            "positive" if net_tension > 0.05 else
            "negative" if net_tension < -0.05 else
            "boundary"
        )

        if tensions:
            peak_idx  = int(np.argmax(np.abs(tensions)))
            peak_val  = tensions[peak_idx]
            non_zero  = [(i, s) for i, s in enumerate(symbol_stream) if s != chr(48)]
            peak_pair = (
                (non_zero[peak_idx][1], non_zero[peak_idx + 1][1])
                if peak_idx < len(non_zero) - 1 else ("?", "?")
            )
        else:
            peak_val, peak_pair = 0.0, ("?", "?")

        all_group_ids = []
        for wfp in word_fps:
            all_group_ids.extend(wfp.group_ids)
        group_counts: Dict[int, int] = {}
        for gid in all_group_ids:
            group_counts[gid] = group_counts.get(gid, 0) + 1
        top_groups   = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        field_stress = float(np.std(tensions)) if len(tensions) > 1 else 0.0
        boundary_count = len(stream_ctx.get("zero_boundaries", []))

        # Pocket-side tagging
        # _insert_pockets splits the text into short segments (~10 chars each)
        # AND inserts an explicit '0' segment at the sentence boundary.
        # zero_breaks has one entry per segment boundary — most are short-segment
        # breaks, but one is the sentence-boundary break.
        #
        # Strategy: find the zero_break just BEFORE the '0' pocket.
        # The pockets list is stored in tri_data["pockets"]. Find the pocket
        # that is exactly '0' (the sentence boundary marker), then take the
        # zero_break at index pocket_index-1.
        #
        # Fallback: use the last zero_break (tends to be near sentence boundary).
        zero_boundaries = stream_ctx.get("zero_boundaries", [])
        raw_breaks      = stream_ctx.get("_zero_breaks_raw", [])
        raw_pockets     = stream_ctx.get("_pockets_raw", [])

        split_sym_idx = len(symbol_stream)  # default: no split
        if raw_breaks and raw_pockets:
            # Find the pocket that is exactly '0' (sentence boundary marker)
            for pi, pocket_text in enumerate(raw_pockets):
                if pocket_text == '0' and pi > 0 and (pi - 1) < len(raw_breaks):
                    split_sym_idx = raw_breaks[pi - 1]
                    break
            else:
                # Fallback: last break before any '?' pocket
                for pi, pocket_text in enumerate(raw_pockets):
                    if '?' in pocket_text and pi > 0 and (pi - 1) < len(raw_breaks):
                        split_sym_idx = raw_breaks[pi - 1]
                        break
        elif zero_boundaries:
            split_sym_idx = zero_boundaries[-2] if len(zero_boundaries) >= 2 else zero_boundaries[-1]
        per_word_dicts  = []
        sym_cursor      = 0
        for wfp in word_fps:
            d             = wfp.to_dict()
            # Use full word length (chars including punctuation) as stream advance.
            # Each char maps to exactly one stream position, so this keeps sym_cursor
            # on the same scale as split_sym_idx from zero_breaks (full stream index).
            full_word_len = len(wfp.word)
            d["pocket"]   = 0 if (sym_cursor + full_word_len // 2) < split_sym_idx else 1
            sym_cursor   += full_word_len + 1   # +1 for inter-word space
            per_word_dicts.append(d)

        context_groups:  set = set()
        question_groups: set = set()
        for d in per_word_dicts:
            gid = d.get("dominant_group", -1)
            if gid < 0:
                continue
            (context_groups if d["pocket"] == 0 else question_groups).add(gid)

        return {
            "sentence":               sentence,
            "word_count":             len(word_fps),
            "symbol_count":           len(symbol_stream),
            "boundary_count":         boundary_count,
            "mean_tension":           round(stream_ctx.get("mean_tension", 0.0), 4),
            "net_tension":            round(net_tension, 4),
            "direction":              direction,
            "field_stress":           round(field_stress, 4),
            "peak_tension":           round(peak_val, 4),
            "peak_pair":              peak_pair,
            "top_groups":             top_groups,
            "coherence":              stream_ctx.get("coherence_used", 0.0),
            "tension_profile":        profile,
            "per_word":               per_word_dicts,
            "context_groups":         context_groups,
            "question_groups":        question_groups,
            "answer_candidate_groups": context_groups - question_groups,
        }

    def process(self, sentence: str) -> Dict[str, Any]:
        self._process_count += 1
        start_time = time.time()

        sentence = sentence.strip()
        nonsense_flag = is_nonsense(sentence)
        penalty = 0.5 if nonsense_flag else 1.0
        context_key = sentence.lower()

        if nonsense_flag:
            print("[warning] Input detected as low-signal / possible nonsense")

        # Reset exhaust for clean per-sentence signature
        bipolar_lattice.reset_exhaust()

        # ── Need 3: Question-only detection + context priming ─────────────
        _was_primed    = False
        _context_words = []
        try:
            from language.conversation_field import conversation_field
            if conversation_field.is_question_only(sentence):
                _primed, _was_primed, _context_words = conversation_field.prime(sentence)
                if _was_primed:
                    sentence = _primed
        except Exception:
            pass

        tri_data      = self.sw.triangulate(sentence)
        tri_data["prompt"] = sentence
        symbol_stream = tri_data.get("symbol_stream", [])

        prior_carry     = relational_tension.get_current_carry()
        carry_direction = relational_tension.get_carry_direction()

        prop        = WavePropagator()
        prop_result = prop.propagate(tri_data, steps=60)
        recall_triggered = len(generator.memory_store) > 0
        if ouroboros_engine.should_go_generative(prop_result["persistence"], recall_triggered):
            prop_result = prop.propagate_generative(prop_result, tri_data, recall_triggered)

        numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                        if isinstance(x, (int, float))]
        wave_amp = float(np.mean(np.abs(numeric_wave))) if numeric_wave else 0.1
        for _ in range(6):
            fold_line_resonance.tick(external_wave_amp=wave_amp)

        bipolar_lattice.react_to_wave(np.array(numeric_wave))
        for _ in range(4):
            bipolar_lattice.apply_tension_cycle(wave_amp)
        linked_wave = bipolar_lattice.band_emit_and_core_propagate(tri_data)
        wave_amp    = float(np.mean(np.abs(linked_wave)))

        clarity_ratio.measure(
            tri_data["width"], tri_data["height"],
            tri_data["total_triangles"], tri_data["n_original"],
            penalty=penalty
        )

        # ── Exhaust -> diagonal structure -> nearest recall ───────────────────
        exhaust_recall: Optional[Dict] = None
        exhaust_sig = bipolar_lattice.get_exhaust_signature()
        ring_phase  = bipolar_lattice._ring_net_phase()

        if exhaust_sig.sum() > 1e-10:
            # Generate diagonal structure for this sentence and store in session
            current_structure = diagonal_structure_generator.generate(
                exhaust_signature = exhaust_sig,
                ring_net_phase    = ring_phase,
                core_id           = bipolar_lattice.core_id,
                prompt            = sentence,
            )

            # Find nearest in session diagonal history
            session_matches = diagonal_structure_generator.nearest(
                current_structure, top_n=1
            )

            # Find nearest in cross-session exhaust memory (from disk)
            cross_matches = bipolar_lattice.nearest_exhaust(top_n=1)

            # Pick the closer of the two
            best_match = None
            if session_matches and cross_matches:
                # session_matches uses similarity [0,1] higher=better
                # cross_matches uses distance [0,∞] lower=better
                # Normalise: convert session similarity to distance = 1 - sim
                session_dist = 1.0 - session_matches[0]["similarity"]
                cross_dist   = cross_matches[0]["distance"]
                if session_dist <= cross_dist:
                    best_match = {
                        "prompt":   session_matches[0]["prompt"],
                        "distance": session_dist,
                        "source":   "session_diagonal",
                    }
                else:
                    best_match = {
                        "prompt":   cross_matches[0]["prompt"],
                        "distance": cross_dist,
                        "source":   "cross_session_exhaust",
                    }
            elif session_matches:
                sim = session_matches[0]["similarity"]
                best_match = {
                    "prompt":   session_matches[0]["prompt"],
                    "distance": 1.0 - sim,
                    "source":   "session_diagonal",
                }
            elif cross_matches:
                best_match = {
                    "prompt":   cross_matches[0]["prompt"],
                    "distance": cross_matches[0]["distance"],
                    "source":   "cross_session_exhaust",
                }

            exhaust_recall = best_match   # may be None if no history yet

        stream_ctx = symbol_grouping.stream_context(symbol_stream)
        words      = sentence.strip().split()
        word_fps   = [self._fingerprint_word(w) for w in words]

        vocab_hits  = []
        newly_named = []
        for wfp in word_fps:
            familiarity, is_stable, distinct_contexts = self.vocabulary.update(wfp, context_key)
            stored   = self.vocabulary.lookup(wfp.word)
            centroid = 0.0
            if stored:
                grp      = symbol_grouping.group_for(
                    wfp.symbol_stream[0] if wfp.symbol_stream else "A"
                )
                centroid = grp.tension_centroid if grp else 0.0
                
            if (is_stable or familiarity >= _FAMILIARITY_THRESHOLD) and distinct_contexts >= 2:
                named = invariant_engine.try_name_word(
                    word=wfp.word.rstrip('?!.,;:"\'').lstrip('"\'('),
                    symbol_stream=wfp.symbol_stream,
                    appearances=stored.appearances if stored else 1,
                    familiarity=familiarity,
                    centroid=centroid,
                    nonsense_flag=nonsense_flag,
                    penalty=penalty,
                    distinct_contexts=distinct_contexts,
                )
                
                if named:
                    newly_named.append(wfp.word)
                    
            if familiarity >= _FAMILIARITY_THRESHOLD:
                vocab_hits.append({
                    "word":        wfp.word,
                    "familiarity": familiarity,
                    "stable":      is_stable,
                    "named":       invariant_engine.is_named(wfp.word),
                    "appearances": stored.appearances if stored else 1,
                })

        invariant_engine.apply_decay(symbol_grouping.groups)

        # Pass explicit zero_breaks AND pockets so pocket split finds sentence boundary
        stream_ctx["_zero_breaks_raw"] = tri_data.get("zero_breaks", [])
        stream_ctx["_pockets_raw"]     = tri_data.get("pockets", [])
        fingerprint = self._fingerprint_sentence(
            sentence, symbol_stream, stream_ctx, word_fps
        )
        fingerprint["named_hits"]      = [h["word"] for h in vocab_hits if h.get("named")]
        fingerprint["exhaust_distance"] = exhaust_recall["distance"] if exhaust_recall else None
        fingerprint["session_epoch"]    = self._session_epoch
        fingerprint["newly_named"]     = newly_named
        fingerprint["prior_carry"]     = round(prior_carry, 4)
        fingerprint["carry_direction"] = carry_direction

        alignment = relational_tension.measure_alignment(fingerprint)
        fingerprint["carry_alignment"] = alignment

        prop_result["stream_mean_tension"] = stream_ctx["mean_tension"]
        prop_result["fold_coherence"]      = fold_line_resonance.get_coherence_signal()
        prop_result["field_direction"]     = fingerprint["direction"]
        prop_result["field_stress"]        = fingerprint["field_stress"]
        prop_result["vocab_hits"]          = len(vocab_hits)
        prop_result["vocab_stable"]        = self.vocabulary.stable_count()

        obs         = MultiObserver(num_observers=3)
        vib         = VibrationPropagator()
        linked_numeric = [x for x in prop_result.get("waveform_sample", [0.1])
                          if isinstance(x, (int, float))]
        linked_vib  = vib.holographic_linkage(np.array(linked_numeric) * 10)
        consensus, _ = obs.interact(
            linked_vib, prompt=sentence, iterations=10, prop_result=prop_result
        )

        base_answer = generator.generate(
            prompt=sentence,
            tri_data=tri_data,
            prop_result=prop_result,
            consensus=consensus,
            exhaust_recall=exhaust_recall,
        )

        answer = invariant_engine.generate_response(
            fingerprint=fingerprint,
            base_answer=base_answer,
            consensus=consensus,
            persistence=prop_result.get("persistence", 0.0),
            vocab_hits=vocab_hits,
        )

        geo_result = geometric_output.generate(
            fingerprint=fingerprint,
            vocabulary=self.vocabulary,
            invariant_engine=invariant_engine,
            consensus=consensus,
            persistence=prop_result.get("persistence", 0.0),
        )

        # ── Iterative geometric decode ────────────────────────────────────────
        # If the answer is still a geometry report, the field found structure
        # but couldn't decode it to content on the first pass.
        # Feed the geometry report back through as a second input — the
        # report itself contains structured signal (load bearers, tensions,
        # peak pairs, spin state). Running it through SymbolicWave finds
        # new symbol positions and tensions for those terms, potentially
        # resolving content the first pass missed.
        #
        # This implements the insight that "noise is structure in an
        # unrecognized coordinate system" — the geometry report IS the
        # field's own structure, just expressed in a different basis.
        # One iteration only — prevents infinite loops.

        _GEOMETRY_MARKERS = [
            "reconstruction pass", "field at phase boundary",
            "field resolved", "field processing", "field forming",
            "geometry stable", "field partially resolved",
            "field geometry matches",
        ]
        answer_is_geometry = any(
            m in answer.lower() for m in _GEOMETRY_MARKERS
        )

        iter_result: Optional[Dict] = None
        if answer_is_geometry and len(answer) > 20:
            # Build a second-pass input from the geometry report +
            # the highest net_signed words from the original fingerprint.
            # This is the coordinate-change: we re-express the geometry
            # report as a new symbol stream and let the field find
            # structure in its own output.
            per_word_sorted = sorted(
                [w for w in fingerprint.get("per_word", [])
                 if abs(w.get("net_signed", 0.0)) > 0.8
                 and w.get("pocket", 0) == 0],   # context side only
                key=lambda w: abs(w.get("net_signed", 0.0)),
                reverse=True
            )[:5]
            load_bearer_words = " ".join(
                w["word"].rstrip(".!?,;:") for w in per_word_sorted
            )
            if load_bearer_words:
                # Append the question from the original sentence to preserve
                # the context/query structure for pocket splitting
                original_question = ""
                parts = sentence.split("?")
                if len(parts) >= 2:
                    # Extract just the question part
                    q_start = sentence.rfind(".", 0, sentence.index("?"))
                    if q_start != -1:
                        original_question = sentence[q_start+1:].strip()

                # Strip metadata artifacts before re-encoding.
                # Geo output strings contain '[geometric', 'candidates:',
                # polarity values etc. that have high net_signed but no
                # semantic content. Filter them so the iteration decodes
                # actual content words, not format strings.
                _META_PATTERNS = re.compile(
                    r'\[\w|candidates:|candidates|'
                    r'\+[0-9]|[0-9]+\.[0-9]+|'
                    r'\[parity|alignment|resolution|'
                    r'polarity|approximation|confirmed|'
                    r'\bfield\b|\bpositive\b|\bnegative\b|\bboundary\b',
                    re.IGNORECASE
                )
                clean_bearers = " ".join(
                    w for w in load_bearer_words.split()
                    if not _META_PATTERNS.search(w)
                    and not w.startswith('[')
                    and not w.startswith('+')
                    and not w.startswith('|')
                    and not re.match(r'^[\[\]|+\-0-9.,]+$', w)
                    and len(w) > 2
                )
                if not clean_bearers:
                    clean_bearers = load_bearer_words  # fallback if all filtered

                if original_question:
                    iter_input = f"{clean_bearers}. {original_question}"
                else:
                    iter_input = clean_bearers

                # Run through the full language processor pipeline
                # but mark it as an iteration so we don't recurse
                iter_result = self._process_iteration(
                    iter_input,
                    original_sentence=sentence,
                    session_epoch=self._session_epoch,
                )

        # Etch exhaust and truth library
        bipolar_lattice.etch_exhaust(
            prompt=sentence,
            symbol_stream=symbol_stream,
        )

        persistence = prop_result.get("persistence", 0.0)
        if persistence >= _ETCH_PERSISTENCE_THRESHOLD and geo_result.get("parity_locked", False):
            waveform_raw = [x for x in prop_result.get("waveform_full", [])
                            if isinstance(x, (int, float))]
            if waveform_raw:
                ouroboros_engine.etch_to_library(
                    np.array(waveform_raw),
                    f"session::{sentence[:40].strip()}"
                )

        carry_injected = relational_tension.after_sentence(
            fingerprint=fingerprint,
            vocab_hits=vocab_hits,
            invariant_engine=invariant_engine,
        )

        fold_line_resonance.update_field_state(
            persistence=persistence,
            alignment=fingerprint.get("carry_alignment", 0.0),
            named_count=len(invariant_engine.named_invariants),
            carry=relational_tension.get_current_carry(),
        )

        elapsed = time.time() - start_time

        # If iterative decode produced real content, surface it
        if iter_result and not any(
            m in iter_result.get("answer", "").lower()
            for m in _GEOMETRY_MARKERS
        ):
            # Merge: keep original fingerprint, replace answer with iter answer
            answer = f"{iter_result['answer']} [decoded via geometric iteration]"
            geo_result = iter_result.get("geo_output", geo_result)

        return {
            "sentence":        sentence,
            "fingerprint":     fingerprint,
            "vocab_hits":      vocab_hits,
            "vocab_size":      self.vocabulary.size(),
            "vocab_stable":    self.vocabulary.stable_count(),
            "named_count":     len(invariant_engine.named_invariants),
            "newly_named":     newly_named,
            "was_primed":      _was_primed,
            "context_words":   _context_words,
            "answer":          answer,
            "geo_output":      geo_result,
            "consensus":       round(consensus, 4),
            "persistence":     round(persistence, 4),
            "gen_mode":        invariant_engine.get_generation_mode()["mode"],
            "carry_injected":  round(carry_injected, 4),
            "carry_alignment": alignment,
            "net_carry":       round(relational_tension.get_current_carry(), 4),
            "exhaust_recall":  exhaust_recall,
            "elapsed":         round(elapsed, 3),
        }

    def _process_iteration(
        self,
        iter_input:        str,
        original_sentence: str,
        session_epoch:     int,
    ) -> Dict[str, Any]:
        """
        Single iteration pass — runs the geometry report back through
        the pipeline to find structure in the field's own output.

        Uses a lighter pipeline than process():
          - Same SymbolicWave encoding
          - Direct propagation only (no generative to avoid library feedback loop)
          - Same fingerprinting and geo_output
          - Does NOT etch, does NOT update vocab, does NOT inject carry
          - One pass only, no recursion

        The iter_input is the load-bearer words from the original sentence
        reassembled with the original question — a new coordinate expression
        of the same underlying structure.
        """
        try:
            tri_iter   = self.sw.triangulate(iter_input)
            tri_iter["prompt"] = iter_input
            sym_iter   = tri_iter.get("symbol_stream", [])

            prop_iter  = WavePropagator()
            pr_iter    = prop_iter.propagate(tri_iter, steps=60)

            num_iter   = [x for x in pr_iter.get("waveform_sample", [0.1])
                          if isinstance(x, (int, float))]
            wamp_iter  = float(np.mean(np.abs(num_iter))) if num_iter else 0.1

            ctx_iter   = symbol_grouping.stream_context(sym_iter)
            words_iter = iter_input.strip().split()
            wfps_iter  = [self._fingerprint_word(w) for w in words_iter]

            # Tag with current session epoch
            for wfp in wfps_iter:
                wfp.session_epoch = session_epoch

            fp_iter = self._fingerprint_sentence(
                iter_input, sym_iter, ctx_iter, wfps_iter
            )
            fp_iter["session_epoch"]    = session_epoch
            fp_iter["exhaust_distance"] = None  # no exhaust recall for iteration

            obs_iter  = MultiObserver(num_observers=3)
            vib_iter  = VibrationPropagator()
            lv_iter   = vib_iter.holographic_linkage(np.array(num_iter) * 10)
            con_iter, _ = obs_iter.interact(
                lv_iter, prompt=iter_input, iterations=10, prop_result=pr_iter
            )

            geo_iter = geometric_output.generate(
                fingerprint=fp_iter,
                vocabulary=self.vocabulary,
                invariant_engine=invariant_engine,
                consensus=con_iter,
                persistence=pr_iter.get("persistence", 0.0),
            )

            base_iter = generator.generate(
                prompt=iter_input,
                tri_data=tri_iter,
                prop_result=pr_iter,
                consensus=con_iter,
            )

            ans_iter = invariant_engine.generate_response(
                fingerprint=fp_iter,
                base_answer=base_iter,
                consensus=con_iter,
                persistence=pr_iter.get("persistence", 0.0),
                vocab_hits=[],
            )

            return {
                "sentence":    iter_input,
                "fingerprint": fp_iter,
                "answer":      ans_iter,
                "geo_output":  geo_iter,
                "consensus":   round(con_iter, 4),
                "persistence": round(pr_iter.get("persistence", 0.0), 4),
            }
        except Exception as e:
            return {"answer": "", "geo_output": {}, "sentence": iter_input}

    def get_vocabulary(self) -> List[Dict[str, Any]]:
        return self.vocabulary.get_stable_words()

    def get_status(self) -> Dict[str, Any]:
        inv_status = invariant_engine.get_status()
        rt_status  = relational_tension.get_status()
        return {
            "process_count":    self._process_count,
            "vocab_size":       self.vocabulary.size(),
            "vocab_stable":     self.vocabulary.stable_count(),
            "named_invariants": inv_status["named_invariants"],
            "named_words":      inv_status["named_words"],
            "generation_mode":  inv_status["generation_mode"],
            "spin_description": inv_status["spin_description"],
            "coherence":        fold_line_resonance.get_coherence_signal(),
            "net_carry":        rt_status["net_carry"],
            "carry_direction":  rt_status["carry_direction"],
            "active_carries":   rt_status["active_carries"],
            "diagonal_structures": diagonal_structure_generator.get_status(),
        }


language_processor = LanguageProcessor()
