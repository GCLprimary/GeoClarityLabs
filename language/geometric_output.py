"""
language/geometric_output.py  (v17 — multi-mode three-layer with combiner fully outside base frame delta)
==================================================================================
Contextual combiner now operates completely outside the base frame delta / stream_pos / gap timing.
It still uses the exact same mechanics (Method 1 geometric scoring, multi-mode library pool, polarity/carry/resolution).
Placement is now purely driven by the on-the-fly field snapshot (carry_sign + polarity) — timeless relative to the base system.
Everything else (Layer 1 heavy, Layer 2 connectors, multi-mode library access, articulation, parity) is unchanged.
"""

import math
import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set

from core.invariants import invariants
from utils.symbol_grouping import symbol_to_signed, symbol_grouping
from utils.bipolar_lattice import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from wave.symbolic_wave import SymbolicWave

_PARITY_THRESHOLD           = 0.381966  # 1/phi² — parity lock threshold
# Polarization constants — confirmed across 3 sessions, 14 prompts
_P_MAX   = invariants.P_max   # 3/φ² — dielectric saturation ceiling
_P0_COLD = invariants.P0_cold  # √φ/φ² — geometric cold floor
# _PARITY_THRESHOLD / _P_MAX = 1/3 exactly
# Quantization level midpoint thresholds
_P_LEVEL_0_MAX = 0.650   # below = L0 cold
_P_LEVEL_1_MAX = 0.950   # below = L1 activating
_P_LEVEL_2_MAX = 1.110   # below = L2 warm, above = L3 saturated
_CONTENT_THRESHOLD          = 1.5
_ANSWER_CONTENT_THRESHOLD   = round(49 * 0.016395102, 6)             # 49×AD ≈ 0.8034
_BOUNDARY_CONTENT_THRESHOLD = round(49 * 0.016395102, 6)               # 49×AD ≈ 0.8034
_NET_TENSION_SCALE          = 8.0  # = _N_STRUCTURAL backbone waypoints in bipolar_lattice

_STRUCTURAL_ANCHORS = {
    # Core structural (originally present)
    "the", "and", "is", "in", "of", "a", "to", "it", "as",
    "that", "this", "was", "be", "are", "for", "on", "or",
    "from", "by", "at", "an", "not", "but", "so", "if",
    "its", "has", "had", "have", "with", "how", "what",
    # Temporal/connective — observed in outputs, never content
    "when", "then", "than", "been", "will", "just", "once",
    "while", "where", "which", "still", "also", "even",
    # Pronouns that slip through _PRONOUN_FILTER into score gate
    "their", "they", "them", "its", "our", "your", "his", "her",
    # Auxiliary verbs
    "did", "does", "do", "can", "may", "will", "could", "would",
    "should", "shall", "might", "must", "let",
    # Prepositions observed in outputs
    "into", "onto", "upon", "per", "via",
    # Connectors and comparators
    "than", "rather", "each", "every", "both", "such",
    "while", "since", "once", "even", "just", "also",
    "only", "more", "most", "very", "well", "still",
    "yet", "too", "then", "thus", "hence", "though",
    # Pronouns — never a useful output word
    "they", "them", "their", "those", "these", "there",
    "here", "who", "whom", "whose", "where", "which",
    "he", "she", "we", "you", "him", "her", "our", "your",
    # Quantifiers and determiners
    "major", "minor", "previous", "further", "other",
    "around", "about", "almost", "nearly", "beyond",
    "against", "between", "among", "across", "along",
    "whether", "either", "neither", "another", "instead",
    # Auxiliaries that slip through
    "might", "shall", "cannot", "could", "would", "should",
}

_PRONOUN_FILTER = {
    "their", "its", "they", "them", "themselves", "itself",
    "each", "other", "himself", "herself", "ourselves", "yourself",
    "those", "these", "whom", "whose",
}

_AUX_FILTER = {
    "can", "will", "would", "could", "should", "may", "might",
    "must", "shall", "been", "being", "had", "was", "were", "did",
}

_CONTENT_NS_MIN    = 1.3
_ACTION_NS_MIN     = 0.3
_ACTION_NS_MAX     = 2.5
_ACTION_T_MIN      = 0.05
_ACTION_SCORE_MIN  = 0.06

_PREP_FILTER = {
    "under", "over", "across", "through", "between", "among",
    "within", "beyond", "along", "toward", "against", "around",
    "above", "below", "behind", "beside", "before", "after",
    "into", "onto", "upon", "inside", "outside",
}

_MULT_ANSWER_CANDIDATE     = round(((1+math.sqrt(5))/2)**2, 4)       # φ² ≈ 2.6180
_MULT_CONTEXT_ONLY         = 1.5
_MULT_ANCHOR               = 1.0
_MULT_QUESTION_ONLY        = round(1.0 / ((1+math.sqrt(5))/2)**2, 6)  # 1/φ² = parity_threshold ≈ 0.381966
_MULT_UNKNOWN              = round(49 * 0.016395102, 6)              # 49×AD ≈ 0.8034
_MULT_SAME_SESS_OTHER      = round(1.0/((1+math.sqrt(5))/2)**2, 6)  # 1/φ² ≈ 0.381966
_MULT_CROSS_SESS_CLOSE     = 0.75
_MULT_CROSS_SESS_MID       = 0.50
_MULT_CROSS_SESS_FAR       = 0.30
_EXHAUST_CLOSE_THRESHOLD   = 0.002
_CROSS_SESS_THRESHOLD      = 3.0
_EXHAUST_MID_THRESHOLD     = 0.02


class GeometricOutput:
    def __init__(self):
        self._sw = SymbolicWave()

    # Generic fallback verbs that bleed across unrelated prompts.
    # 'involves' appears in stable vocab with action-range scores
    # and fires as best verb even when a real fingerprint verb exists.
    _GENERIC_VERB_BLOCKLIST = {
        "involves", "requires", "produces", "contains",
        "consists", "represents", "indicates",
        # Generic high-frequency verbs that weaken output
        "uses", "used", "using", "use",
        "gets", "got", "getting",
        "makes", "made", "making",
        "goes", "went", "going",
        "comes", "came", "coming",
        "gives", "gave", "giving",
        "takes", "took", "taking",
        "puts", "put", "putting",
        "sets", "set", "setting",
        "lets", "let", "letting",
        "keeps", "kept", "keeping",
        "shows", "showed", "showing",
        "seems", "seemed", "seeming",
        "becomes", "became", "becoming",
    }

    _Q_SKIP_VERBS = {
        'how','what','why','where','which','when','who',
        'do','does','did','is','are','was','were','will','would',
        'can','could','should','may','might','must','have','has','had',
        'be','been','being','a','an','the','ever','truly','usually',
        'often','always','never','only','just','some','any','this','that',
        # Prepositions — never question verbs
        'in','on','at','of','up','by','or','as','to','from','for',
        'with','into','onto','upon','within','without','through',
        # Copular/labeling verbs
        'considered','called','known','named','regarded','deemed','seen',
        'described','defined','classified','recognized','identified',
    }

    def _read_field(self, fingerprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if fingerprint is not None:
            net_t    = fingerprint.get("net_tension", 0.0)
            polarity = float(np.tanh(net_t / _NET_TENSION_SCALE))
        else:
            pos_tension = sum(s.tension for s in bipolar_lattice.strings if s.active and s.polarity > 0)
            neg_tension = sum(abs(s.tension) for s in bipolar_lattice.strings if s.active and s.polarity < 0)
            n_active    = max(1, sum(1 for s in bipolar_lattice.strings if s.active))
            differential = (pos_tension - neg_tension) / (n_active * 0.5)
            polarity     = float(np.clip(differential, -1.0, 1.0))

        resolution  = fold_line_resonance.get_resolution_score()
        field_state = fold_line_resonance._field_persistence
        return {
            "polarity":    polarity,
            "resolution":  resolution,
            "persistence": field_state,
            "carry":       fold_line_resonance._field_carry,
            "carry_sign":  int(math.copysign(1, fold_line_resonance._field_carry)) if fold_line_resonance._field_carry != 0.0 else 0,
        }

    def _identify_target_region(self, field: Dict[str, Any]) -> Dict[str, Any]:
        polarity   = field["polarity"]
        resolution = field["resolution"]
        window     = int(np.clip(8 * (1.0 - resolution) + 2, 4, 8))
        if polarity > 0.1:
            centre = int(np.clip(round(polarity * 9), 1, 13))
            low, high = max(1, centre - window), min(13, centre + window)
            side = "positive"
        elif polarity < -0.1:
            centre = int(np.clip(round(abs(polarity) * 9), 1, 13))
            low, high = -min(13, centre + window), -max(1, centre - window)
            side = "negative"
        else:
            low, high = -3, 3
            side = "boundary"
        return {"side": side, "low": low, "high": high, "centre": polarity * 9, "window": window, "polarity": polarity}

    def _sample_vocabulary(self, target: Dict[str, Any], vocabulary: Any, invariant_engine: Any,
                           fingerprint: Dict[str, Any], n_candidates: int = 8,
                           target_side: str = "boundary",
                           pressure_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # ── Pressure-modulated multiplier ─────────────────────────────────────
        # Derived from ferroelectric model: pmult scales with pressure deficit.
        # FOCUS: field needs gradient → boost high-charge candidates
        # SATURATE: field at ceiling  → normalize to P_max
        # SUSTAIN: field coasting     → no modification
        _ps        = pressure_state or {}
        _ps_mode   = _ps.get("mode", "FOCUS")
        _ps_delta  = _ps.get("pressure_delta", 0.0)
        _ps_G_sat  = _ps.get("G_sat", 3.0)
        _ps_P0     = _ps.get("P0_current", 0.7)
        _ps_P_MAX  = _ps.get("P_MAX", 1.1459)

        # Mobius face modulates mode: INNER→FOCUS (local/causal), OUTER→SATURATE
        _mobius_face = _ps.get("mobius_face", "unknown")
        if _mobius_face == "INNER" and _ps_mode == "SUSTAIN":
            _ps_mode = "FOCUS"    # INNER face tightens to local resolution
        elif _mobius_face == "OUTER" and _ps_delta > 0.5:
            _ps_mode = "SATURATE" # OUTER with excess gradient → push to saturation

        if _ps_mode == "FOCUS" and _ps_delta < 0:
            _pmult_factor = 1.0 + abs(_ps_delta) / max(_ps_G_sat * 2, 1.0)
        elif _ps_mode == "SATURATE":
            _pmult_factor = _ps_P_MAX / max(_ps_P0, invariants.P0_cold)
        else:
            _pmult_factor = 1.0
        _pmult_factor = max(0.5, min(_pmult_factor, 3.0))

        # Diagonal recall pool — words from geometrically similar prior prompt
        # Only activate when similarity is meaningful (>= 0.65)
        _recall_sim   = _ps.get("recall_similarity", 0.0)
        _recall_words = set(
            w.lower().strip() for w in _ps.get("recall_candidates", [])
            if w and len(w) > 2
        ) if _recall_sim >= 0.65 else set()
        _recall_boost = _recall_sim  # boost weight = similarity score [0.65, 1.0]
        low, high = target["low"], target["high"]
        per_word_list = fingerprint.get("per_word", [])
        context_word_set: Set[str] = set()
        question_word_set: Set[str] = set()
        for w in per_word_list:
            clean = w.get("word", "").rstrip(".!?,;:").lower()
            if not clean: continue
            if w.get("pocket", 0) == 0:
                context_word_set.add(clean)
            else:
                question_word_set.add(clean)

        has_pocket_data = bool(context_word_set or question_word_set)
        current_words: Set[str] = context_word_set | question_word_set
        current_epoch = fingerprint.get("session_epoch", 0)
        exhaust_dist  = fingerprint.get("exhaust_distance")

        def _cross_session_mult(dist: Optional[float]) -> Tuple[float, str]:
            if dist is None: return _MULT_CROSS_SESS_MID, "cross_sess_unknown"
            if dist < _EXHAUST_CLOSE_THRESHOLD: return _MULT_CROSS_SESS_CLOSE, "cross_sess_close"
            if dist < _EXHAUST_MID_THRESHOLD: return _MULT_CROSS_SESS_MID, "cross_sess_mid"
            return _MULT_CROSS_SESS_FAR, "cross_sess_far"

        def pocket_multiplier(word: str, from_current: bool, word_epoch: int = 0) -> Tuple[float, str]:
            if not from_current:
                if word_epoch == current_epoch: return _MULT_SAME_SESS_OTHER, "same_sess_other"
                return _cross_session_mult(exhaust_dist)
            if not has_pocket_data: return 1.0, "no_pocket"
            w = word.lower()
            in_ctx = w in context_word_set
            in_q   = w in question_word_set
            if in_ctx and in_q: return _MULT_ANCHOR, "anchor"
            if in_ctx and not in_q: return _MULT_ANSWER_CANDIDATE, "answer_cand"
            if in_q and not in_ctx: return _MULT_QUESTION_ONLY, "q_only"
            return _MULT_UNKNOWN, "unknown"

        def effective_threshold(pmult: float, side: str) -> float:
            if side == "boundary": return _BOUNDARY_CONTENT_THRESHOLD
            if pmult == _MULT_ANSWER_CANDIDATE: return _ANSWER_CONTENT_THRESHOLD
            if pmult == _MULT_SAME_SESS_OTHER: return _CROSS_SESS_THRESHOLD
            return _CONTENT_THRESHOLD

        candidates = []

        # Prompt-based candidates (Layer 1 core)
        _n_per_word = max(len(per_word_list) - 1, 1)
        for _pw_idx, w in enumerate(per_word_list):
            ns   = w.get("net_signed", 0.0)
            word = w.get("word", "").rstrip(".!?,;:")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            pmult, plabel = pocket_multiplier(word, from_current=True, word_epoch=current_epoch)
            thresh = effective_threshold(pmult, target_side)
            in_range = (ns <= high) if plabel == "answer_cand" else (low <= ns <= high)
            if in_range and abs(ns) >= thresh:
                score = (abs(ns) / 13.0) * pmult * _pmult_factor
                if _recall_words and word.lower() in _recall_words:
                    score *= (1.0 + _recall_boost)
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "load_bearer", "priority": 3,
                    "named": invariant_engine.is_named(word),
                    "pocket_mult": pmult, "pocket_label": plabel,
                    "score": score,
                    "mean_tension": w.get("mean_tension", 0.0),
                    "stream_pos":   _pw_idx / _n_per_word,
                })

        # Library-augmented mode: pull matching named invariants
        for word_key, data in invariant_engine.named_invariants.items():
            word = data.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            stream  = [self._sw._token_to_27_symbol(c) for c in word if c and not c.isspace()]
            zero_ch = chr(48)
            ns      = sum(symbol_to_signed(s) / 13.0 for s in stream if s != zero_ch)
            from_current  = word.lower() in current_words
            ni_epoch      = current_epoch if from_current else 0
            pmult, plabel = pocket_multiplier(word, from_current, word_epoch=ni_epoch)
            thresh = effective_threshold(pmult, target_side)
            # Skip library words whose polarity conflicts with field direction
            _fnet = fingerprint.get('net_tension', 0.0) if fingerprint else 0.0
            if _fnet > 0.5 and ns < -0.5: continue
            if _fnet < -0.5 and ns > 0.5: continue

            if low <= ns <= high and abs(ns) >= thresh:
                score = (abs(ns) / 13.0) * pmult * _pmult_factor
                if _recall_words and word.lower() in _recall_words:
                    score *= (1.0 + _recall_boost)
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "named_invariant", "priority": 4,
                    "named": True, "pocket_mult": pmult,
                    "pocket_label": plabel, "score": score,
                })

        # Stable vocab
        stable = vocabulary.get_stable_words() if hasattr(vocabulary, "get_stable_words") else []
        for entry in stable:
            word = entry.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            ns   = entry.get("net_signed", 0.0)
            from_current  = word.lower() in current_words
            sv_epoch      = entry.get("session_epoch", 0)
            is_named      = invariant_engine.is_named(word)
            pmult, plabel = pocket_multiplier(word, from_current, word_epoch=sv_epoch)
            thresh = effective_threshold(pmult, target_side)
            # Skip library words whose polarity conflicts with field direction
            _fnet = fingerprint.get('net_tension', 0.0) if fingerprint else 0.0
            if _fnet > 0.5 and ns < -0.5: continue
            if _fnet < -0.5 and ns > 0.5: continue

            if low <= ns <= high and abs(ns) >= thresh:
                score = (abs(ns) / 13.0) * pmult * _pmult_factor
                # Recency decay: words not in the current prompt decay.
                # Named invariants: gentle decay (×0.7) — they earned their status
                # but must still fade when context shifts. Prevents 'fire' bleed.
                # Unnamed stable: stronger decay (×0.5) — same as before.
                # Both: only applies when word is absent from current fingerprint.
                if not from_current:
                    score *= 0.5  # absent from context → decay regardless of named status
                    # Named invariants ARE the field's memory of significant words,
                    # but they must still fade when context shifts entirely.
                    # 0.7 was too gentle — 'fire','ice' kept bleeding across prompts.
                    # Named words in the CURRENT fingerprint get full weight already
                    # (from_current=True for those), so this only affects absent ones.
                if _recall_words and word.lower() in _recall_words:
                    score *= (1.0 + _recall_boost)
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "stable_vocab", "priority": 1,
                    "named": is_named,
                    "pocket_mult": pmult, "pocket_label": plabel,
                    "score": score,
                })

        seen = {}
        for c in candidates:
            w = c["word"].lower()
            if w not in seen or c["priority"] > seen[w]["priority"]:
                seen[w] = c

        current_only = [c for c in seen.values() if c.get("pocket_label", "") not in ("cross_sess_close", "cross_sess_mid", "cross_sess_far", "cross_sess_unknown", "same_sess_other")]
        pool = current_only if current_only else list(seen.values())
        return sorted(pool, key=lambda c: (c["score"] * (1 + c["priority"] * 0.1)), reverse=True)[:n_candidates]

    def _select_two_pools(self, per_word_list: list, anchor_word: Optional[str],
                          n_content: int = 3, n_action: int = 1) -> List[Dict]:
        content, action = [], []
        anchor_entry    = None
        anchor_clean    = anchor_word.lower().rstrip(".!?,;:") if anchor_word else None
        _anchor_parts   = anchor_clean.split() if anchor_clean else []
        _anchor_first   = _anchor_parts[0] if _anchor_parts else None
        _anchor_second  = _anchor_parts[1] if len(_anchor_parts) > 1 else None

        _n = max(len(per_word_list) - 1, 1)

        for _idx, w in enumerate(per_word_list):
            word = w.get("word", "").rstrip(".!?,;:")
            wl   = word.lower()
            ns   = w.get("net_signed", 0.0)
            t    = w.get("mean_tension", 0.0)
            pos  = _idx / _n
            ans  = abs(ns)

            if wl in _PRONOUN_FILTER or wl in _STRUCTURAL_ANCHORS: continue
            if wl in _AUX_FILTER: continue
            if wl in {"when", "where", "while", "although", "because", "since",
                      "than", "like", "nor", "yet", "just", "only", "even"}: continue

            if _anchor_first and wl == _anchor_first:
                display = word
                if _anchor_second: display = word + " " + _anchor_second
                anchor_entry = {"word": display, "ns": ns, "t": t, "pos": pos, "pool": "anchor"}
                continue
            if _anchor_second and wl == _anchor_second: continue

            action_score = t * ans

            if ans >= _CONTENT_NS_MIN and wl not in _PREP_FILTER:
                content.append({"word": word, "ns": ns, "t": t, "pos": pos, "pool": "content"})
            elif (ans >= _ACTION_NS_MIN and ans < _ACTION_NS_MAX and
                  t >= _ACTION_T_MIN and action_score >= _ACTION_SCORE_MIN and
                  wl not in _PREP_FILTER):
                action.append({"word": word, "ns": ns, "t": t, "pos": pos, "pool": "action", "score": action_score})

        n_action = min(2, max(1, len(per_word_list) // 12))
        top_action  = sorted(action, key=lambda x: -x.get("score", 0))[:n_action]

        if top_action:
            verb_pos = top_action[0]["pos"]
            proximal = [w for w in content if w["pos"] <= 0.60 or abs(w["pos"] - verb_pos) <= 0.40]
            content_pool = proximal if proximal else content
        else:
            content_pool = content

        top_content = sorted(content_pool, key=lambda x: -abs(x["ns"]))[:n_content]

        merged = top_content + top_action
        if anchor_entry: merged.append(anchor_entry)

        return sorted(merged, key=lambda x: x["pos"])

    def _detect_question_type(self, fingerprint: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        if fingerprint is None: return "entity", "and"
        per_word = fingerprint.get("per_word", [])
        _Q_WORDS = {"how","why","what","where","which","when","who"}
        q_words = [w for w in per_word if w.get("word","").lower().rstrip("?") in _Q_WORDS and w.get("pocket", 0) == 1]
        if not q_words: return "entity", "and"
        qw = min(q_words, key=lambda w: w.get("net_signed", 0.0))
        ns = qw.get("net_signed", 0.0)
        if ns <= -0.8: return "causal", "because"
        elif ns <= -0.2: return "process", "through"
        elif ns <= 0.5: return "entity", "and"
        elif ns <= 1.8: return "spatial", "in"
        else: return "select", "and"

    def _pocket_multiplier(self, word: str, fingerprint: Optional[Dict[str, Any]], from_current: bool = False) -> Tuple[float, str]:
        if fingerprint is None:
            return 1.0, "fallback"
        return 1.0, "connector"

    def _effective_threshold_for_connector(self, pmult: float, field: Dict[str, Any]) -> float:
        return _CONTENT_THRESHOLD * (1.0 - field.get("resolution", 0.875))

    def _multi_pass_assembly(self, candidates: List[Dict[str, Any]], field: Dict[str, Any],
                             target: Dict[str, Any], vocabulary: Any,
                             fingerprint: Optional[Dict[str, Any]] = None,
                             invariant_engine: Any = None) -> Tuple[str, str]:
        if not candidates:
            return ("Field geometry unresolved — no vocabulary in target region.", "fallback")

        qtype, q_link = self._detect_question_type(fingerprint)

        # Named word decay: if the anchor/top candidate is a named invariant
        # but NOT in the current prompt, halve its effective score so it
        # doesn't dominate assembly for unrelated topics.
        _fp_words_set = set(
            w.get("word", "").lower().rstrip(".,!?;:")
            for w in (fingerprint.get("per_word", []) if fingerprint else [])
        )
        _decayed = []
        for c in candidates:
            w = c.get("word", "").lower()
            if c.get("named", False) and w not in _fp_words_set:
                c = dict(c)
                c["score"] = c.get("score", 0.0) * 0.5
            _decayed.append(c)
        candidates = _decayed

        # ── Verb proximity filter (restored from v6) ────────────────────────
        # Apply BEFORE heavy/light split so late-sentence context words
        # (location, time, manner) can't displace early answer words.
        # Rule: keep if stream pos <= 0.55 OR within 0.35 of action verb.
        # This is the fix for 'gorillas eat central africa' class of errors.
        _action_words = [c for c in candidates
                         if _ACTION_NS_MIN <= abs(c.get("net_signed", 0.0)) <= _ACTION_NS_MAX
                         and c.get("mean_tension", 0.0) > _ACTION_T_MIN]
        if _action_words:
            verb_pos = min(_action_words, key=lambda x: x.get("pos", 0.5)).get("pos", 0.5)
            proximal = [c for c in candidates
                        if c.get("pos", 0.5) <= 0.55
                        or abs(c.get("pos", 0.5) - verb_pos) <= 0.35]
            if proximal:
                candidates = proximal

        # Layer 1: Heavy candidates (multi-mode)
        heavy = [c for c in candidates if abs(c.get("net_signed", 0.0)) >= _CONTENT_NS_MIN]

        # Layer 2: Connectors — unified filtering (multi-mode)
        # Recency guard: only include stable words that appear in the current
        # fingerprint. Words from prior prompts (e.g. 'plants', 'arch') should
        # not bleed into assembly for unrelated topics.
        _current_fp_words = set(
            w.get("word", "").lower().rstrip(".,!?;:")
            for w in (fingerprint.get("per_word", []) if fingerprint else [])
        )
        light = []
        if hasattr(vocabulary, "get_stable_words"):
            for entry in vocabulary.get_stable_words():
                word = entry.get("word", "").lower()
                if word in _STRUCTURAL_ANCHORS: continue
                # Skip stable words not present in current prompt
                if word not in _current_fp_words: continue
                ns = entry.get("net_signed", 0.0)
                t = entry.get("mean_tension", 0.0)
                pmult, plabel = self._pocket_multiplier(word, fingerprint, from_current=True)
                thresh = self._effective_threshold_for_connector(pmult, field)
                if 0.1 <= abs(ns) < _CONTENT_NS_MIN and t > 0.08 and abs(ns) >= thresh:
                    light.append({"word": entry.get("word"), "ns": ns, "t": t, "pos": 0.5, "pool": "connector"})

        # Merge Layers 1 + 2
        ordered = heavy + light
        if ordered:
            ordered = sorted(ordered, key=lambda x: x.get("pos", 0.5))
            for i, item in enumerate(ordered):
                item["pos"] = i / max(len(ordered) - 1, 1)

        # Compute question verbs before combiner AND spine — both need them
        _question_verbs = set()
        if fingerprint:
            _QV_ENDINGS = ("s","es","ed","ing","ize","ise","ate","fy","en",
                           "it","mit","pt","nd","ld","nt")
            for _qw in fingerprint.get("per_word", []):
                if _qw.get("pocket", 0) == 1:
                    _qwl = _qw.get("word","").lower().rstrip(".,!?;:")
                    if (_qwl not in self._Q_SKIP_VERBS
                            and _qwl not in _STRUCTURAL_ANCHORS
                            and _qwl not in self._GENERIC_VERB_BLOCKLIST
                            and len(_qwl) > 3
                            and any(_qwl.endswith(e) for e in _QV_ENDINGS)):
                        _question_verbs.add(_qwl)

        # Layer 3: Contextual combiner — now fully outside base frame delta
        final_words = self._contextual_combiner(ordered, field, fingerprint, qtype, vocabulary,
                                                question_verbs=_question_verbs)

        # Layer 4: Axis-driven semantic role chain
        #
        # Role assignment derives directly from Dual-13 group geometry:
        #
        #   NS arm (odd groups ±1,±3,±5,±7,±9,±11,±13) = builders/inverters
        #     → SUBJECT (highest-scoring NS pkt=0 word)
        #     → PRIMARY OBJECTS (remaining NS pkt=0 words, score-ordered)
        #
        #   EW arm (even groups ±2,±4,±6,±8,±10,±12) = recognizers/compressors
        #     → VERB (highest-scoring EW pkt=0 word with ns >= 0.8)
        #     → SECONDARY OBJECTS (remaining EW pkt=0 words)
        #
        #   Negative groups (gid < 0) = structural/bridge words
        #     → CONNECTIVE (best connective from pkt=1 negative-group words)
        #
        # Skeleton: [NS-subject] [EW-verb] [connective] [NS-objects] [EW-objects]
        #
        # Resolution-gated chain length:
        #   res >= 0.875 → 6 words   res >= 0.750 → 5 words
        #   res >= 0.700 → 4 words   res <  0.700 → 3 words
        #
        # Verb fallback: if no EW verb found, skip verb slot entirely and
        # use all remaining slots for objects — richer output beats weak verb.

        _fp_per_word = fingerprint.get("per_word", []) if fingerprint else []

        # ── Resolution-gated chain length ──────────────────────────────────────
        # max_chain = 5 = number of Möbius twist points (T1-T5) in the system.
        # Each twist point corresponds to one semantic slot in the output.
        # min_chain = 2 = subject + one object minimum.
        # Scale from P0_cold (cold floor) to 0.875 (observed parity ceiling).
        # P0_cold = √φ/φ² = 0.4859 — the geometric cold-start floor.
        _res_now   = field.get("resolution", 0.15)
        _P0_COLD   = invariants.P0_cold          # √φ/φ² ≈ 0.4859
        _RES_CEIL  = 0.875                        # observed parity ceiling
        _max_chain = max(2, min(5, round(
            2 + (_res_now - _P0_COLD) / (_RES_CEIL - _P0_COLD) * 3
        )))

        # ── Named invariant set ────────────────────────────────────────────────
        _named_set = set()
        if hasattr(invariant_engine, "get_named_words"):
            _named_set = {w.lower() for w in invariant_engine.get_named_words()}

        # ── Four-arm Dual-13 role partition ────────────────────────────────────
        # Each arm maps directly to a semantic role:
        #
        #   N arm: gid > 0 AND odd  (+1,+3,+5...+13) builders   → SUBJECT + primary objects
        #   S arm: gid < 0 AND odd  (-1,-3,-5...-13) inverters  → VERB (inverted action)
        #   E arm: gid > 0 AND even (+2,+4,+6...+12) recognizers→ OBJECTS / relations
        #   W arm: gid < 0 AND even (-2,-4,-6...-12) compressors→ CONNECTIVES / modifiers
        #
        # This is the quad displacer geometry read directly as syntax.

        _fp_group_map = {
            w.get("word","").lower().rstrip(".,!?;:"): w.get("dominant_group", w.get("grp", 0))
            for w in _fp_per_word
        }
        _fp_ns_map = {
            w.get("word","").lower().rstrip(".,!?;:"): abs(w.get("net_signed", w.get("net", 0)))
            for w in _fp_per_word
        }

        _all_cands = [w for w in final_words if w.get("pool") not in ("action","verb")]
        _verb_from_combiner = next(
            (w for w in final_words if w.get("pool") in ("action","verb")), None
        )

        # Four pools
        _N_cands = []   # +odd  → subject / primary nouns
        _S_cands = []   # -odd  → verb candidates
        _E_cands = []   # +even → object / relational
        _W_cands = []   # -even → connective / modifier

        for c in _all_cands:
            wl  = c.get("word","").lower().rstrip(".,!?;:")
            gid = _fp_group_map.get(wl, 0)
            if   gid > 0 and gid % 2 == 1:   _N_cands.append((c, gid))
            elif gid < 0 and abs(gid) % 2 == 1: _S_cands.append((c, gid))
            elif gid > 0 and gid % 2 == 0:   _E_cands.append((c, gid))
            elif gid < 0 and abs(gid) % 2 == 0: _W_cands.append((c, gid))
            else:                               _N_cands.append((c, gid))  # boundary → treat as noun

        # Sort each pool by score descending
        for pool in [_N_cands, _S_cands, _E_cands, _W_cands]:
            pool.sort(key=lambda x: x[0].get("score", 0.0), reverse=True)

        # Flatten for legacy compat (ns_cands = N+S for subject fallback)
        _ns_cands    = _N_cands + _S_cands
        _ew_cands    = _E_cands + _W_cands
        _other_cands = []

        # ── SUBJECT: highest-scoring N-arm named invariant, else highest N-arm ─
        # N arm (positive odd) = builders = nouns/subjects
        # Named N-arm invariants are highest priority — field-confirmed domain nouns
        _subject = None
        _N_named = [
            (c, gid) for c, gid in _N_cands
            if c.get("word","").lower().rstrip(".,!?;:") in _named_set
        ]
        if _N_named:
            _subject = _N_named[0][0]
        elif _N_cands:
            _subject = _N_cands[0][0]
        elif _ns_cands:   # fallback: any NS if no pure N
            _subject = _ns_cands[0][0]
        elif _ew_cands:
            _subject = _ew_cands[0][0]

        if _subject is None:
            self._svo_capped = False
        else:
            _subj_wl = _subject.get("word","").lower().rstrip(".,!?;:")

            # ── VERB: S arm first (-odd = inverters), then E arm, then combiner ──
            # Verb ns threshold: 49 × AD ≈ 0.8034
            # Derived: 49 is the largest integer where 49×AD stays below parity_threshold×2
            # This filters low-charge words (nouns/modifiers) from the verb pool
            _VERB_NS_MIN = 49 * invariants.asymmetric_delta   # ≈ 0.8034
            # S arm words are inverters — they negate/act-upon the subject.
            # E arm words are recognizers — relational/processual.
            # If no clean verb found → skip slot for richer objects.
            _svo_verb = None
            _VERB_NOUN_ENDINGS_L4 = ("tion","sion","ness","ment","ity","ance","ence",
                                      "ogen","agen","gen","ism","ist","ogy","ium")
            _GENERIC_SKIP = {"produces","requires","involves","contains","consists",
                             "uses","makes","gets","goes","comes","gives","takes",
                             "puts","sets","lets","keeps","shows","seems","becomes"}

            # Try S arm first (negative odd = inverters = action words)
            for c, gid in _S_cands:
                wl        = c.get("word","").lower().rstrip(".,!?;:")
                actual_ns = _fp_ns_map.get(wl, 0)
                if (wl != _subj_wl
                        and actual_ns >= _VERB_NS_MIN
                        and not any(wl.endswith(e) for e in _VERB_NOUN_ENDINGS_L4)
                        and wl not in _GENERIC_SKIP
                        and wl not in _STRUCTURAL_ANCHORS):
                    _svo_verb = c
                    break

            # Try E arm if S arm had nothing (positive even = recognizers)
            if _svo_verb is None:
                for c, gid in _E_cands:
                    wl        = c.get("word","").lower().rstrip(".,!?;:")
                    actual_ns = _fp_ns_map.get(wl, 0)
                    if (wl != _subj_wl
                            and actual_ns >= _VERB_NS_MIN
                            and not any(wl.endswith(e) for e in _VERB_NOUN_ENDINGS_L4)
                            and wl not in _GENERIC_SKIP
                            and wl not in _STRUCTURAL_ANCHORS):
                        _svo_verb = c
                        break

            # Try combiner verb only if it's a real domain verb (not generic)
            if _svo_verb is None and _verb_from_combiner:
                _vw = _verb_from_combiner.get("word","").lower().rstrip(".,!?;:")
                if _vw not in _GENERIC_SKIP and _vw != _subj_wl:
                    _svo_verb = _verb_from_combiner

            # ── CONNECTIVE from pkt=1 negative-group words ────────────────────
            _conn_word = None
            if fingerprint:
                from language.output_translator import _find_connective
                _conn_word = _find_connective(_fp_per_word)

            # ── DEDUPLICATE helper ────────────────────────────────────────────
            def _stem_s(w):
                return w.lower().rstrip(".,!?;:s")

            _used = {_subj_wl, _stem_s(_subj_wl)}
            if _svo_verb:
                _vw = _svo_verb.get("word","").lower().rstrip(".,!?;:")
                _used.add(_vw)
                _used.add(_stem_s(_vw))
            if _conn_word:
                _used.add(_conn_word)

            # ── REMAINING OBJECTS: N arm first, then E arm, then S/W leftovers ──
            # N arm nouns lead (most concrete domain words)
            # E arm relations follow (processual/relational words)
            # S arm leftovers last (unused inverters as modifiers)
            _remaining = []
            for pool in [_N_cands, _E_cands, _S_cands, _W_cands]:
                for c, gid in pool:
                    wl = c.get("word","").lower().rstrip(".,!?;:")
                    if _stem_s(wl) not in _used and wl not in _used:
                        _remaining.append(c)
                        _used.add(wl)
                        _used.add(_stem_s(wl))

            # ── SLOT BUDGET ───────────────────────────────────────────────────
            _has_verb = _svo_verb is not None
            _has_conn = _conn_word is not None
            _fixed_slots = 1 + (1 if _has_verb else 0) + (1 if _has_conn else 0)
            _obj_slots   = max(1, _max_chain - _fixed_slots)

            # ── ASSEMBLE CHAIN ────────────────────────────────────────────────
            _chain = [_subject]
            if _svo_verb:
                _chain.append(_svo_verb)
            if _conn_word:
                _chain.append({"word": _conn_word, "pool": "connective", "pos": 0.5})
            _chain.extend(_remaining[:_obj_slots])

            final_words = _chain
            self._svo_capped = (len(_remaining) > _obj_slots)

        # Final articulation — detect anchor from pkt=1 subject
        # The first non-skip word in pkt=1 after question words is the subject
        _anchor = None
        if fingerprint:
            _pkt1 = [w for w in fingerprint.get("per_word", [])
                     if w.get("pocket", 0) == 1]
            _skip = {'how','why','what','where','when','which','who',
                     'do','does','did','is','are','was','were','will',
                     'would','can','could','should','may','might','be',
                     # Structural articles, prepositions, conjunctions
                     'a','an','the','in','of','to','at','by','as','on',
                     'or','if','it','its','and','but','not','so','for',
                     'from','with','that','this','than','then','into',
                     # Additional words that produce bad conjugations as anchors
                     'rather','each','every','these','those','both','such',
                     'while','since','once','even','just','also','only',
                     'more','most','very','well','still','yet','too'}
            for _w in _pkt1:
                _wl = _w.get("word","").lower().rstrip("?!.,;:")
                if _wl and _wl not in _skip:
                    _anchor = _wl
                    break

        text = self._articulate(
            word_dicts=final_words,
            per_word=fingerprint.get("per_word", []) if fingerprint else [],
            anchor_word=_anchor,
            qtype=qtype
        )

        if not text or text == ".":
            text = " ".join([w["word"] for w in final_words]) + "."

        text = self._post_touchup(text)
        mode = f"directed_{qtype}"
        return text, mode

    def _contextual_combiner(self, ordered: List[Dict], field: Dict[str, Any],
                             fingerprint: Optional[Dict[str, Any]], qtype: str,
                             vocabulary: Any,
                             question_verbs: Optional[set] = None) -> List[Dict]:
        """Method 1 Contextual combiner — fully outside base frame delta.
        Verb selection uses the exact same geometric scoring and multi-mode pool.
        Insertion position is now determined solely by field snapshot (carry_sign + polarity)
        with zero dependence on stream_pos, gaps, or base frame delta timing."""
        if not ordered:
            return ordered

        resolution = field.get("resolution", 0.875)
        polarity   = field.get("polarity", 0.0)
        carry      = field.get("carry", 0.0)
        carry_sign = field.get("carry_sign", 0)

        # Build verb pool (multi-mode: prompt + library)
        # Recency filter: stable vocab verbs only eligible if they appear
        # in the CURRENT fingerprint. 'involves' persists in stable vocab
        # from prior sessions but must not fire on unrelated prompts.
        _current_fp_word_set = set()
        if fingerprint:
            for _fw in fingerprint.get("per_word", []):
                _current_fp_word_set.add(_fw.get("word","").lower().rstrip(".,!?;:"))

        # Noun endings — words ending in these are never verbs regardless of other signals
        _VERB_NOUN_ENDINGS = ("tion","sion","ness","ment","ity","ance","ence",
                              "ogen","agen","gen","ism","ist","ogy","ium",
                              "ary","ery","ory","phy","thy","chy","nce","nse")

        verb_pool = []
        if fingerprint:
            for w in fingerprint.get("per_word", []):
                if w.get("pocket", 0) == 1:
                    wl = w.get("word", "").lower().rstrip("?!.,;:")
                    _vns = abs(w.get("net_signed", 0.0))
                    if (wl not in self._Q_SKIP_VERBS
                            and 0.8 <= _vns < 3.0
                            and not any(wl.endswith(e) for e in _VERB_NOUN_ENDINGS)):
                        verb_pool.append({"word": w.get("word"), "ns": w.get("net_signed", 0.0), "t": w.get("mean_tension", 0.0)})

        if hasattr(vocabulary, "get_stable_words"):
            for entry in vocabulary.get_stable_words():
                wl = entry.get("word", "").lower()
                ns = entry.get("net_signed", 0.0)
                t = entry.get("mean_tension", 0.0)
                # Only include stable vocab verbs present in current prompt
                if wl not in _current_fp_word_set:
                    continue
                if _ACTION_NS_MIN <= abs(ns) <= _ACTION_NS_MAX and t > _ACTION_T_MIN:
                    verb_pool.append({"word": entry.get("word"), "ns": ns, "t": t})

        # Score verbs geometrically (unchanged)
        scored_verbs = []
        for v in verb_pool:
            vw = v.get("word","").lower().rstrip(".,!?;:")
            if vw in self._GENERIC_VERB_BLOCKLIST:
                continue  # never use generic fallback verbs from stable vocab
            score = (v["t"] * abs(v["ns"])) * resolution * (1.0 + abs(polarity)) * (1.0 + abs(carry))
            # Question verb boost: 3× if it appeared explicitly in the question
            if question_verbs and vw in question_verbs:
                score *= 3.0
            scored_verbs.append((score, v["word"]))

        # Threshold grounded in geometry from coupling experiments:
        # floor = AD (asymmetric delta = phi/100 ≈ 0.01640)
        # slope = phi/10 ≈ 0.1618 (one order below phi)
        # Both derive from the same constant driving the field.
        # Previous: max(0.04, 0.12*res) — empirically tuned
        _PHI = 1.61803399  # (1+sqrt(5))/2
        _AD  = invariants.asymmetric_delta  # 2π/3 - 2.078
        threshold = max(_AD, (_PHI / 10.0) * resolution)
        scored_verbs.sort(reverse=True)
        best_verb = None
        if scored_verbs and scored_verbs[0][0] > threshold:
            best_verb = scored_verbs[0][1]

        # Fallback if needed — prefer a real verb from the fingerprint
        # over a hardcoded generic word that bleeds across prompts.
        # Scan pkt=1 words for the highest-tension word that reads as a verb
        # (ends in common verb suffixes or appears in known action patterns).
        if not best_verb:
            _VERB_ENDINGS = ("s","es","ed","ing","ize","ise","ate","fy","en")
            # Noun endings that override verb suffix matches — these words
            # end in verb-like suffixes but are definitely nouns
            _NOUN_ENDINGS = ("tion","sion","ness","ment","ity","ance","ence",
                             "ogen","agen","ogen","gen","ism","ist","ogy",
                             "ium","ary","ery","ory","phy","thy","chy",
                             "nce","nse","dge","lge","rge","nge")
            _fp_verbs = []
            if fingerprint:
                for w in fingerprint.get("per_word", []):
                    if w.get("pocket", 0) == 1:
                        wl = w.get("word","").lower().rstrip(".,!?;:")
                        t  = abs(w.get("mean_tension", 0.0))
                        ns = abs(w.get("net_signed", 0.0))
                        if (any(wl.endswith(e) for e in _VERB_ENDINGS)
                                and not any(wl.endswith(e) for e in _NOUN_ENDINGS)
                                and wl not in self._Q_SKIP_VERBS
                                and wl not in _STRUCTURAL_ANCHORS
                                and 0.8 <= ns < 3.0 and t > 0.05):
                            _fp_verbs.append((t * ns, wl))
            if _fp_verbs:
                _fp_verbs.sort(reverse=True)
                best_verb = _fp_verbs[0][1]
            elif qtype == "causal":
                best_verb = "causes"
            elif qtype == "process":
                best_verb = "produces"
            else:
                best_verb = "requires"

        # Insertion: purely field-driven (outside base frame delta)
        result = ordered[:]
        if best_verb and len(result) > 1:
            # Decide position solely from carry_sign + polarity (no pos, no gaps)
            if carry_sign > 0 and polarity > 0.05:
                insert_idx = 1                          # strongly action-forward
            elif carry_sign < 0 or polarity < -0.1:
                insert_idx = len(result) - 1            # reflective / causal at end
            else:
                insert_idx = len(result) // 2           # neutral middle

            result.insert(insert_idx, {"word": best_verb, "pool": "verb", "pos": 0.5})

        return result

    _BRIDGE_PREFS = {
        "causal":  ["because", "and", "as", "from"],
        "process": ["through", "using", "and"],
        "entity":  ["and", "in", "the"],
        "spatial": ["in", "and", "of"],
        "select":  ["and", "the"],
    }
    _WH_CONNECTORS = {"where", "when", "who", "which", "how", "why", "what", "that"}
    _DRESS_GAP_THRESHOLD = 0.28

    def _dress_output(self, word_dicts: List[Dict], qtype: str, vocabulary: Any) -> List[str]:
        if len(word_dicts) <= 1:
            return [w["word"] for w in word_dicts]

        prefs = self._BRIDGE_PREFS.get(qtype, ["and"])
        bridge = prefs[0]
        if hasattr(vocabulary, "get_stable_words"):
            named_words = {e.get("word", "").lower() for e in vocabulary.get_stable_words()
                           if e.get("named", False) or e.get("appearances", 0) >= 2}
            for p in prefs:
                if p in named_words:
                    bridge = p
                    break

        result = []
        bridges_inserted = 0

        for i, w in enumerate(word_dicts):
            result.append(w["word"])
            if i < len(word_dicts) - 1 and bridges_inserted < 1:
                gap = word_dicts[i + 1]["pos"] - w["pos"]
                curr_pool = w.get("pool", "content")
                next_pool = word_dicts[i + 1].get("pool", "content")
                curr_word = w["word"].lower().rstrip(".!?,;:")

                is_bridgeable = (
                    curr_pool != "action" and next_pool != "action" and
                    curr_word not in self._WH_CONNECTORS and
                    curr_word not in _STRUCTURAL_ANCHORS
                )

                if gap > self._DRESS_GAP_THRESHOLD and is_bridgeable:
                    result.append(bridge)
                    bridges_inserted += 1

        return result

    _QTYPE_DEFAULT_VERB = {
        "process": "uses",
        "causal":  "causes",
        "entity":  "contains",   # 'involves' was bleeding; 'contains' is neutral
        "spatial": "exists in",
        "select":  "selects",
    }

    def _articulate(self, word_dicts: List[Dict], per_word: List[Dict],
                    anchor_word: Optional[str], qtype: str) -> str:
        if not word_dicts: return ""

        words = [w["word"] for w in word_dicts]
        has_action = any(w.get("pool") == "action" for w in word_dicts)

        # If action word already present OR no anchor to conjugate to,
        # return words as-is — combiner handles verb placement separately.
        if has_action or anchor_word is None:
            return " ".join(words) + "."

        question_verb = self._extract_question_verb(per_word, anchor_word)
        if not question_verb:
            question_verb = self._QTYPE_DEFAULT_VERB.get(qtype, "contains")

        # Noun guard: if anchor_word is a content noun from the fingerprint,
        # don't conjugate it — it's the SUBJECT, not the verb.
        # Nouns have non-zero net_signed charge in per_word.
        # Conjugating a noun produces 'myelins', 'oxidations', 'evolutions' etc.
        _anchor_lower = anchor_word.lower().rstrip(".,!?;:")
        _known_nouns = {
            w.get("word", "").lower().rstrip(".,!?;:")
            for w in per_word
            if abs(w.get("net_signed", 0.0)) > 0.5
            and w.get("word", "").lower().rstrip(".,!?;:") not in _STRUCTURAL_ANCHORS
        }
        if _anchor_lower in _known_nouns:
            # Anchor is a noun — use verb as-is, don't conjugate anchor
            conj = question_verb
        else:
            conj = self._conjugate(question_verb, anchor_word)

        anchor_clean = anchor_word.rstrip(".!?,;:").lower()
        if words and words[0].lower().rstrip(".!?,;:") == anchor_clean.split()[0]:
            result = [words[0], conj] + words[1:]
        else:
            result = [anchor_word, conj] + words

        return " ".join(result) + "."

    def _extract_question_verb(self, per_word: List[Dict], anchor_word: str) -> Optional[str]:
        anchor_parts = set(anchor_word.lower().rstrip(".!?,;:").split())
        pkt1 = [w for w in per_word if w.get("pocket", 0) == 1]

        for w in pkt1:
            wl = w.get("word", "").lower().rstrip("?!.,;:")
            if wl in self._Q_SKIP_VERBS: continue
            if wl in anchor_parts: continue
            ns = abs(w.get("net_signed", 0.0))
            if ns > 3.0: continue
            _VERB_S_ENDINGS = ('ates','etes','ites','otes','utes','izes','ises','akes','okes','ikes','aves','oves','ives','ures','ares','eres','ires','ores','ses','ces','ges','zes','xes','nds','rts','lts','nts')
            if (wl.endswith('s') and ns > 1.0 and not wl.endswith(('ess','ous','ius')) and not wl.endswith(_VERB_S_ENDINGS)):
                continue
            return wl
        return None

    def _conjugate(self, verb: str, anchor: str) -> str:
        if not verb or not anchor: return verb
        if verb.endswith(("ing", "ed")): return verb
        if verb.endswith("es") and not verb.endswith("ees"): return verb

        anchor_last = anchor.split()[-1].lower().rstrip(".,!?;:")
        singular_endings = ("ss", "us", "is", "ous", "ness", "sis", "ae", "ix", "ex", "um", "on")
        is_plural = anchor_last.endswith("s") and not anchor_last.endswith(singular_endings)

        if is_plural: return verb
        if verb.endswith(("s", "x", "z", "ch", "sh")): return verb + "es"
        if verb.endswith("y") and len(verb) > 2 and verb[-2] not in "aeiou": return verb[:-1] + "ies"
        return verb + "s"

    def _post_touchup(self, text: str) -> str:
        if not text or text.startswith("Field geometry"):
            return text

        text = text[0].upper() + text[1:]

        if not text.endswith((".", "!", "?")):
            text = text.rstrip() + "."

        text = text.replace("yearses", "years")
        text = text.replace("fincheses", "finches")
        text = text.replace("ratss", "rats")
        text = text.replace("penguinses", "penguins")
        text = text.replace("statudes", "statud")

        # NOTE: Removed blunt involves-injection fallback.
        # The contextual_combiner handles verb placement.
        # Silence is better than a semantically wrong appended verb.

        return text.strip()

    def _verify_parity(self, generated_text: str, input_carry_sign: int) -> Tuple[float, bool]:
        if not generated_text or generated_text.startswith("Field geometry"):
            return 0.0, False
        tri = self._sw.triangulate(generated_text)
        syms = tri.get("symbol_stream", [])
        zero_ch = chr(48)
        gen_sum = sum(symbol_to_signed(s) / 13.0 for s in syms if s != zero_ch)
        if abs(gen_sum) < 1e-4:
            return 0.5, False
        gen_sign = math.copysign(1.0, gen_sum)
        if input_carry_sign == 0:
            return 0.5, False
        alignment = gen_sign * input_carry_sign
        return float(alignment), alignment >= _PARITY_THRESHOLD

    def generate(self, fingerprint: Dict[str, Any], vocabulary: Any, invariant_engine: Any,
                 consensus: float, persistence: float,
                 pressure_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        field = self._read_field(fingerprint=fingerprint)
        field["persistence"] = persistence
        target = self._identify_target_region(field)
        candidates = self._sample_vocabulary(
            target, vocabulary, invariant_engine, fingerprint,
            target_side=target["side"], pressure_state=pressure_state
        )
        # ── Question-only: geometric library query ───────────────────────────
        # When pkt=0 is empty or minimal (question-only prompt), the normal
        # candidate pool only draws from the question words themselves.
        # Instead of faking context, query the library geometrically:
        # find words whose group/tension signature matches the question's
        # own geometric fingerprint. These are the library's best answer
        # to "what is geometrically related to this question?"
        _fp_pkt0_count = len([w for w in (fingerprint.get("per_word",[]) if fingerprint else [])
                              if w.get("pocket",0) == 0])
        _fp_pkt1_count = len([w for w in (fingerprint.get("per_word",[]) if fingerprint else [])
                              if w.get("pocket",0) == 1])
        # Question-only: pkt=1 is empty (no pocket split) OR very sparse.
        # With per-word scoring, pure questions have pkt=1=0 because there's
        # no boundary to push words into pkt=1. Use full fingerprint as query.
        _is_q_only_gen = _fp_pkt1_count == 0 or (_fp_pkt0_count <= 3 and _fp_pkt1_count <= 2)

        if _is_q_only_gen and fingerprint and hasattr(vocabulary, "get_stable_words"):
            # Use ALL per_word as query — full question fingerprint
            _q_per_word = fingerprint.get("per_word",[])
            if _q_per_word:
                _q_groups   = {w.get("dominant_group", w.get("grp",-1))
                               for w in _q_per_word}
                _q_mean_t   = sum(abs(w.get("mean_tension",0)) for w in _q_per_word) / len(_q_per_word)
                _q_mean_ns  = sum(abs(w.get("net_signed",0)) for w in _q_per_word) / len(_q_per_word)

                # Words in the current question fingerprint — exclude from library
                _q_word_set = {w.get("word","").lower().rstrip(".,!?;:")
                               for w in _q_per_word}

                # Score every library word by geometric proximity to question
                _lib_candidates = []
                for entry in vocabulary.get_stable_words():
                    wl = entry.get("word","").lower()
                    if wl in _STRUCTURAL_ANCHORS: continue
                    if len(wl) < 3: continue
                    # Skip words already in the question — they're already candidates
                    if wl in _q_word_set: continue
                    # Only use words with sufficient appearance history (stability)
                    # Session-fresh words have inflated tension — skip them
                    if entry.get("appearances", 0) < 3: continue
                    _grp  = entry.get("dominant_group", -1)
                    _t    = abs(entry.get("mean_tension", 0.0))
                    _ns   = abs(entry.get("net_signed", 0.0))
                    # BOUNDED ORBIT DISTANCE METRIC
                    # From Ouroboros: S_{k+1} = {x | |f^k(x)+δ| < r_resonant}
                    # A library word is a candidate if its orbit position
                    # is within r_resonant of the question's orbit position.
                    #
                    # r_resonant = 1/φ² = _PARITY_THRESHOLD = 0.381966
                    # This is the system's own convergence radius — the
                    # same constant used in polarity/pressure thresholding.
                    _NS_SCALE   = 6.0     # normalize ns to [-1, 1] range
                    _GRP_SCALE  = 13.0    # normalize group to [-1, +1] — signed Dual-13 range
                    _R_RESONANT = 0.381966  # 1/φ² — parity threshold

                    # 1. Euclidean orbit distance in normalized field space
                    _dt  = abs(_t - _q_mean_t)
                    _dns = abs(_ns / _NS_SCALE - _q_mean_ns / _NS_SCALE)
                    _dg  = abs(_grp / _GRP_SCALE - (_q_grp_mean := (
                        sum(_q_groups) / max(len(_q_groups), 1)) / _GRP_SCALE)) * 0.5
                    _orbit_dist  = (_dt**2 + _dns**2 + _dg**2) ** 0.5
                    _orbit_score = math.exp(-_orbit_dist / _R_RESONANT)

                    # 2. Cosine trajectory similarity (direction in t,ns space)
                    # Words on same trajectory have same tension/charge ratio.
                    # Negative-charge words (opposite orbit) score 0.
                    _dot  = _t * _q_mean_t + (_ns/_NS_SCALE) * (_q_mean_ns/_NS_SCALE)
                    _mag1 = (_t**2 + (_ns/_NS_SCALE)**2) ** 0.5
                    _mag2 = (_q_mean_t**2 + (_q_mean_ns/_NS_SCALE)**2) ** 0.5
                    _cos_score = max(0.0, _dot / max(_mag1 * _mag2, 1e-8))

                    # 3. Group resonance bonus (same symbol group = same resonance)
                    _q_grp_int = round(sum(_q_groups) / max(len(_q_groups), 1))
                    # Same sign hemisphere = shared axis orientation → group bonus
                    # Adjacent integer distance ≤ 2 on same hemisphere also scores
                    _same_hemi = ((_grp > 0 and _q_grp_int > 0) or
                                  (_grp < 0 and _q_grp_int < 0) or
                                  (_grp == 0 and _q_grp_int == 0))
                    _grp_bonus = (0.2 if _grp == _q_grp_int
                                  else 0.15 if (_same_hemi and abs(_grp - _q_grp_int) <= 2)
                                  else 0.1 if abs(_grp - _q_grp_int) <= 2 else 0.0)

                    # Combined bounded orbit score
                    _lib_score = (_orbit_score * 0.5
                                  + _cos_score  * 0.3
                                  + _grp_bonus  * 0.2)
                    if _lib_score > 0.3:
                        _lib_candidates.append({
                            "word":       entry.get("word", wl),
                            "net_signed": _ns,
                            "source":     "library_query",
                            "priority":   2,
                            "named":      entry.get("appearances",0) >= 2,
                            "pocket_mult": 1.0,
                            "pocket_label": "lib_q",
                            "score":      _lib_score,
                        })

                # Add top library matches to candidate pool
                _lib_candidates.sort(key=lambda c: c["score"], reverse=True)
                candidates = candidates + _lib_candidates[:8]

        # ── Score-gated output quality filter ────────────────────────────────
        # Only pass candidates above a resolution-scaled threshold to assembly.
        # Cold field (res≈0.15): threshold≈0.35 — liberal, field needs words.
        # Warm field (res≈0.80): threshold≈0.45 — selective, prune low-signal.
        # This stops the assembly padding output with geometrically weak words
        # that dilute the high-quality candidates selected first.
        _res_now   = field.get("resolution", 0.15)
        _score_floor = max(0.35, 0.35 + (_res_now - 0.5) * 0.3)
        _gated = [c for c in candidates if c.get("score", 0.0) >= _score_floor]
        # Also remove structural anchors that slipped through scoring
        # (e.g. 'these','must','adds' can score above floor but must never be output)
        _gated = [c for c in _gated
                  if c.get("word", "").lower().rstrip(".,!?;:") not in _STRUCTURAL_ANCHORS]
        # Always keep at least top 2 — assembly needs minimum words
        if len(_gated) < 2:
            _gated = sorted(
                [c for c in candidates
                 if c.get("word","").lower().rstrip(".,!?;:") not in _STRUCTURAL_ANCHORS],
                key=lambda c: c.get("score", 0.0), reverse=True
            )[:2]
        candidates_for_assembly = _gated

        text, template = self._multi_pass_assembly(candidates_for_assembly, field, target, vocabulary, fingerprint=fingerprint, invariant_engine=invariant_engine)
        carry_sign = field["carry_sign"]
        alignment, locked = self._verify_parity(text, carry_sign)
        # Threshold lowered 0.7→0.3: experiments confirmed polarization
        # is correct even at resolution=0.15 (cold start).
        # Resolution and polarization are orthogonal axes.
        if locked and field["resolution"] >= 0.3:
            confidence = "high"
        elif alignment >= 0.0:
            confidence = "medium"
        else:
            confidence = "low"

        # Polarization level after this prompt
        ps = pressure_state or {}
        top_score = max((c.get("score", 0.0) for c in candidates), default=0.0)
        if top_score < 0.650:   level_after = 0
        elif top_score < 0.950: level_after = 1
        elif top_score < 1.110: level_after = 2
        else:                   level_after = 3
        domain_flipped = (level_after != ps.get("level_current", level_after))

        return {
            "text":           text,
            "parity_locked":  locked,
            "alignment":      round(alignment, 4),
            "template":       template,
            "field_polarity": round(field["polarity"], 4),
            "target_region":  target,
            "candidates":     [c["word"] for c in candidates_for_assembly],
            "pocket_scores":  [{"word": c["word"], "pocket_mult": c.get("pocket_mult", 1.0), "pocket_label": c.get("pocket_label", "?"), "score": round(c.get("score", 0.0), 4)} for c in candidates[:6]],
            "confidence":     confidence,
            "resolution":     field["resolution"],
            # ── Pressure state report ─────────────────────────────────
            "pressure_mode":  ps.get("mode", "UNKNOWN"),
            "pressure_delta": ps.get("pressure_delta", 0.0),
            "G_actual":       ps.get("G_actual", 0.0),
            "G_needed":       ps.get("G_needed", 0.0),
            "P0_current":     ps.get("P0_current", 0.0),
            "level_current":  ps.get("level_current", 0),
            "level_after":    level_after,
            "domain_flipped": domain_flipped,
        }

    def format_output(self, result: Dict[str, Any],
                      fingerprint: Optional[Dict[str, Any]] = None) -> str:
        from language.output_translator import translate_raw
        text       = translate_raw(result["text"], fingerprint=fingerprint)
        locked     = result["parity_locked"]
        alignment  = result["alignment"]
        confidence = result["confidence"]
        resolution = result["resolution"]
        polarity   = result["field_polarity"]
        candidates = result.get("candidates", [])
        if locked and confidence == "high":
            return text
        if locked:
            return f"{text} [parity confirmed, alignment {alignment:+.3f}]"
        if confidence == "medium":
            return (f"{text} [geometric approximation — "
                    f"field polarity {polarity:+.3f}, resolution {resolution:.3f}]")
        candidate_str = ", ".join(candidates[:3]) if candidates else "none"
        return (f"Field geometry active but parity unconfirmed. "
                f"Strongest candidates: {candidate_str}. "
                f"Polarity {polarity:+.3f}, resolution {resolution:.3f}.")


geometric_output = GeometricOutput()