"""
language/geometric_output.py  (v17 — multi-mode three-layer with combiner fully outside base frame delta)
==================================================================================
Contextual combiner now operates completely outside the base frame delta / stream_pos / gap timing.
It still uses the exact same mechanics (Method 1 geometric scoring, multi-mode library pool, polarity/carry/resolution).
Placement is now purely driven by the on-the-fly field snapshot (carry_sign + polarity) — timeless relative to the base system.
Everything else (Layer 1 heavy, Layer 2 connectors, multi-mode library access, articulation, parity) is unchanged.
"""

import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set

from core.invariants import invariants
from utils.symbol_grouping import symbol_to_signed, symbol_grouping
from utils.bipolar_lattice import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from wave.symbolic_wave import SymbolicWave

_PARITY_THRESHOLD           = 0.35
_CONTENT_THRESHOLD          = 1.5
_ANSWER_CONTENT_THRESHOLD   = 0.8
_BOUNDARY_CONTENT_THRESHOLD = 0.8
_NET_TENSION_SCALE          = 8.0

_STRUCTURAL_ANCHORS = {
    "the", "and", "is", "in", "of", "a", "to", "it", "as",
    "that", "this", "was", "be", "are", "for", "on", "or",
    "from", "by", "at", "an", "not", "but", "so", "if",
    "its", "has", "had", "have", "with", "how", "what",
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

_MULT_ANSWER_CANDIDATE     = 2.5
_MULT_CONTEXT_ONLY         = 1.5
_MULT_ANCHOR               = 1.0
_MULT_QUESTION_ONLY        = 0.4
_MULT_UNKNOWN              = 0.8
_MULT_SAME_SESS_OTHER      = 0.4
_MULT_CROSS_SESS_CLOSE     = 0.75
_MULT_CROSS_SESS_MID       = 0.50
_MULT_CROSS_SESS_FAR       = 0.30
_EXHAUST_CLOSE_THRESHOLD   = 0.002
_CROSS_SESS_THRESHOLD      = 3.0
_EXHAUST_MID_THRESHOLD     = 0.02


class GeometricOutput:
    def __init__(self):
        self._sw = SymbolicWave()

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
                           target_side: str = "boundary") -> List[Dict[str, Any]]:
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
                score = (abs(ns) / 13.0) * pmult
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
                score = (abs(ns) / 13.0) * pmult
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
            pmult, plabel = pocket_multiplier(word, from_current, word_epoch=sv_epoch)
            thresh = effective_threshold(pmult, target_side)
            # Skip library words whose polarity conflicts with field direction
            _fnet = fingerprint.get('net_tension', 0.0) if fingerprint else 0.0
            if _fnet > 0.5 and ns < -0.5: continue
            if _fnet < -0.5 and ns > 0.5: continue

            if low <= ns <= high and abs(ns) >= thresh:
                score = (abs(ns) / 13.0) * pmult
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "stable_vocab", "priority": 1,
                    "named": invariant_engine.is_named(word),
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
                             fingerprint: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        if not candidates:
            return ("Field geometry unresolved — no vocabulary in target region.", "fallback")

        qtype, q_link = self._detect_question_type(fingerprint)

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
        light = []
        if hasattr(vocabulary, "get_stable_words"):
            for entry in vocabulary.get_stable_words():
                word = entry.get("word", "").lower()
                if word in _STRUCTURAL_ANCHORS: continue
                ns = entry.get("net_signed", 0.0)
                t = entry.get("mean_tension", 0.0)
                pmult, plabel = self._pocket_multiplier(word, fingerprint, from_current=False)
                thresh = self._effective_threshold_for_connector(pmult, field)
                if 0.1 <= abs(ns) < _CONTENT_NS_MIN and t > 0.08 and abs(ns) >= thresh:
                    light.append({"word": entry.get("word"), "ns": ns, "t": t, "pos": 0.5, "pool": "connector"})

        # Merge Layers 1 + 2
        ordered = heavy + light
        if ordered:
            ordered = sorted(ordered, key=lambda x: x.get("pos", 0.5))
            for i, item in enumerate(ordered):
                item["pos"] = i / max(len(ordered) - 1, 1)

        # Layer 3: Contextual combiner — now fully outside base frame delta
        final_words = self._contextual_combiner(ordered, field, fingerprint, qtype, vocabulary)

        # Final articulation — detect anchor from pkt=1 subject
        # The first non-skip word in pkt=1 after question words is the subject
        _anchor = None
        if fingerprint:
            _pkt1 = [w for w in fingerprint.get("per_word", [])
                     if w.get("pocket", 0) == 1]
            _skip = {'how','why','what','where','when','which','who',
                     'do','does','did','is','are','was','were','will',
                     'would','can','could','should','may','might','be'}
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
                             vocabulary: Any) -> List[Dict]:
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
        verb_pool = []
        if fingerprint:
            for w in fingerprint.get("per_word", []):
                if w.get("pocket", 0) == 1:
                    wl = w.get("word", "").lower().rstrip("?!.,;:")
                    if wl not in self._Q_SKIP_VERBS and abs(w.get("net_signed", 0.0)) < 3.0:
                        verb_pool.append({"word": w.get("word"), "ns": w.get("net_signed", 0.0), "t": w.get("mean_tension", 0.0)})

        if hasattr(vocabulary, "get_stable_words"):
            for entry in vocabulary.get_stable_words():
                wl = entry.get("word", "").lower()
                ns = entry.get("net_signed", 0.0)
                t = entry.get("mean_tension", 0.0)
                if _ACTION_NS_MIN <= abs(ns) <= _ACTION_NS_MAX and t > _ACTION_T_MIN:
                    verb_pool.append({"word": entry.get("word"), "ns": ns, "t": t})

        # Score verbs geometrically (unchanged)
        scored_verbs = []
        for v in verb_pool:
            score = (v["t"] * abs(v["ns"])) * resolution * (1.0 + abs(polarity)) * (1.0 + abs(carry))
            scored_verbs.append((score, v["word"]))

        # Threshold lowered: 0.65*res was too high (~0.47) for most real scores
        # which average ~0.15-0.20. Now uses 0.12*res as the floor.
        threshold = max(0.04, 0.12 * resolution)
        scored_verbs.sort(reverse=True)
        best_verb = None
        if scored_verbs and scored_verbs[0][0] > threshold:
            best_verb = scored_verbs[0][1]

        # Fallback if needed
        if not best_verb:
            if qtype == "causal":
                best_verb = "causes"
            elif qtype == "process":
                best_verb = "through"
            else:
                best_verb = "involves"

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
        "entity":  "involves",
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
            question_verb = self._QTYPE_DEFAULT_VERB.get(qtype, "involves")

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
                 consensus: float, persistence: float) -> Dict[str, Any]:
        field = self._read_field(fingerprint=fingerprint)
        field["persistence"] = persistence
        target = self._identify_target_region(field)
        candidates = self._sample_vocabulary(target, vocabulary, invariant_engine, fingerprint, target_side=target["side"])
        text, template = self._multi_pass_assembly(candidates, field, target, vocabulary, fingerprint=fingerprint)
        carry_sign = field["carry_sign"]
        alignment, locked = self._verify_parity(text, carry_sign)
        if locked and field["resolution"] >= 0.7:
            confidence = "high"
        elif alignment >= 0.0:
            confidence = "medium"
        else:
            confidence = "low"
        return {
            "text":           text,
            "parity_locked":  locked,
            "alignment":      round(alignment, 4),
            "template":       template,
            "field_polarity": round(field["polarity"], 4),
            "target_region":  target,
            "candidates":     [c["word"] for c in candidates],
            "pocket_scores":  [{"word": c["word"], "pocket_mult": c.get("pocket_mult", 1.0), "pocket_label": c.get("pocket_label", "?"), "score": round(c.get("score", 0.0), 4)} for c in candidates[:6]],
            "confidence":     confidence,
            "resolution":     field["resolution"],
        }

    def format_output(self, result: Dict[str, Any]) -> str:
        text       = result["text"]
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