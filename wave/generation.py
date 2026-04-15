from typing import Dict, Any, Tuple, Optional
import re
import numpy as np
from core.clarity_ratio import clarity_ratio
from core.safeguards import safeguards
from utils.radial_displacer import radial_displacer
from core.semantic_layer import semantic_layer
from utils.fold_line_resonance import fold_line_resonance

_CONSENSUS_AGREE    =  0.20
_CONSENSUS_DIVIDED  = -0.20
_PERSISTENCE_HIGH   =  0.35
_EXHAUST_MAX_DIST   =  0.05   # max distance to act on exhaust recall
_EXHAUST_RES_GUARD  =  0.65   # local resolution above which exhaust recall yields
_EXHAUST_PKT_GUARD  =  0.30   # pocket confidence above which exhaust recall yields


def _modulate(answer: str, consensus: float, persistence: float) -> str:
    high_p = persistence >= _PERSISTENCE_HIGH
    pos_c  = consensus   >= _CONSENSUS_AGREE
    neg_c  = consensus   <= _CONSENSUS_DIVIDED
    if high_p and pos_c:
        return answer
    if high_p and neg_c:
        return (f"{answer} "
                f"(field stable, observers divided "
                f"[consensus {consensus:.2f}, persistence {persistence:.2f}])")
    if not high_p and pos_c:
        return (f"{answer} "
                f"(observers converging, wave still forming "
                f"[consensus {consensus:.2f}, persistence {persistence:.2f}] — "
                f"structure resolving)")
    if not high_p and neg_c:
        return (f"{answer} "
                f"(radial field unresolved "
                f"[consensus {consensus:.2f}, persistence {persistence:.2f}])")
    return (f"{answer} "
            f"(field ambiguous "
            f"[consensus {consensus:.2f}, persistence {persistence:.2f}])")


def _similarity_recall(
    prompt:       str,
    tri_data:     Dict[str, Any],
    memory_store: list,
) -> Optional[str]:
    if not memory_store:
        return None
    current_stream = tri_data.get("symbol_stream", [])
    if not current_stream:
        return None
    _THRESHOLD    = 0.22
    prompt_words  = prompt.split()
    subject_names = {
        w.lower().rstrip('?.,!') for w in prompt_words
        if w[0].isupper() and len(w) > 1
        and w.lower() not in {"i","earlier","remember","you","we","the"}
    }
    best_score  = 0.0
    best_answer = None
    current_set = set(current_stream)
    for entry in reversed(memory_store):
        prior_stream = entry.get("symbol_stream", [])
        prior_answer = entry.get("answer", "")
        prior_prompt = entry.get("prompt", "").lower()
        if not prior_stream or not prior_answer:
            continue
        if subject_names:
            if not any(name in prior_prompt for name in subject_names):
                continue
        min_len = min(len(current_stream), len(prior_stream))
        if min_len == 0:
            continue
        positional  = sum(
            1 for i in range(min_len)
            if current_stream[i] == prior_stream[i]
        ) / min_len
        prior_set   = set(prior_stream)
        set_overlap = len(current_set & prior_set) / max(len(current_set), 1)
        raw_score   = 0.5 * positional + 0.5 * set_overlap
        len_ratio   = min_len / max(len(current_stream), len(prior_stream))
        score       = raw_score * (len_ratio ** 0.5)
        if score >= best_score:
            best_score  = score
            best_answer = prior_answer
    if best_score >= _THRESHOLD and best_answer:
        return f"{best_answer} (recall, similarity {best_score:.2f})"
    return None


class AnswerGenerator:
    """
    Geometry-driven answer generator — semantic scaffolding removed.

    Resolution order:
      1. Same-turn memory recall
      2. Cross-turn geometric similarity recall
      3. Pocket confidence signal
      4. Geometry-only report: consensus, persistence, convergence, spin
    """

    def __init__(self):
        self.memory_store: list = []

    @staticmethod
    def _strip_annotations(answer: str) -> str:
        clean = re.sub(
            r'\s+\((?:field stable|observers|wave|radial field|field ambiguous|'
            r'recall|pocket confidence|Field resolved|Field processing|'
            r'Geometry stable|dual held|vertical build|horizontal recognize)'
            r'[^)]*\).*$',
            '', answer, flags=re.IGNORECASE | re.DOTALL
        )
        return clean.strip()

    def _store(self, tri_data: Dict[str, Any], prompt: str, answer: str) -> None:
        stream = tri_data.get("symbol_stream", [])
        if not (stream and answer and len(answer) > 4):
            return
        skip = [
            "field resolving", "field stable", "field ambiguous",
            "radial field", "observers converging", "geometry reports",
            "no prior context", "geometric pattern", "radial signal",
            "reversal applied", "web is still forming", "field resolved",
            "field processing", "geometry stable", "dual held",
            "strong geometric boundary", "geometric boundary plausible",
        ]
        if any(m in answer.lower() for m in skip):
            return
        clean = self._strip_annotations(answer)
        if len(clean) > 4:
            self.memory_store.append({
                "symbol_stream": stream,
                "answer":        clean,
                "prompt":        prompt,
            })

    def _geometry_report(
        self,
        consensus:      float,
        persistence:    float,
        convergence:    float,
        pocket_conf:    float,
        num_displacers: int,
        clarity:        float,
        is_question:    bool,
        spin_signal:    float,
        zero_braking:   bool,
    ) -> str:
        if zero_braking:
            spin_desc = "zero unresolved (dual held — braking)"
        elif spin_signal > 0:
            spin_desc = f"positive ({spin_signal:+.3f}, vertical build)"
        else:
            spin_desc = f"negative ({spin_signal:+.3f}, horizontal recognize)"

        if (is_question
                and consensus >= _CONSENSUS_AGREE
                and persistence >= _PERSISTENCE_HIGH):
            return (f"Field resolved. "
                    f"Convergence {convergence:.3f} across {num_displacers} displacers. "
                    f"Spin {spin_desc}. "
                    f"Pocket {pocket_conf:.2f}.")

        if is_question:
            return (f"Field processing. "
                    f"Convergence {convergence:.3f} | "
                    f"consensus {consensus:.2f} | "
                    f"persistence {persistence:.2f} | "
                    f"spin {spin_desc}.")

        return (f"Geometry stable. "
                f"Convergence {convergence:.3f} | "
                f"clarity {clarity:.3f} | "
                f"spin {spin_desc}.")

    def _resolve(
        self,
        prompt:         str,
        tri_data:       Dict[str, Any],
        prop_result:    Dict[str, Any],
        consensus:      float,
        memory_recall:  str,
        convergence:    float,
        num_displacers: int,
        clarity:        float,
        persistence:    float,
        pocket_conf:    float,
        prompt_lower:   str,
        spin_signal:    float,
        zero_braking:   bool,
        exhaust_recall: Optional[Dict] = None,
    ) -> str:

        is_question = (
            prompt_lower.endswith("?") or
            any(w in prompt_lower for w in semantic_layer.question_words)
        )

        # ── Step 0: Exhaust-diagonal recall (local-resolution guard) ──────────
        # Cross-session geometric fingerprint match. Only fires when the local
        # field is NOT already strongly resolved — strong local geometry wins.
        if exhaust_recall is not None:
            dist         = exhaust_recall.get("distance", 1.0)
            prior_prompt = exhaust_recall.get("prompt", "")
            local_res    = fold_line_resonance.get_resolution_score()
            if (dist < _EXHAUST_MAX_DIST and prior_prompt and
                    not (local_res >= _EXHAUST_RES_GUARD or
                         pocket_conf >= _EXHAUST_PKT_GUARD)):
                cross = _similarity_recall(prior_prompt, tri_data, self.memory_store)
                if cross:
                    return cross
                snippet = prior_prompt[:60].rstrip()
                return (f"Field geometry matches prior context "
                        f"(exhaust distance {dist:.4f}): '{snippet}...'")
            # Local field is strong — fall through to local resolution

        # 1. Same-turn memory recall
        if memory_recall and len(memory_recall) > 10 \
                and "turbulence prune" not in memory_recall.lower():
            recall_prompt = memory_recall.split(" || ")[0].strip()
            segments = [s.strip() for s in recall_prompt.split(" | ") if s.strip()]
            if segments:
                candidate = segments[-1]
                if (len(candidate) >= 8
                        and not all(c in ".?!0 " for c in candidate)
                        and candidate.lower() not in prompt_lower
                        and prompt_lower not in candidate.lower()):
                    return candidate

        # 2. Cross-turn geometric similarity recall
        _MEMORY_REF = {
            "i told you","you said","you told me","we discussed",
            "earlier","remember","previously","last time",
            "you mentioned","i mentioned","as i said","from before"
        }
        context, _ = semantic_layer._split_context_and_query(prompt)
        is_memory_reference = any(ref in context.lower() for ref in _MEMORY_REF)
        if not context or is_memory_reference:
            cross = _similarity_recall(prompt, tri_data, self.memory_store)
            if cross:
                return cross

        # 3. Pocket confidence signal
        if pocket_conf >= 0.4:
            return (f"Strong geometric boundary detected "
                    f"(pocket confidence {pocket_conf:.2f}). "
                    f"Field has not yet resolved content from geometry alone.")
        if pocket_conf >= 0.2:
            return (f"Geometric boundary plausible "
                    f"(pocket confidence {pocket_conf:.2f}). "
                    f"Field processing.")

        # 4. Geometry-only report
        return self._geometry_report(
            consensus, persistence, convergence, pocket_conf,
            num_displacers, clarity, is_question, spin_signal, zero_braking
        )

    def generate(
        self,
        prompt:         str,
        tri_data:       Dict[str, Any],
        prop_result:    Dict[str, Any],
        consensus:      float,
        memory_recall:  str = "",
        exhaust_recall: Optional[Dict] = None,
    ) -> str:

        prompt_lower = prompt.lower().strip()
        clarity      = prop_result.get("clarity_ratio_score", clarity_ratio.current_ratio)
        persistence  = prop_result.get("persistence", 0.0)

        numeric_waveform = [
            x for x in prop_result.get("waveform_sample", [0.1])
            if isinstance(x, (int, float))
        ]
        waveform_amp = float(np.mean(np.abs(numeric_waveform))) if numeric_waveform else 0.0

        radial         = radial_displacer.generate_structure(
            prompt, tri_data, wave_amplitude=waveform_amp
        )
        convergence    = radial["web_convergence_score"]
        num_displacers = radial["num_displacers"]

        _, pocket_conf = semantic_layer.extract_with_pocket_alignment(
            prompt, tri_data, prop_result
        )

        spin_signal  = 0.0
        zero_braking = False
        try:
            from utils.bipolar_lattice import bipolar_lattice as _bl
            spin_signal  = _bl._ring_spin_signal()
            zero_braking = (spin_signal == 0.0)
        except Exception:
            pass

        answer = self._resolve(
            prompt, tri_data, prop_result, consensus, memory_recall,
            convergence, num_displacers, clarity, persistence,
            pocket_conf, prompt_lower, spin_signal, zero_braking,
            exhaust_recall=exhaust_recall,
        )

        self._store(tri_data, prompt, answer)

        already_annotated = any(m in answer for m in [
            "recall, similarity", "pocket confidence",
            "Field resolved", "Field processing", "Geometry stable",
            "dual held", "vertical build", "horizontal recognize",
            "Strong geometric boundary", "Geometric boundary plausible",
            "field geometry matches",
        ])
        if not already_annotated:
            answer = _modulate(answer, consensus, persistence)

        return answer

    def simple_generate(self, prompt: str) -> str:
        return self.generate(
            prompt,
            {"width": 4, "height": 4, "total_triangles": 8,
             "n_original": len(prompt), "symbol_stream": [], "zero_breaks": []},
            {"clarity_ratio_score": 1.0, "persistence": 0.5,
             "waveform_sample": [0.1], "steps": 100},
            0.0, ""
        )


generator = AnswerGenerator()