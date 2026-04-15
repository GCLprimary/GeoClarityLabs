"""
diagnostics/semantic_probe.py
==============================
Semantic Diagnostics — Symbol Basis Matrix + Excitation Sequence

Three-layer diagnostic:

  LAYER 1 — BASIS MATRIX
    All 27×27 symbol pair tensions computed from current fold-line
    imprint state. This is the geometric ground truth — how the system
    sees every 1:1 symbol relationship before any sequence bias.
    Recomputed after each excitation batch so you can see drift.

  LAYER 2 — EXCITATION SEQUENCE
    Systematic short prompts walking the symbol space: "aa. ab. ac..."
    Each prompt is a controlled geometric stimulus — known symbol
    combinations in isolation. Feeds real fold events and imprint
    updates without semantic noise from natural language.

  LAYER 3 — DELTA TRACKING
    Which basis matrix cells shifted after each prompt, by how much,
    and in which direction. The delta IS the signal — it shows which
    symbol relationships the system is actively resolving.

WHY THIS ORDER
──────────────
The basis matrix gives you the coordinate system.
The excitation sequence perturbs it in controlled ways.
The deltas show you which coordinates are load-bearing.
That's the same logic the model uses: geometry first,
then observe what moves, then interpret.
"""

import math
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional

from wave.symbolic_wave import SymbolicWave
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping
from core.clarity_ratio import clarity_ratio
from core.invariants import invariants
from utils.bipolar_lattice import bipolar_lattice

_SYMBOLS = ['0'] + [chr(ord('A') + i) for i in range(26)]  # 27 symbols
_N       = len(_SYMBOLS)


# ── Basis matrix ──────────────────────────────────────────────────────────────

def compute_basis_matrix() -> np.ndarray:
    """
    27×27 matrix of pair tensions.
    Entry [i][j] = pair_tension(_SYMBOLS[i], _SYMBOLS[j]).tension
    Diagonal = 0 (symbol with itself, no interaction).
    Matrix is NOT symmetric — order matters (s1→s2 vs s2→s1 differ
    by odd/even polarity of the leading symbol).
    """
    matrix = np.zeros((_N, _N), dtype=float)
    for i, s1 in enumerate(_SYMBOLS):
        for j, s2 in enumerate(_SYMBOLS):
            if i == j:
                continue
            pt = symbol_grouping.pair_tension(s1, s2)
            matrix[i][j] = pt["tension"]
    return matrix


def basis_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    """
    Compute what changed between two basis matrices.

    Returns:
      delta_matrix   — raw difference (after - before)
      changed_pairs  — list of (s1, s2, before, after, delta) sorted by |delta|
      max_delta      — largest absolute change
      mean_delta     — mean absolute change across all pairs
      active_symbols — symbols involved in the top-shifting pairs
    """
    delta   = after - before
    abs_d   = np.abs(delta)
    max_d   = float(np.max(abs_d))
    mean_d  = float(np.mean(abs_d))

    # Collect non-trivial changes
    threshold = mean_d * 0.5   # only report changes above half the mean
    changed   = []
    for i in range(_N):
        for j in range(_N):
            if i == j:
                continue
            d = float(delta[i][j])
            if abs(d) >= threshold:
                changed.append({
                    "s1":     _SYMBOLS[i],
                    "s2":     _SYMBOLS[j],
                    "before": round(float(before[i][j]), 4),
                    "after":  round(float(after[i][j]),  4),
                    "delta":  round(d, 4),
                })
    changed.sort(key=lambda x: abs(x["delta"]), reverse=True)

    active = set()
    for c in changed[:20]:
        active.add(c["s1"])
        active.add(c["s2"])

    return {
        "delta_matrix":    delta,
        "changed_pairs":   changed[:20],   # top 20 shifts
        "max_delta":       round(max_d,  4),
        "mean_delta":      round(mean_d, 4),
        "active_symbols":  sorted(active),
    }


# ── Excitation sequence generator ────────────────────────────────────────────

def generate_excitation_sequence(
    mode: str = "pairs",
    max_prompts: int = 54,
    target_symbols: Optional[List[str]] = None,
    chain_length: int = 2,
) -> List[str]:
    """
    Generate the controlled excitation prompt sequence.

    mode='pairs'    — 2-symbol prompts (chain_length=2)
    mode='triples'  — 3-symbol prompts (chain_length=3)
    mode='chain'    — N-symbol prompts where N = chain_length (4, 5, 6...)
                      Walks systematic combinations of chain_length symbols.
                      Each additional symbol is another radial stack layer —
                      the geometry compounds from each prior embedding.
    mode='targeted' — dense excitation focused on specific symbols.
    """
    prompts = []

    if mode == "pairs":
        count = 0
        for s1 in _SYMBOLS:
            for s2 in _SYMBOLS:
                if s1 == s2:
                    continue
                prompts.append(f"{s1}{s2}.")
                count += 1
                if count >= max_prompts:
                    return prompts

    elif mode == "triples":
        count = 0
        for s1 in _SYMBOLS:
            for s2 in _SYMBOLS:
                if s1 == s2:
                    continue
                for s3 in _SYMBOLS:
                    if s3 == s1 or s3 == s2:
                        continue
                    prompts.append(f"{s1}{s2}{s3}.")
                    count += 1
                    if count >= max_prompts:
                        return prompts

    elif mode == "chain":
        # Systematic chain of exactly chain_length symbols.
        # Uses a deterministic walk: start symbol sweeps A-Z,
        # each subsequent symbol steps forward by a prime offset
        # derived from its position — ensures coverage without
        # repeating the same symbol in a chain.
        # Primes used: 1, 2, 3, 5, 7 for positions 1-5
        prime_offsets = [1, 2, 3, 5, 7, 11, 13]
        active = [s for s in _SYMBOLS if s != '0']  # exclude regulator
        n      = len(active)
        count  = 0
        seen   = set()

        for start_idx in range(n):
            chain = []
            used  = set()
            valid = True

            for pos in range(chain_length):
                offset = prime_offsets[pos % len(prime_offsets)]
                idx    = (start_idx + sum(prime_offsets[:pos+1])) % n
                sym    = active[idx]
                if sym in used:
                    # Nudge forward to find unused symbol
                    for nudge in range(1, n):
                        candidate = active[(idx + nudge) % n]
                        if candidate not in used:
                            sym = candidate
                            break
                    else:
                        valid = False
                        break
                chain.append(sym)
                used.add(sym)

            if not valid:
                continue

            prompt = ''.join(chain) + '.'
            if prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
                count += 1
                if count >= max_prompts:
                    return prompts

        # If we haven't hit max_prompts, fill with random valid chains
        import random
        random.seed(42)  # deterministic fill
        while len(prompts) < max_prompts:
            chain = random.sample(active, min(chain_length, len(active)))
            prompt = ''.join(chain) + '.'
            if prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)

    elif mode == "targeted":
        # Resolve target symbols
        if target_symbols is None:
            groups  = symbol_grouping.get_group_summary()
            targets = [
                g["members"][0] for g in groups
                if g["size"] == 1
                and g["tension_centroid"] < 0.005
                and g["members"][0] != '0'
            ]
            if not targets:
                targets = ['N', 'O', 'P']
        else:
            targets = [s for s in target_symbols if s != '0']

        print(f"    Targeting: {targets}")

        seen = set()

        # Strategy 1 — target as s1 against every other symbol
        for t in targets:
            for s2 in _SYMBOLS:
                if s2 == t or s2 == '0':
                    continue
                p = f"{t}{s2}."
                if p not in seen:
                    prompts.append(p)
                    seen.add(p)

        # Strategy 2 — target as s2 against every other symbol
        for t in targets:
            for s1 in _SYMBOLS:
                if s1 == t or s1 == '0':
                    continue
                p = f"{s1}{t}."
                if p not in seen:
                    prompts.append(p)
                    seen.add(p)

        # Strategy 3 — boundary-adjacent pairings
        boundary_adjacent = ['M', 'L', 'K', 'N', 'O', 'P']
        for t in targets:
            for b in boundary_adjacent:
                if b == t:
                    continue
                for p in [f"{t}{b}.", f"{b}{t}."]:
                    if p not in seen:
                        prompts.append(p)
                        seen.add(p)

        # Strategy 4 — triples with imprinted symbols as context
        imprinted_groups = [
            g for g in symbol_grouping.get_group_summary()
            if g["tension_centroid"] >= 0.005 and g["members"][0] not in targets
        ]
        imprinted_syms = [g["members"][0] for g in imprinted_groups[:6]]
        for t in targets:
            for s1 in imprinted_syms:
                for s2 in imprinted_syms:
                    if s1 == s2:
                        continue
                    p = f"{s1}{t}{s2}."
                    if p not in seen:
                        prompts.append(p)
                        seen.add(p)

        return prompts[:max_prompts]

    return prompts


# ── Single prompt probe ───────────────────────────────────────────────────────

def probe_prompt(prompt: str) -> Dict[str, Any]:
    """
    Run one excitation prompt through the minimal geometric pipeline
    and return only the semantics-relevant metrics.

    Deliberately strips answer generation, memory etching, observer
    consensus — those add noise to a diagnostics pass. We want:
      - Symbol stream + zero breaks
      - Fold events fired (how many, which lattice indices, coupling)
      - Stream tension profile
      - Pair tensions for adjacent symbols
      - Group membership changes
    """
    sw       = SymbolicWave()
    tri_data = sw.triangulate(prompt)

    # Minimal wave — just enough to drive fold ticks
    prop        = WavePropagator()
    prop_result = prop.propagate(tri_data, steps=30)

    numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                    if isinstance(x, (int, float))]
    wave_amp = float(np.mean(np.abs(numeric_wave))) if numeric_wave else 0.1

    # Bipolar lattice — minimal cycle to keep tension live
    bipolar_lattice.react_to_wave(np.array(numeric_wave))
    bipolar_lattice.apply_tension_cycle(wave_amp)

    # Fold line ticks — this is the core of the diagnostic
    fold_tick_results = []
    for _ in range(4):
        ft = fold_line_resonance.tick(external_wave_amp=wave_amp)
        fold_tick_results.append(ft)

    total_fold_events = sum(r["fold_events_this_tick"] for r in fold_tick_results)
    fold_indices      = []
    fold_couplings    = []
    for r in fold_tick_results:
        for ev in r.get("fold_events", []):
            fold_indices.append(ev["lattice_idx"])
            fold_couplings.append(ev["coupling"])

    coherence = fold_line_resonance.get_coherence_signal()

    # Stream context
    symbol_stream = tri_data.get("symbol_stream", [])
    stream_ctx    = symbol_grouping.stream_context(symbol_stream)

    # Adjacent pair tensions — the direct semantic signal
    pair_tensions = []
    for i in range(len(symbol_stream) - 1):
        s1, s2 = symbol_stream[i], symbol_stream[i + 1]
        if s1 == '0' or s2 == '0':
            pair_tensions.append({"s1": s1, "s2": s2, "tension": 0.0,
                                   "relationship": "boundary"})
            continue
        pt = symbol_grouping.pair_tension(s1, s2)
        pair_tensions.append({
            "s1":           s1,
            "s2":           s2,
            "tension":      pt["tension"],
            "relationship": pt["relationship"],
            "same_group":   pt["same_group"],
        })

    # Group membership for each unique symbol in stream
    unique_syms  = list(dict.fromkeys(s for s in symbol_stream if s != '0'))
    group_report = {}
    for sym in unique_syms:
        grp = symbol_grouping.group_for(sym)
        if grp:
            group_report[sym] = {
                "group_id":    grp.group_id,
                "members":     grp.members,
                "polarity":    grp.dominant_polarity,
                "base_tension": round(grp.base_tension, 4),
            }

    return {
        "prompt":            prompt,
        "symbol_stream":     symbol_stream,
        "zero_breaks":       tri_data.get("zero_breaks", []),
        "n_symbols":         tri_data["n_original"],
        "fold_events":       total_fold_events,
        "fold_indices":      fold_indices,
        "mean_coupling":     round(float(np.mean(fold_couplings)), 4) if fold_couplings else 0.0,
        "coherence":         round(coherence, 4),
        "stream_tension":    stream_ctx["mean_tension"],
        "tension_profile":   stream_ctx["tension_profile"],
        "zero_boundaries":   stream_ctx["zero_boundaries"],
        "pair_tensions":     pair_tensions,
        "group_membership":  group_report,
        "wave_amp":          round(wave_amp, 4),
    }


# ── Main diagnostic runner ────────────────────────────────────────────────────

def run_semantic_diagnostic(
    mode:           str = "pairs",
    max_prompts:    int = 54,
    verbose:        bool = True,
    target_symbols: Optional[List[str]] = None,
    chain_length:   int = 2,
) -> Dict[str, Any]:
    """
    Full semantic diagnostic pass.

    1. Warm-up fold ticks to seed ground state
    2. Compute initial basis matrix
    3. Run excitation sequence
    4. After each prompt: update basis, compute delta, log key shifts
    5. Return full session record

    mode='targeted' focuses on specific symbols (or auto-detects
    unimprinted singletons) to push fold events into underexcited
    angle ranges.
    """
    print("\n" + "=" * 70)
    print("SEMANTIC DIAGNOSTIC — Symbol Basis + Excitation Sequence")
    print("=" * 70)
    print(f"Mode: {mode} | Max prompts: {max_prompts}")
    print(f"Fold tolerance    : {invariants.asymmetric_delta:.8f}")
    print(f"Initial coherence : {fold_line_resonance.get_coherence_signal():.4f}")

    # ── Warm-up: seed the fold line before measuring ground state ────────────
    # Run a short neutral excitation pass — just enough fold ticks to give
    # the lattice a non-zero imprint state. Without this, the initial basis
    # is all zeros because centroids are 0.0 before any folds have fired.
    # Using asymmetric_delta as the seed amplitude keeps it minimal and
    # derived — not an arbitrary warm-up value.
    print("\n[0] Warm-up (seeding fold line for non-zero ground state)...")
    seed_amp = invariants.asymmetric_delta
    warmup_folds = 0
    for _ in range(32):
        ft = fold_line_resonance.tick(external_wave_amp=seed_amp)
        warmup_folds += ft["fold_events_this_tick"]
    symbol_grouping._compute_groups()   # force group recompute after warm-up
    print(f"    Warm-up folds  : {warmup_folds}")
    print(f"    Imprinted pts  : {fold_line_resonance.get_status()['imprinted_points']}")
    print(f"    Coherence      : {fold_line_resonance.get_coherence_signal():.4f}")

    # ── Layer 1: initial basis matrix ────────────────────────────────────────
    print("\n[1] Computing initial basis matrix (27×27)...")
    basis_before = compute_basis_matrix()

    # Print basis summary — top non-zero tensions at start
    flat_tensions = [(float(basis_before[i][j]), _SYMBOLS[i], _SYMBOLS[j])
                     for i in range(_N) for j in range(_N) if i != j]
    flat_tensions.sort(key=lambda x: abs(x[0]), reverse=True)

    print(f"    Basis range    : [{float(basis_before.min()):.4f},"
          f" {float(basis_before.max()):.4f}]")
    print(f"    Mean |tension| : {float(np.mean(np.abs(basis_before))):.4f}")
    print(f"    Non-zero pairs : "
          f"{int(np.count_nonzero(basis_before))} / {_N*(_N-1)}")

    if verbose and flat_tensions[0][0] != 0.0:
        print("    Top 5 tensions :")
        for t, s1, s2 in flat_tensions[:5]:
            print(f"      {s1}→{s2} : {t:+.4f}")

    # ── Layer 2: excitation sequence ─────────────────────────────────────────
    prompts  = generate_excitation_sequence(mode=mode, max_prompts=max_prompts,
                                             target_symbols=target_symbols,
                                             chain_length=chain_length)
    print(f"\n[2] Running {len(prompts)} excitation prompts...")

    session_log  = []
    basis_current = basis_before.copy()
    cumulative_active_symbols = set()

    for p_idx, prompt in enumerate(prompts):
        result       = probe_prompt(prompt)
        basis_after  = compute_basis_matrix()
        delta        = basis_delta(basis_current, basis_after)
        basis_current = basis_after

        cumulative_active_symbols.update(delta["active_symbols"])

        session_log.append({
            "prompt_idx": p_idx,
            "probe":      result,
            "delta":      {
                "max_delta":       delta["max_delta"],
                "mean_delta":      delta["mean_delta"],
                "active_symbols":  delta["active_symbols"],
                "changed_pairs":   delta["changed_pairs"][:5],  # top 5 per prompt
            },
        })

        # ── Per-prompt output ─────────────────────────────────────────────────
        if verbose:
            has_change = delta["max_delta"] > 0.001
            fold_str   = f"folds={result['fold_events']} coup={result['mean_coupling']:.3f}"
            tens_str   = f"tension={result['stream_tension']:+.4f}"
            coh_str    = f"coh={result['coherence']:.3f}"
            delta_str  = (f"Δmax={delta['max_delta']:.4f} "
                          f"active={delta['active_symbols']}"
                          if has_change else "Δ=0 (no imprint shift)")

            print(f"  [{p_idx+1:3d}] {prompt:12s} | "
                  f"{fold_str} | {tens_str} | {coh_str} | {delta_str}")

    # ── Layer 3: session summary ──────────────────────────────────────────────
    final_basis  = compute_basis_matrix()
    session_delta = basis_delta(basis_before, final_basis)

    print(f"\n[3] Session delta (initial → final basis):")
    print(f"    Max shift      : {session_delta['max_delta']:.4f}")
    print(f"    Mean shift     : {session_delta['mean_delta']:.4f}")
    print(f"    Active symbols : {session_delta['active_symbols']}")
    print(f"\n    Top shifted pairs:")
    for cp in session_delta["changed_pairs"][:10]:
        direction = "▲" if cp["delta"] > 0 else "▼"
        print(f"      {cp['s1']}→{cp['s2']} : "
              f"{cp['before']:+.4f} → {cp['after']:+.4f} "
              f"({direction}{abs(cp['delta']):.4f})")

    print(f"\n    Cumulative active symbols across all prompts:")
    print(f"      {sorted(cumulative_active_symbols)}")
    print(f"\n    Final coherence : {fold_line_resonance.get_coherence_signal():.4f}")
    print(f"    Imprinted pts   : {fold_line_resonance.get_status()['imprinted_points']}")

    sg_status = symbol_grouping.get_status()
    print(f"    Groups          : {sg_status['total_groups']}"
          f" (imprinted: {sg_status['imprinted_groups']})")

    # ── Group membership breakdown ────────────────────────────────────────────
    print(f"\n[4] Group membership (imprinted groups only):")
    groups = symbol_grouping.get_group_summary()
    imprinted_groups = [g for g in groups if g["tension_centroid"] >= 0.005]

    if not imprinted_groups:
        print("    No imprinted groups yet.")
    else:
        # Sort by tension_centroid descending — strongest first
        imprinted_groups.sort(key=lambda g: abs(g["tension_centroid"]), reverse=True)
        for g in imprinted_groups:
            polarity_marker = "↑" if g["dominant_polarity"] == "vertical" else \
                              "↔" if g["dominant_polarity"] == "horizontal" else "~"
            print(f"    Group {g['group_id']:2d} {polarity_marker} | "
                  f"members={g['members']} | "
                  f"size={g['size']} | "
                  f"centroid={g['tension_centroid']:+.4f} | "
                  f"base_tension={g['base_tension']:+.4f} | "
                  f"odd={g['odd_count']} even={g['even_count']}")

    # ── Lattice collision report ──────────────────────────────────────────────
    collisions = symbol_grouping.get_collisions()
    if collisions:
        print(f"\n[5] Lattice collisions (symbols sharing same index):")
        for c in collisions:
            print(f"    idx={c['lattice_idx']:3d} → {c['symbols']}")
        print(f"    Note: 0/N collision means the phase separator and N")
        print(f"    share a coordinate — watch these symbols for coupled behaviour.")

    # ── Singleton report — symbols not yet grouped with anything ─────────────
    singletons = [g for g in groups
                  if g["size"] == 1 and g["tension_centroid"] < 0.005]
    if singletons:
        singleton_names = [g["members"][0] for g in singletons]
        print(f"\n    Unimprinted singletons ({len(singletons)}): {singleton_names}")
        print(f"    These symbols have not yet been touched by fold events.")
        print(f"    Run more prompts or triples mode to excite them.")

    return {
        "mode":                     mode,
        "prompts_run":              len(prompts),
        "session_log":              session_log,
        "initial_basis":            basis_before,
        "final_basis":              final_basis,
        "session_delta":            session_delta,
        "cumulative_active_symbols": sorted(cumulative_active_symbols),
        "final_coherence":          fold_line_resonance.get_coherence_signal(),
    }
