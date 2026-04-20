"""
GeometricClarityLab - Language Test Runner (option 7 only)
Includes inline fix verification at startup.
"""
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).parent.absolute()
sys.path.insert(0, str(root))

from core.invariants        import invariants
from core.ouroboros_engine  import ouroboros_engine
from core.safeguards        import safeguards
from utils.bipolar_lattice  import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping  import symbol_grouping
from utils.radial_displacer import radial_displacer
from language.processor     import language_processor
from language.geometric_output import geometric_output
from language.output_translator import translate_raw
from tools.sleep_cycle         import run_sleep_cycle, should_sleep
from utils.mobius_reader       import mobius_reader
from wave.generation           import generator
from core.field_state          import field_state_manager


def _verify_fixes():
    """
    Startup check — prints whether each fix is live so deployment
    issues are visible immediately before any test runs.
    """
    print("\n── Fix verification ─────────────────────────────────────────")

    # FIX 1: symbol_grouping centroid weight floor
    from utils.symbol_grouping import symbol_grouping as sg, symbol_to_signed
    import inspect, re
    src = inspect.getsource(sg.pair_tension)
    has_floor = "max(0.1," in src
    print(f"  FIX 1 tension floor  : {'LIVE' if has_floor else 'MISSING — symbol_grouping not updated'}")

    # FIX 2: relational_tension carry cap
    from language.relational_tension import relational_tension as rt
    src2 = inspect.getsource(rt.get_current_carry)
    has_cap = "np.clip" in src2 or "clip" in src2
    print(f"  FIX 2 carry cap      : {'LIVE' if has_cap else 'MISSING — relational_tension not updated'}")

    # FIX 3: soft polarity ceiling via tanh + scale=8.0
    import language.geometric_output as go_mod
    go_src = inspect.getsource(go_mod)
    has_scale = ("np.tanh" in go_src or "tanh" in go_src) and "8.0" in go_src
    print(f"  FIX 3 polarity scale : {'LIVE' if has_scale else 'MISSING — geometric_output not updated (v6)'}")

    # FIX 4: window minimum
    from language.geometric_output import geometric_output as go
    src4 = inspect.getsource(go._identify_target_region)
    has_win = ", 4, 8" in src4
    print(f"  FIX 4 window min=4   : {'LIVE' if has_win else 'MISSING — geometric_output not updated (v5)'}")
    print()


def main():
    print("\n" + "=" * 70)
    print("GeometricClarityLab — Language Processing")
    print("=" * 70)
    print(f"Ouroboros    : {ouroboros_engine.get_status()}")
    print(f"Bipolar      : {bipolar_lattice.get_status()}")
    print(f"Fold Line    : {fold_line_resonance.get_status()}")
    print(f"Sym Grouping : {symbol_grouping.get_status()}")

    _verify_fixes()

    # ── Restore field state from previous session ─────────────────────────────
    _saved_state = field_state_manager.load()
    if _saved_state:
        field_state_manager.apply_fold_line(fold_line_resonance, _saved_state,
                                            symbol_grouping=symbol_grouping)
        field_state_manager.apply_mobius(mobius_reader, _saved_state)
    else:
        print(f"  [field_state] No saved state — cold start")

    # Warm-up fold line if needed
    sg_status = symbol_grouping.get_status()
    _has_saved_groups = (_saved_state is not None
                         and len(_saved_state.get("symbol_groups", [])) >= 10)

    if sg_status["imprinted_groups"] == 0 and not _has_saved_groups:
        print("  [warm-up] Seeding fold line and symbol groups...")
        # Need >= 500 ticks to sweep full 2pi and activate all 27 symbol lattice indices.
        # The 512-point Fibonacci lattice has symbol indices spread across [0,512).
        # probe_prompt runs 108 ticks which activates 159 total lattice points but only
        # 12/27 symbol-specific indices (the positive side A-M). The negative side (N-Z)
        # maps to lattice indices only reachable after a full spin sweep (~383 ticks).
        # Without all symbols activated, _compute_groups produces 0 imprinted groups.
        # Force spin_sign = +1 for the seeding sweep.
        # At cold start resolution=0.15 < threshold=0.45 so tick() sets
        # spin_sign=-1, making phase decrement instead of advance.
        # Phase oscillates near 0 and only hits ~10 lattice points.
        # Forcing +1 lets the phase sweep forward through full 2π,
        # activating all 27 symbol lattice indices.
        fold_line_resonance.spin_sign = 1
        wave_amp = 0.3
        total_fold_events = 0
        for i in range(500):
            fold_line_resonance.spin_sign = 1   # hold positive throughout sweep
            result = fold_line_resonance.tick(external_wave_amp=wave_amp)
            total_fold_events += result.get("fold_events_this_tick", 0)
        # Force recompute after ticks
        symbol_grouping._compute_groups()
        after      = fold_line_resonance.get_status()["imprinted_points"]
        grps       = symbol_grouping.get_status()["imprinted_groups"]
        imp_sum    = float(fold_line_resonance.lattice_imprints.sum())
        print(f"  [warm-up] Done — fold_events={total_fold_events} "
              f"imp_sum={imp_sum:.2f} "
              f"imprinted_pts={after} | active_groups={grps}")
    else:
        grps = sg_status["imprinted_groups"]
        if _has_saved_groups and grps == 0:
            print(f"  [warm-up] Saved field state loaded — skipping warmup sweep.")
            print(f"  [field_state] {field_state_manager.summary()}")
        else:
            print(f"  [warm-up] Groups already active ({grps} groups) — skipping.")

    print("\n  Enter sentences. Commands: vocab | carry | status | groups | quit")
    print()

    _prev_conv_delta = None   # for Δ velocity tracking across prompts

    while True:
        try:
            sentence = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() in ("quit", "q", "exit"):
            print("Goodbye.")
            field_state_manager.save(
                fold_line      = fold_line_resonance,
                symbol_grouping= symbol_grouping,
                bipolar_lattice= bipolar_lattice,
                mobius_reader  = mobius_reader,
                processor      = language_processor,
            )
            break

        # ── Special commands ──────────────────────────────────────────────────

        if sentence.lower() == "carry":
            from language.relational_tension import relational_tension as rt
            s = rt.get_status()
            print(f"\n  Net carry: {s['net_carry']:+.4f} ({s['carry_direction']})")
            for e in s["window"]:
                print(f"    [age {e['age']}] {e['sentence']} | "
                      f"carry={e['carry_value']:+.4f}")
            print()
            continue

        if sentence.lower() == "vocab":
            stable = language_processor.get_vocabulary()
            print(f"\n  Stable vocabulary ({len(stable)} words):")
            for e in sorted(stable, key=lambda x: -x["appearances"]):
                print(f"    {e['word']:15s} app={e['appearances']} "
                      f"tension={e['mean_tension']:+.4f} "
                      f"grp={e['dominant_group']} "
                      f"net={e['net_signed']:+.3f}")
            print()
            continue

        if sentence.lower() == "status":
            s = language_processor.get_status()
            print("\n  Processor status:")
            for k, v in s.items():
                print(f"    {k}: {v}")
            print()
            continue

        if sentence.lower() == "groups":
            groups    = symbol_grouping.get_group_summary()
            imprinted = [g for g in groups if g["tension_centroid"] > 0.005]
            print(f"\n  Active groups ({len(imprinted)} of {len(groups)}):")
            for g in sorted(imprinted, key=lambda x: -abs(x["base_tension"]))[:10]:
                print(f"    grp{g['group_id']:2d} | members={g['members']} | "
                      f"tension={g['base_tension']:+.4f} | "
                      f"centroid={g['tension_centroid']:.4f}")
            print()
            continue

        if sentence.lower() == "diag":
            # Deep diagnostic: inspect fold_line internals directly
            imp = fold_line_resonance.lattice_imprints
            fl_status = fold_line_resonance.get_status()
            print(f"\n  ── Fold Line Diagnostics ────────────────────────────")
            print(f"  lattice_imprints array size : {len(imp)}")
            print(f"  _LATTICE_POINTS constant    : {fold_line_resonance.lattice_points}")
            print(f"  spin_phase                  : {fold_line_resonance.spin_phase:.6f}")
            print(f"  total_fold_events (history) : {len(fold_line_resonance.fold_events)}")
            print(f"  imprint_sum                 : {float(imp.sum()):.4f}")
            print(f"  imprint_max                 : {float(imp.max()):.4f}")
            print(f"  points > 0.005              : {int((imp > 0.005).sum())}")
            print(f"  points > 0.001              : {int((imp > 0.001).sum())}")
            print(f"  points > 0.0                : {int((imp > 0.0).sum())}")
            print()
            # Check symbol-specific indices
            print(f"  ── Symbol lattice indices ───────────────────────────")
            sym_indices = symbol_grouping._symbol_lattice_indices
            print(f"  Total symbols mapped: {len(sym_indices)}")
            print(f"  Index range: {min(sym_indices.values())} – {max(sym_indices.values())}")
            print(f"  Indices >= array size: "
                  f"{sum(1 for idx in sym_indices.values() if idx >= len(imp))}")
            active_sym = sum(1 for idx in sym_indices.values() if imp[idx] > 0.005)
            print(f"  Symbols with active imprint: {active_sym}/27")
            print()
            # Show each symbol
            import math
            for sym in sorted(sym_indices.keys()):
                lidx = sym_indices[sym]
                val  = float(imp[lidx]) if lidx < len(imp) else -1.0
                flag = "✓" if val > 0.005 else ("OOB" if lidx >= len(imp) else "✗")
                print(f"    {sym} → lattice[{lidx:4d}] imp={val:.4f} {flag}")
            print()
            # Test if lattice_imprints is writable
            test_idx = 0
            orig_val = float(fold_line_resonance.lattice_imprints[test_idx])
            try:
                fold_line_resonance.lattice_imprints[test_idx] = 0.99
                readback = float(fold_line_resonance.lattice_imprints[test_idx])
                fold_line_resonance.lattice_imprints[test_idx] = orig_val
                writable = (readback == 0.99)
            except Exception as e:
                writable = False
                readback = str(e)
            print(f"  lattice_imprints writable: {writable} (wrote 0.99, read {readback})")
            print(f"  lattice_imprints flags: {fold_line_resonance.lattice_imprints.flags}")
            print()
            # Run one tick and show what fires
            result = fold_line_resonance.tick(external_wave_amp=0.3)
            print(f"  Live tick: fold_events_this_tick={result['fold_events_this_tick']}")
            print(f"  Phase after tick: {fold_line_resonance.spin_phase:.6f}")
            # Check if any imprint changed
            max_imp = float(fold_line_resonance.lattice_imprints.max())
            n_nonzero = int((fold_line_resonance.lattice_imprints > 0).sum())
            print(f"  After tick: max_imp={max_imp:.6f}  nonzero_count={n_nonzero}")
            print()
            continue

        # ── Process sentence ──────────────────────────────────────────────────
        result = language_processor.process(sentence)
        fp     = result["fingerprint"]
        geo    = result.get("geo_output", {})

        # Show context priming indicator if question-only was detected
        if result.get("was_primed"):
            ctx = result.get("context_words", [])
            print(f"\n  ── Conversation priming ──────────────────────────────")
            print(f"  [question-only detected — context primed from {len(ctx)} window words]")
            print(f"  Context: {' '.join(ctx)}")

        # Fingerprint block
        print(f"\n  ── Fingerprint ──────────────────────────────────────")
        print(f"  Direction    : {fp['direction']}")
        print(f"  Mean tension : {fp['mean_tension']:+.4f}")
        print(f"  Net tension  : {fp['net_tension']:+.4f}")
        print(f"  Field stress : {fp['field_stress']:.4f}")
        print(f"  Boundaries   : {fp['boundary_count']}  (words: {fp['word_count']})")
        print(f"  Peak pair    : {fp['peak_pair'][0]}→{fp['peak_pair'][1]}"
              f"  ({fp['peak_tension']:+.4f})")
        if fp["top_groups"]:
            print(f"  Top groups   : " +
                  "  ".join(f"grp{g}×{c}" for g, c in fp["top_groups"]))

        # Per-word
        print(f"\n  ── Per-word ─────────────────────────────────────────")
        for wd in fp["per_word"]:
            pkt = wd.get("pocket", 0)
            print(f"  {wd['word']:15s} | t={wd['mean_tension']:+.4f} | "
                  f"grp={wd['dominant_group']:2d} | net={wd['net_signed']:+.3f} | "
                  f"pkt={pkt}")

        # Vocab hits
        if result["vocab_hits"]:
            print(f"\n  ── Vocabulary hits ──────────────────────────────────")
            for h in result["vocab_hits"]:
                print(f"  '{h['word']}'"
                      f"  fam={h['familiarity']:.3f}"
                      f"  app={h['appearances']}"
                      f"{' [STABLE]' if h['stable'] else ''}"
                      f"{' [NAMED]'  if h['named']  else ''}")

        # Answer
        print(f"\n  ── Answer ({result['gen_mode']}) ──────────────────────────")
        print(f"  {result['answer']}")

        # Exhaust recall — only show when it actually influenced the answer.
        # Suppressed when: local field resolved, geo output exists, or iterative
        # decode fired (all mean the system used its own geometry, not exhaust).
        er = result.get("exhaust_recall")
        answer_text = result.get("answer", "")
        guard_fired = (
            "Field resolved"          in answer_text or
            "Field partially resolved" in answer_text or
            "[decoded via geometric"   in answer_text or
            bool(geo)   # geo output populated = local field was sufficient
        )
        if er and not guard_fired:
            print(f"\n  ── Exhaust Recall ───────────────────────────────────")
            print(f"  source={er['source']}  dist={er['distance']:.4f}")
            print(f"  matched: '{er['prompt'][:70]}'")

        # Geo output
        if geo:
            locked = "⟳ parity locked" if geo.get("parity_locked") else "~ approximate"
            print(f"\n  ── Geometric Output ({locked}) ──────────────")
            _cap_str = "  [SVO capped]" if getattr(geometric_output, '_svo_capped', False) else ""
            _raw_out    = geometric_output.format_output(geo)
            _translated = translate_raw(
                _raw_out,
                fingerprint=result.get("fingerprint"),
            )
            print(f"  {_translated}{_cap_str}")
            tr = geo["target_region"]
            print(f"  [polarity {geo['field_polarity']:+.3f} | "
                  f"{tr['side']} [{tr['low']:.1f},{tr['high']:.1f}] | "
                  f"candidates: {', '.join(geo.get('candidates', [])[:4])}]")
            # Pocket scores
            ps = geo.get("pocket_scores", [])
            if ps:
                print(f"  pocket scores: " +
                      "  ".join(f"{p['word']}({p['pocket_label']},{p['score']:.3f})"
                                for p in ps))

        # Iterative decode — show when it fired
        if "[decoded via geometric iteration]" in result.get("answer", ""):
            print(f"\n  ── Iterative Decode ─────────────────────────────────")
            print(f"  Field re-processed its own geometry report.")
            print(f"  Output: {result['answer']}")

        if result.get("newly_named"):
            print(f"\n  ★ Named: {result['newly_named']}")

        # Carry + summary
        carry   = result.get("net_carry", 0.0)
        align   = result.get("carry_alignment", 0.0)
        inj     = result.get("carry_injected", 0.0)
        alabel  = "aligned" if align > 0.3 else ("opposing" if align < -0.3 else "neutral")
        res     = fold_line_resonance.get_resolution_score()
        print(f"\n  carry={carry:+.4f} | align={align:+.4f} ({alabel}) | inj={inj:+.4f}")
        print(f"  consensus={result['consensus']:+.4f}  "
              f"persist={result['persistence']:.4f}  "
              f"res={res:.3f}  "
              f"vocab={result['vocab_size']} ({result['vocab_stable']} stable, "
              f"{result['named_count']} named)  "
              f"t={result['elapsed']}s")

        # ── Pressure + field drive metrics ───────────────────────────────────
        mob_state = None   # initialize here — populated below in Möbius block
        if geo:
            p_mode    = geo.get("pressure_mode",  "")
            p_delta   = geo.get("pressure_delta",  0.0)
            g_actual  = geo.get("G_actual",         0.0)
            g_needed  = geo.get("G_needed",         0.0)
            p0        = geo.get("P0_current",       0.0)
            lv_before = geo.get("level_current",    0)
            lv_after  = geo.get("level_after",      0)
            flipped   = geo.get("domain_flipped",   False)
            if p_mode:
                flip_str = "  ⚡ DOMAIN FLIP" if flipped else ""
                _settle = result.get("settling_ticks", 0)
                _q_only = result.get("was_q_only", False)
                _settle_str = f"  ⟳{_settle}t" if _settle > 0 else ""
                _qonly_str  = " [Q]" if _q_only else ""
                print(f"  pressure={p_mode}  G={g_actual:.3f}/need {g_needed:.3f}"
                      f"  Δ={p_delta:+.3f}  P0={p0:.3f}"
                      f"  L{lv_before}→L{lv_after}{flip_str}{_settle_str}{_qonly_str}")

        # ── Field drive metrics (G×(1-res), T4×AD attractor, Δ velocity) ──────
        if mob_state and geo:
            _PHI        = 1.61803398
            _AD         = 0.016395102
            _face       = mob_state.get("face", "OUTER")
            _conv_delta = mob_state.get("convergence_delta", None)
            _g_actual   = geo.get("G_actual", 0.0)
            _res        = fold_line_resonance.get_resolution_score()

            # Metric 1: Field drive = G × (1 - res)
            # Predicts whether field has headroom to be driven toward φ
            _drive = round(_g_actual * (1.0 - _res), 4)

            # Metric 2: Predicted attractor = T4_raw × AD
            # CONFIRMED across sessions 9, 12, 14, 15:
            #   Convergence Δ = T4_angle × AD exactly
            # T4 is the warm-field polar phase of the Möbius oscillator.
            # The attractor is not content-dependent — it's set by the
            # current oscillator phase BEFORE the prompt is processed.
            # This means we can predict where the field will land BEFORE
            # seeing the convergence Δ output.
            # Predicted Δ = T4_polar × AD = convergence_gap exactly.
            # Use convergence_delta directly — it already equals T4_polar×AD.
            _pred_attr = mob_state.get("convergence_delta", None)

            # Near-φ override: G>2.0 AND drive>0.8 on OUTER face
            # These conditions allow the input to overwhelm the T4 attractor
            _near_phi = (_face.upper() == "OUTER"
                         and _g_actual >= 2.0
                         and _drive >= 0.8)

            # Predicted band around T4×AD attractor
            # Normal band: ±0.08 around predicted attractor
            # Near-φ override: 0.001-0.15
            if _near_phi:
                _pred_lo, _pred_hi = 0.001, 0.150
                _pred_label = "near-φ"
            elif _pred_attr is not None:
                _band = 0.08
                _pred_lo = max(0.0, round(_pred_attr - _band, 3))
                _pred_hi = round(_pred_attr + _band, 3)
                _pred_label = f"T4×AD"
            else:
                _pred_lo, _pred_hi = 0.0, 1.62
                _pred_label = "unknown"

            # Metric 3: Δ velocity (change from previous prompt)
            _delta_v_str = ""
            if _conv_delta is not None:
                if _prev_conv_delta is not None:
                    _dv = round(_conv_delta - _prev_conv_delta, 6)
                    _arrow = "→φ" if _dv < 0 else "←φ"
                    _delta_v_str = f"  Δv={_dv:+.6f} {_arrow}"
                _prev_conv_delta = _conv_delta

            # Deviation flag: actual Δ outside predicted band
            _dev_str = ""
            if _conv_delta is not None and _pred_attr is not None:
                if _conv_delta < _pred_lo or _conv_delta > _pred_hi:
                    _dev_str = "  ⚠ outside T4×AD band"

            # Build predicted attractor string
            _attr_str = (f"{_pred_label}[{_pred_lo:.3f}-{_pred_hi:.3f}]"
                         + (f"  T4×AD={_pred_attr:.6f}" if _pred_attr else ""))

            print(f"  drive={_drive:.4f}  predicted={_attr_str}"
                  f"{_delta_v_str}{_dev_str}")

        # ── Möbius surface state ──────────────────────────────────────────────
        mob_state = None
        try:
            mob_state = mobius_reader.read(
                fingerprint   = fp,
                fold_status   = fold_line_resonance.get_status(),
                group_summary = symbol_grouping.get_group_summary(),
                exhaust_sig   = bipolar_lattice.get_exhaust_signature(),
            )
            print()
            print(mobius_reader.format_state(mob_state))
        except Exception:
            pass  # non-fatal

        # ── Record exchange in conversation window ────────────────────────────
        try:
            # Only carry named invariants as conversation context.
            # Domain-specific candidates (e.g. "plants", "arch") should not
            # prime future unrelated prompts. Named invariants are
            # geometrically stable cross-domain words (appearances >= 2).
            _named_set = set(w.lower() for w in result.get("named_words", []))
            _all_cands = geo.get("candidates", [])[:8] if geo else []
            _stable_cands = [w for w in _all_cands if w.lower() in _named_set] or _all_cands[:2]
            field_state_manager.add_exchange(
                anchor_word = (geo.get("text", "") or "").split()[0] if geo else "",
                top_words   = _stable_cands,
                net_tension = fp.get("net_tension", 0.0),
                face        = mob_state.get("face", "unknown") if mob_state else "unknown",
                output      = geo.get("text", "") if geo else "",
                candidates  = _stable_cands,
            )
        except Exception:
            pass  # non-fatal

        # ── Sleep cycle check ────────────────────────────────────────────────
        try:
            # Use live session count + saved baseline for accurate total.
            # _get_total_prompts() reads from disk (only updated on quit),
            # so we add the in-session process_count to get the true running total.
            _saved_total   = field_state_manager._get_total_prompts()
            _session_count = result.get("process_count", 0)
            _prompt_count  = _saved_total + _session_count
            if should_sleep(_prompt_count):
                _sleep_status = run_sleep_cycle()
                print(f"  {_sleep_status}")
        except Exception:
            pass  # sleep cycle is never fatal

        print()


if __name__ == "__main__":
    main()
