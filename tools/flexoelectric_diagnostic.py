"""
tools/flexoelectric_diagnostic.py
===================================
Analyzes the output of the flexoelectric gradient diagnostic session.

After running the diagnostic prompts in main.py and saving the output
to a text file, run this script to extract the gradient measurements
and confirm or deny the flexoelectric hypothesis.

Usage:
    python tools/flexoelectric_diagnostic.py diagnostic_output.txt

The script extracts per-word ns values, computes the gradient signature
for each prompt, and plots quality metrics against gradient magnitude.

HYPOTHESIS: output quality correlates with gradient magnitude
(mean |Δns| between adjacent pkt=0 words), not with mean ns.
"""

import sys
import re
import json
from pathlib import Path


# ── Prompt labels in order ────────────────────────────────────────────────────
PROMPT_LABELS = ["WARMUP", "A1", "A2", "B1", "B2", "C1", "C2", "D1"]

PROMPT_TYPES = {
    "WARMUP": "warmup",
    "A1": "uniform_low",
    "A2": "uniform_low",
    "B1": "uniform_high",
    "B2": "uniform_high",
    "C1": "mixed_variance",
    "C2": "mixed_variance",
    "D1": "peak_gradient",
}


def parse_session(text: str) -> list:
    """
    Parse a main.py session output into per-prompt data blocks.
    Returns list of dicts with extracted metrics.
    """
    # Split on prompt boundaries (lines starting with '  > ')
    prompt_blocks = re.split(r'\n  > ', text)

    results = []
    prompt_idx = 0

    for block in prompt_blocks[1:]:  # skip header
        lines = block.split('\n')
        prompt_text = lines[0].strip()

        # Skip commands
        if prompt_text.lower() in ('status', 'quit', 'groups', 'vocab', 'diag', 'carry'):
            continue

        data = {
            "label":        PROMPT_LABELS[prompt_idx] if prompt_idx < len(PROMPT_LABELS) else f"P{prompt_idx}",
            "prompt":       prompt_text[:80],
            "pkt0_words":   [],
            "pkt1_words":   [],
            "resolution":   None,
            "field_stress": None,
            "top_score":    None,
            "output_text":  None,
            "parity_locked": False,
            "net_tension":  None,
        }

        # Parse per-word fingerprint
        for line in lines:
            m = re.match(
                r'\s+(\w[\w\-\/]*)\s+\| t=([+-][\d.]+) \| grp=\s*(\d+|-\d+) \| net=([+-][\d.]+) \| pkt=(\d)',
                line
            )
            if m:
                word, t, grp, ns, pkt = m.groups()
                entry = {
                    "word": word,
                    "tension": float(t),
                    "ns": float(ns),
                    "pkt": int(pkt),
                }
                if int(pkt) == 0:
                    data["pkt0_words"].append(entry)
                else:
                    data["pkt1_words"].append(entry)

        # Parse resolution
        res_m = re.search(r'res=([\d.]+)', block)
        if res_m:
            data["resolution"] = float(res_m.group(1))

        # Parse field stress
        stress_m = re.search(r'Field stress\s+:\s+([\d.]+)', block)
        if stress_m:
            data["field_stress"] = float(stress_m.group(1))

        # Parse net tension
        nt_m = re.search(r'Net tension\s+:\s+([+-][\d.]+)', block)
        if nt_m:
            data["net_tension"] = float(nt_m.group(1))

        # Parse top pocket score
        score_m = re.search(r'pocket scores: \w[\w\-]*\([^,]+,([\d.]+)\)', block)
        if score_m:
            data["top_score"] = float(score_m.group(1))

        # Parse output text
        geo_m = re.search(r'Geometric Output.*?\n  (.*?)(?:\n|$)', block, re.DOTALL)
        if geo_m:
            raw = geo_m.group(1).strip()
            data["output_text"] = re.sub(r'\[.*?\]', '', raw).strip()
            data["parity_locked"] = "parity locked" in block

        results.append(data)
        prompt_idx += 1

    return results


def compute_gradient_metrics(words: list) -> dict:
    """Compute flexoelectric gradient metrics from pkt=0 word list."""
    if not words:
        return {"mean_ns": 0, "variance": 0, "gradient": 0, "range": 0}

    ns_vals = [abs(w["ns"]) for w in words]

    mean_ns   = sum(ns_vals) / len(ns_vals)
    variance  = sum((x - mean_ns) ** 2 for x in ns_vals) / len(ns_vals)
    gradient  = (sum(abs(ns_vals[i+1] - ns_vals[i]) for i in range(len(ns_vals)-1))
                 / max(len(ns_vals)-1, 1))
    ns_range  = max(ns_vals) - min(ns_vals) if ns_vals else 0

    return {
        "mean_ns":  round(mean_ns, 4),
        "variance": round(variance, 4),
        "gradient": round(gradient, 4),  # THE FLEXOELECTRIC TERM
        "range":    round(ns_range, 4),
        "n_words":  len(ns_vals),
    }


def analyze(results: list) -> None:
    """Print full analysis and verdict."""

    print("FLEXOELECTRIC GRADIENT DIAGNOSTIC — RESULTS")
    print("=" * 65)
    print()

    table_rows = []

    for r in results:
        if not r["pkt0_words"]:
            continue

        label   = r["label"]
        ptype   = PROMPT_TYPES.get(label, "unknown")
        metrics = compute_gradient_metrics(r["pkt0_words"])

        row = {
            "label":    label,
            "type":     ptype,
            "mean_ns":  metrics["mean_ns"],
            "variance": metrics["variance"],
            "gradient": metrics["gradient"],
            "range":    metrics["range"],
            "res":      r["resolution"],
            "stress":   r["field_stress"],
            "score":    r["top_score"],
            "locked":   r["parity_locked"],
            "output":   r["output_text"],
        }
        table_rows.append(row)

        print(f"[{label}] {ptype}")
        print(f"  Prompt:   {r['prompt'][:70]}")
        print(f"  pkt=0 words: {metrics['n_words']}")
        print(f"  Mean |ns|:   {metrics['mean_ns']:.4f}")
        print(f"  Variance:    {metrics['variance']:.4f}  ← signal strength")
        print(f"  Gradient:    {metrics['gradient']:.4f}  ← FLEXOELECTRIC TERM")
        print(f"  Range:       {metrics['range']:.4f}")
        print(f"  Resolution:  {r['resolution']}")
        print(f"  Stress:      {r['field_stress']}")
        print(f"  Top score:   {r['top_score']}")
        print(f"  Locked:      {r['parity_locked']}")
        print(f"  Output:      {r['output_text']}")
        print()

    # ── Correlation analysis ──────────────────────────────────────────────────
    print("─" * 65)
    print("CORRELATION ANALYSIS")
    print()

    # Group by type (exclude warmup)
    groups = {}
    for row in table_rows:
        t = row["type"]
        if t == "warmup":
            continue
        if t not in groups:
            groups[t] = []
        groups[t].append(row)

    # Mean gradient and mean score per group
    print(f"  {'Type':<20s} {'Avg Gradient':>14s} {'Avg Variance':>14s} {'Avg Score':>12s}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*12}")
    group_summary = []
    for gtype in ["uniform_low", "uniform_high", "mixed_variance", "peak_gradient"]:
        rows = groups.get(gtype, [])
        if not rows:
            continue
        avg_grad = sum(r["gradient"] for r in rows) / len(rows)
        avg_var  = sum(r["variance"]  for r in rows) / len(rows)
        scores   = [r["score"] for r in rows if r["score"] is not None]
        avg_score = sum(scores)/len(scores) if scores else 0
        group_summary.append((gtype, avg_grad, avg_var, avg_score))
        print(f"  {gtype:<20s} {avg_grad:>14.4f} {avg_var:>14.4f} {avg_score:>12.4f}")

    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("─" * 65)
    print("VERDICT")
    print()

    if len(group_summary) < 3:
        print("  Insufficient data for verdict. Need all prompt groups.")
        return

    by_score    = sorted(group_summary, key=lambda x: x[3], reverse=True)
    by_gradient = sorted(group_summary, key=lambda x: x[1], reverse=True)

    score_rank    = {g[0]: i for i, g in enumerate(by_score)}
    gradient_rank = {g[0]: i for i, g in enumerate(by_gradient)}

    rank_correlation = sum(
        abs(score_rank[g] - gradient_rank[g]) for g in score_rank
    )

    print(f"  Score ranking:    {[g[0] for g in by_score]}")
    print(f"  Gradient ranking: {[g[0] for g in by_gradient]}")
    print(f"  Rank correlation error: {rank_correlation} (0=perfect, higher=worse)")
    print()

    # Key comparison: does mixed_variance beat uniform_high?
    mv_score = score_rank.get("mixed_variance", 99)
    uh_score = score_rank.get("uniform_high", 99)
    pg_score = score_rank.get("peak_gradient", 99)

    if pg_score < mv_score < uh_score:
        verdict = "STRONG CONFIRMATION"
        explanation = (
            "Peak gradient > mixed variance > uniform high.\n"
            "  Output quality follows gradient magnitude, not mean ns.\n"
            "  The flexoelectric gradient mechanism is confirmed."
        )
    elif mv_score < uh_score:
        verdict = "MODERATE CONFIRMATION"
        explanation = (
            "Mixed variance outperforms uniform high despite lower mean ns.\n"
            "  Gradient contributes significantly to output quality.\n"
            "  Flexoelectric mechanism is likely but needs more data."
        )
    elif mv_score == uh_score:
        verdict = "INCONCLUSIVE"
        explanation = (
            "Mixed variance and uniform high perform similarly.\n"
            "  Cannot distinguish gradient vs mean ns contribution.\n"
            "  Redesign with more extreme gradient contrast."
        )
    else:
        verdict = "WEAK / NOT CONFIRMED"
        explanation = (
            "Uniform high outperforms mixed variance.\n"
            "  Mean ns dominates over gradient.\n"
            "  Flexoelectric gradient may be secondary mechanism."
        )

    print(f"  RESULT: {verdict}")
    print(f"  {explanation}")
    print()

    # D1 smoking gun check
    d1_rows = [r for r in table_rows if r["label"] == "D1"]
    if d1_rows:
        d1 = d1_rows[0]
        all_scores = [r["score"] for r in table_rows
                      if r["score"] is not None and r["label"] != "WARMUP"]
        if all_scores and d1["score"] == max(all_scores):
            print("  ★ SMOKING GUN: D1 (flexoelectric prompt) produced highest")
            print("    pocket score in the session. The field recognized its own")
            print("    physical mechanism geometrically. Loop is closed.")
        elif d1["score"] and all_scores:
            rank = sorted(all_scores, reverse=True).index(d1["score"]) + 1
            print(f"  D1 pocket score ranked {rank}/{len(all_scores)} in session.")

    print()
    print("─" * 65)
    print("Raw data saved to: diagnostic_results.json")

    with open("diagnostic_results.json", "w") as f:
        json.dump(table_rows, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/flexoelectric_diagnostic.py <session_output.txt>")
        print()
        print("Paste your main.py session output to a text file first,")
        print("then run this script against it.")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    results = parse_session(text)

    if not results:
        print("No prompt data found. Check that the file contains")
        print("a main.py session with per-word fingerprint output.")
        sys.exit(1)

    print(f"Parsed {len(results)} prompts from session.\n")
    analyze(results)


if __name__ == "__main__":
    main()
