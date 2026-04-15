"""
visualizer.py — GeometricClarityLab Field Visualizer
=====================================================
Four-panel live view of the geometric field state after each pipeline run.
Non-blocking — pipeline loop continues while window stays open.

Panels:
  1. Waveform — propagated wave with zero-break pocket markers
  2. Bipolar ring — 53 waypoints at actual radial positions,
                    colored by role, core highlighted, spin arrows on structural
  3. Field metrics — signed bar chart of key scalars this turn
  4. Session history — consensus + pocket confidence across all turns

Modular: each panel is a standalone function.
Adding a new panel = one function + one ax reference.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # non-blocking on Windows; falls back gracefully
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, List, Optional

# ── Color palette — consistent across panels ──────────────────────────────────
_COLORS = {
    "positive":   "#4fc3f7",   # light blue
    "negative":   "#ef9a9a",   # light red
    "structural": "#fff176",   # yellow — golden zone
    "stabilizer": "#a5d6a7",   # green — pressure relief
    "symbol":     "#b0bec5",   # grey — symbol ring
    "core":       "#ff6f00",   # amber — elected core
    "waveform":   "#80cbc4",   # teal
    "pocket":     "#ff8a65",   # orange — zero-break markers
    "consensus":  "#81c784",   # green for positive consensus
    "consensus_neg": "#e57373",# red for negative consensus
    "persistence":"#64b5f6",   # blue
    "pocket_conf":"#ff8a65",   # orange
    "field_stress":"#ce93d8",  # purple
    "fold_neg":   "#80deea",   # cyan
    "bg":         "#1a1a2e",   # dark background
    "fg":         "#e0e0e0",   # foreground text
    "grid":       "#2d2d4e",   # subtle grid
}

_FIG  = None
_AXS  = None
_HIST: List[Dict] = []   # session history across turns


def _ensure_figure():
    global _FIG, _AXS
    if _FIG is not None and plt.fignum_exists(_FIG.number):
        return
    _FIG = plt.figure(figsize=(16, 9), facecolor=_COLORS["bg"])
    _FIG.canvas.manager.set_window_title("GeometricClarityLab — Field Visualizer")
    gs   = GridSpec(2, 2, figure=_FIG, hspace=0.38, wspace=0.32,
                    left=0.06, right=0.97, top=0.93, bottom=0.07)
    _AXS = {
        "wave":    _FIG.add_subplot(gs[0, 0]),
        "ring":    _FIG.add_subplot(gs[0, 1]),
        "metrics": _FIG.add_subplot(gs[1, 0]),
        "history": _FIG.add_subplot(gs[1, 1]),
    }
    for ax in _AXS.values():
        ax.set_facecolor(_COLORS["bg"])
        ax.tick_params(colors=_COLORS["fg"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_COLORS["grid"])


# ── Panel 1: Waveform ─────────────────────────────────────────────────────────

def _draw_waveform(ax, prop_result: Dict, tri_data: Dict):
    ax.cla()
    ax.set_facecolor(_COLORS["bg"])

    raw      = prop_result.get("waveform_full",
               prop_result.get("waveform_sample", []))
    numeric  = [x for x in raw if isinstance(x, (int, float))]
    if not numeric:
        ax.text(0.5, 0.5, "No waveform", color=_COLORS["fg"],
                ha="center", va="center", transform=ax.transAxes)
        return

    wave        = np.array(numeric)
    xs          = np.linspace(0, 1, len(wave))
    zero_breaks = tri_data.get("zero_breaks", [])
    n_symbols   = tri_data.get("n_original", max(zero_breaks) + 1 if zero_breaks else 1)
    steps       = len(wave)

    ax.plot(xs, wave, color=_COLORS["waveform"], linewidth=1.2, alpha=0.9)
    ax.axhline(0, color=_COLORS["grid"], linewidth=0.5, linestyle="--")

    for br in zero_breaks:
        xpos = (br / max(n_symbols, 1)) * (steps / steps)  # normalised
        xpos = br / max(n_symbols, 1)
        ax.axvline(xpos, color=_COLORS["pocket"], linewidth=1.0,
                   linestyle=":", alpha=0.8, label="zero-break" if br == zero_breaks[0] else "")

    mode = prop_result.get("mode", "direct")
    ax.set_title(f"Waveform  [{mode}]", color=_COLORS["fg"], fontsize=9, pad=4)
    ax.set_xlabel("Symbol stream position", color=_COLORS["fg"], fontsize=7)
    ax.set_ylabel("Amplitude", color=_COLORS["fg"], fontsize=7)
    ax.tick_params(colors=_COLORS["fg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_COLORS["grid"])
    if zero_breaks:
        ax.legend(fontsize=7, facecolor=_COLORS["bg"],
                  labelcolor=_COLORS["fg"], framealpha=0.5)


# ── Panel 2: Bipolar ring ─────────────────────────────────────────────────────

def _draw_ring(ax, bipolar_status: Dict, waypoints_snapshot: Optional[List] = None):
    ax.cla()
    ax.set_facecolor(_COLORS["bg"])
    ax.set_aspect("equal")

    if waypoints_snapshot is None:
        ax.text(0.5, 0.5, "No waypoint data", color=_COLORS["fg"],
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Bipolar Ring", color=_COLORS["fg"], fontsize=9, pad=4)
        return

    core_id = bipolar_status.get("core_id")

    for wp in waypoints_snapshot:
        role   = wp["role"]
        angle  = wp["angle"]
        radius = wp["radius"]
        wp_id  = wp["wp_id"]
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        is_core = (wp_id == core_id)
        color   = _COLORS["core"] if is_core else _COLORS.get(role, _COLORS["symbol"])
        size    = 120 if is_core else (60 if role == "structural" else 30)
        zorder  = 5 if is_core else 3

        ax.scatter(x, y, c=color, s=size, zorder=zorder, alpha=0.9,
                   edgecolors="#ffffff" if is_core else "none", linewidths=1.0)

        # Spin arrow on structural waypoints
        if role == "structural" and "spin_phase" in wp:
            sp    = wp["spin_phase"]
            dx    = 0.06 * math.cos(sp)
            dy    = 0.06 * math.sin(sp)
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=_COLORS["fg"],
                                        lw=0.7), zorder=4)

    # Legend patches
    patches = [
        mpatches.Patch(color=_COLORS["positive"],   label="positive"),
        mpatches.Patch(color=_COLORS["negative"],   label="negative"),
        mpatches.Patch(color=_COLORS["structural"], label="structural"),
        mpatches.Patch(color=_COLORS["stabilizer"], label="stabilizer"),
        mpatches.Patch(color=_COLORS["symbol"],     label="symbol"),
        mpatches.Patch(color=_COLORS["core"],       label="core"),
    ]
    ax.legend(handles=patches, fontsize=6, facecolor=_COLORS["bg"],
              labelcolor=_COLORS["fg"], framealpha=0.4,
              loc="lower right", ncol=2)

    spin  = bipolar_status.get("ring_spin_signal", 0.0)
    phase = bipolar_status.get("ring_net_phase", 0.0)
    ticks = bipolar_status.get("geo_tick_count", 0)
    ax.set_title(
        f"Bipolar Ring  |  spin {spin:+.3f}  phase {phase:.3f}  ticks {ticks}",
        color=_COLORS["fg"], fontsize=9, pad=4
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")


# ── Panel 3: Field metrics ────────────────────────────────────────────────────

def _draw_metrics(ax, metrics: Dict):
    ax.cla()
    ax.set_facecolor(_COLORS["bg"])

    labels = [
        "consensus",
        "persistence",
        "pocket_conf",
        "field_stress",
        "fold_neg_sig",
        "clarity",
        "convergence",
    ]
    values = [
        metrics.get("consensus",       0.0),
        metrics.get("persistence",     0.0),
        metrics.get("pocket_conf",     0.0),
        metrics.get("field_stress",    0.0),
        metrics.get("fold_neg_signal", 0.0),
        metrics.get("clarity",         0.0),
        metrics.get("convergence",     0.0),
    ]
    colors = []
    for i, (l, v) in enumerate(zip(labels, values)):
        if l == "consensus":
            colors.append(_COLORS["consensus"] if v >= 0 else _COLORS["consensus_neg"])
        elif l == "persistence":
            colors.append(_COLORS["persistence"])
        elif l == "pocket_conf":
            colors.append(_COLORS["pocket_conf"])
        elif l == "field_stress":
            colors.append(_COLORS["field_stress"])
        elif l == "fold_neg_sig":
            colors.append(_COLORS["fold_neg"])
        else:
            colors.append(_COLORS["waveform"])

    ys = range(len(labels))
    bars = ax.barh(list(ys), values, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, color=_COLORS["fg"], fontsize=8)
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color=_COLORS["fg"], linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Field Metrics — this turn", color=_COLORS["fg"], fontsize=9, pad=4)
    ax.set_xlabel("Value", color=_COLORS["fg"], fontsize=7)
    ax.tick_params(colors=_COLORS["fg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_COLORS["grid"])

    for bar, val in zip(bars, values):
        ax.text(
            max(val, 0) + 0.02 if val >= 0 else val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color=_COLORS["fg"],
            fontsize=7,
        )


# ── Panel 4: Session history ──────────────────────────────────────────────────

def _draw_history(ax, history: List[Dict]):
    ax.cla()
    ax.set_facecolor(_COLORS["bg"])

    if not history:
        ax.text(0.5, 0.5, "No history yet", color=_COLORS["fg"],
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Session History", color=_COLORS["fg"], fontsize=9, pad=4)
        return

    turns       = list(range(1, len(history) + 1))
    consensus   = [h.get("consensus",   0.0) for h in history]
    pocket_conf = [h.get("pocket_conf", 0.0) for h in history]
    core_score  = [h.get("core_score",  0.0) for h in history]

    ax.plot(turns, consensus,   color=_COLORS["consensus"],  linewidth=1.5,
            marker="o", markersize=4, label="consensus")
    ax.plot(turns, pocket_conf, color=_COLORS["pocket_conf"], linewidth=1.5,
            marker="s", markersize=4, label="pocket conf")
    ax.plot(turns, core_score,  color=_COLORS["core"],        linewidth=1.0,
            marker="^", markersize=3, label="core score", alpha=0.7)

    ax.axhline(0, color=_COLORS["grid"], linewidth=0.5, linestyle="--")
    ax.set_title("Session History", color=_COLORS["fg"], fontsize=9, pad=4)
    ax.set_xlabel("Turn", color=_COLORS["fg"], fontsize=7)
    ax.set_ylabel("Value", color=_COLORS["fg"], fontsize=7)
    ax.tick_params(colors=_COLORS["fg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_COLORS["grid"])
    ax.legend(fontsize=7, facecolor=_COLORS["bg"],
              labelcolor=_COLORS["fg"], framealpha=0.5)
    ax.set_ylim(-1.1, max(2.0, max(core_score) * 1.1))


# ── Public entry point ────────────────────────────────────────────────────────

def update(
    prop_result:       Dict[str, Any],
    tri_data:          Dict[str, Any],
    bipolar_status:    Dict[str, Any],
    consensus:         float,
    pocket_conf:       float,
    waypoints_snapshot: Optional[List] = None,
):
    """
    Call this after every pipeline run.
    All panels redraw from current state. Non-blocking.

    waypoints_snapshot: list of dicts with keys
      {wp_id, role, angle, radius, spin_phase (optional)}
    Pass None to skip ring drawing (ring panel shows placeholder).
    """
    global _HIST

    _ensure_figure()

    # Accumulate session history
    _HIST.append({
        "consensus":   consensus,
        "pocket_conf": pocket_conf,
        "persistence": prop_result.get("persistence", 0.0),
        "core_score":  bipolar_status.get("core_score", 0.0),
    })

    metrics = {
        "consensus":       consensus,
        "persistence":     prop_result.get("persistence", 0.0),
        "pocket_conf":     pocket_conf,
        "field_stress":    bipolar_status.get("field_stress", 0.0),
        "fold_neg_signal": bipolar_status.get("fold_negotiation_signal", 0.0),
        "clarity":         prop_result.get("clarity_ratio_score", 0.0),
        "convergence":     min(bipolar_status.get("core_score", 0.0), 1.0),
    }

    _draw_waveform(_AXS["wave"],    prop_result, tri_data)
    _draw_ring    (_AXS["ring"],    bipolar_status, waypoints_snapshot)
    _draw_metrics (_AXS["metrics"], metrics)
    _draw_history (_AXS["history"], _HIST)

    _FIG.suptitle(
        f"GeometricClarityLab  |  turn {len(_HIST)}  |  "
        f"core {bipolar_status.get('core_id', 'None')}  |  "
        f"mode {bipolar_status.get('mode', '')}",
        color=_COLORS["fg"], fontsize=10, y=0.98
    )

    plt.pause(0.001)   # non-blocking redraw
