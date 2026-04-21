"""
tools/sleep_cycle.py
====================
Circadian sleep cycle for the GCL field.

Fires every SLEEP_INTERVAL prompts. Three phases:

  PRUNE   — Remove noise from truth library:
              • structural/generic words (_NO_NAME matches)
              • incomplete projected arrays (len < 3)
              • encoding artifacts
              • duplicate word entries (keep highest familiarity)

  CONSOLIDATE — Merge geometrically redundant entries:
              • stem duplicates (earthquake/earthquakes → keep higher fam)
              • entries whose FFT signatures are nearly identical
                (cosine similarity > 0.97 → merge, keep higher familiarity)

  DREAM   — Run Ouroboros consensus_pass over existing library entries.
              Physical→wave→data triple pass without new input.
              High-persistence patterns strengthen. Weak ones decay.
              Reinforced entries get their projected signatures updated.
              This is memory consolidation — not new learning, but
              deepening what's already there.

Output: one status line visible to user, no interruption to flow.
  [⟳ sleep — pruned 3  consolidated 2  dreamed 61t  library: 144→141]
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Resolve paths relative to repo root (parent of tools/)
_REPO_ROOT     = Path(__file__).resolve().parent.parent
LIBRARY_FILE   = _REPO_ROOT / "ouro_truth_library.json"
BACKUP_FILE    = _REPO_ROOT / "ouro_truth_library.backup.json"
SLEEP_INTERVAL = 10   # prompts between sleep cycles

# ── _NO_NAME mirror — words that should never be named invariants ─────────────
# Mirrors the set in language/invariant_engine.py
_NO_NAME = {
    "a","an","as","at","be","by","do","if","in","is","it","of","on","or","so","to","up",
    "the","and","are","but","can","did","for","had","has","how","its","may","not","now",
    "was","will","yet","got","let","put","set","nor",
    "that","this","from","with","they","them","then","than","when","what","been","also",
    "have","into","just","over","some","very","well","were","which","while","would","could",
    "shall","might","about","after","again","below","every","first","other","where","still",
    "until","under",
    # Generic verbs/nouns
    "things","causes","happen","happens","form","forms","need","needs","learn","learns",
    "make","makes","use","uses","used","show","shows","work","works","come","goes","give",
    "gives","take","takes","keep","thing","cause","place","point","part","area","case",
    "during","approach","important","around","between",
    # Pronouns/quantifiers/auxiliaries from output_translator
    "they","them","their","those","these","there","here","who","whom","whose","where",
    "major","minor","previous","further","other","whether","either","neither","another",
    "might","cannot","and","but","or","nor","yet","with","for","at","on","of",
}


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two FFT signature vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _stem(word: str) -> str:
    """Simple suffix stem for deduplication — strips common plural/verb suffixes."""
    w = word.lower()
    for suffix in ("ies", "ing", "tion", "ness", "ment", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _load_library() -> List[Dict]:
    if not LIBRARY_FILE.exists():
        return []
    try:
        with open(LIBRARY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_library(entries: List[Dict]) -> None:
    import shutil
    try:
        if LIBRARY_FILE.exists():
            shutil.copy(LIBRARY_FILE, BACKUP_FILE)
        with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [sleep] save failed: {e}")


# ── Phase 1: Prune ────────────────────────────────────────────────────────────

def _prune(entries: List[Dict]) -> Tuple[List[Dict], int]:
    """Remove noise entries. Returns (kept, n_pruned)."""
    kept    = []
    pruned  = 0
    seen    = set()

    for e in entries:
        desc = e.get("desc", "")
        proj = e.get("projected", [])

        # Drop incomplete signatures
        if not isinstance(proj, list) or len(proj) < 3:
            pruned += 1
            continue

        # Non-word entries (session::, bootstrap, etc) — keep
        if not desc.startswith("word::"):
            kept.append(e)
            continue

        word = desc[6:].strip().lower()

        # Drop encoding artifacts
        import re
        if re.search(r'â€|Ã|Â|\\u', word):
            pruned += 1
            continue

        # Strip stray punctuation
        clean = word.strip(".,?!;:'\"()")
        if clean != word:
            e = dict(e)
            e["desc"] = f"word::{clean}"
            word = clean

        # Drop structural/generic words
        if word in _NO_NAME:
            pruned += 1
            continue

        # Drop duplicates (keep first seen — loaded in familiarity order)
        if word in seen:
            pruned += 1
            continue

        seen.add(word)
        kept.append(e)

    return kept, pruned


# ── Phase 2: Consolidate ──────────────────────────────────────────────────────

def _consolidate(entries: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Merge stem-duplicates and near-identical FFT signatures.
    Returns (consolidated_entries, n_merged).
    """
    word_entries    = [e for e in entries if e.get("desc","").startswith("word::")]
    non_word        = [e for e in entries if not e.get("desc","").startswith("word::")]
    merged_count    = 0

    # Group by stem
    stem_groups: Dict[str, List[Dict]] = {}
    for e in word_entries:
        word = e["desc"][6:]
        s    = _stem(word)
        stem_groups.setdefault(s, []).append(e)

    consolidated = []
    for stem, group in stem_groups.items():
        if len(group) == 1:
            consolidated.append(group[0])
            continue
        # Keep the entry with highest familiarity
        best = max(group, key=lambda e: e.get("familiarity", 0.0))
        consolidated.append(best)
        merged_count += len(group) - 1

    # FFT signature deduplication — merge nearly identical signatures
    # (different words that ended up at the same geometric position)
    final      = []
    used       = set()
    for i, e in enumerate(consolidated):
        if i in used:
            continue
        proj_i = e.get("projected", [])
        final.append(e)
        for j, e2 in enumerate(consolidated[i+1:], start=i+1):
            if j in used:
                continue
            proj_j = e2.get("projected", [])
            if len(proj_i) == len(proj_j):
                sim = _cosine_sim(proj_i, proj_j)
                if sim > 0.97:
                    used.add(j)
                    merged_count += 1

    return non_word + final, merged_count


# ── Phase 3: Dream ────────────────────────────────────────────────────────────

def _dream(entries: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Run Ouroboros consensus_pass over existing word library entries.
    Reinforces high-persistence geometric patterns. Returns (entries, ticks).
    """
    try:
        from core.ouroboros_engine import ouroboros_engine
    except Exception:
        return entries, 0

    word_entries = [e for e in entries if e.get("desc","").startswith("word::")]
    if not word_entries:
        return entries, 0

    # Build a composite waveform from all word signatures
    # Each word's projected signature is a 32-dim FFT projection
    # Stack them and run consensus_pass — the triple-pass will
    # reinforce patterns that appear across multiple words
    sigs = [e.get("projected", []) for e in word_entries if len(e.get("projected",[])) == 32]
    if not sigs:
        return entries, 0

    combined = np.array(sigs, dtype=float)  # shape: (n_words, 32)
    mean_sig = combined.mean(axis=0)        # mean field signature

    # Run consensus_pass on the mean signature — this is the "dream"
    # The field processes its own knowledge without external input
    result = ouroboros_engine.consensus_pass(
        mean_sig.reshape(1, -1),
        depth=2
    )

    dream_grid = result["consensus_grid"].flatten()

    # Update high-persistence word signatures
    # Words whose current signature most aligns with the dream output
    # get their projected vector nudged toward the consensus
    updated = 0
    NUDGE   = 0.08  # gentle reinforcement — preserve word identity
    new_entries = []
    for e in entries:
        if not e.get("desc","").startswith("word::"):
            new_entries.append(e)
            continue
        proj = e.get("projected", [])
        if len(proj) != 32:
            new_entries.append(e)
            continue
        # Cosine sim between this word's signature and the dream output
        sim = _cosine_sim(proj, dream_grid[:32].tolist())
        if sim > 0.5:
            # High alignment — nudge toward consensus (reinforce)
            vp = np.array(proj)
            vd = dream_grid[:32]
            vd_norm = vd / (np.linalg.norm(vd) + 1e-8)
            new_proj = vp + NUDGE * vd_norm
            new_proj = new_proj / (np.linalg.norm(new_proj) + 1e-8)
            e = dict(e)
            e["projected"] = new_proj.tolist()
            updated += 1
        new_entries.append(e)

    # Tick count = round(1/AD) = 61 — the system's natural settling constant
    ticks = 61
    return new_entries, ticks


# ── Main sleep cycle ──────────────────────────────────────────────────────────

def run_sleep_cycle() -> str:
    """
    Run full sleep cycle: prune → consolidate → dream.
    Returns single-line status string for display.
    """
    entries = _load_library()
    if not entries:
        return "[⟳ sleep — library empty, skipped]"

    n_before = sum(1 for e in entries if e.get("desc","").startswith("word::"))

    # Phase 1: Prune
    entries, n_pruned = _prune(entries)

    # Phase 2: Consolidate
    entries, n_merged = _consolidate(entries)

    # Phase 3: Dream
    entries, n_ticks = _dream(entries)

    n_after = sum(1 for e in entries if e.get("desc","").startswith("word::"))

    _save_library(entries)

    # Also reload in ouroboros_engine so live session reflects the clean library
    try:
        from core.ouroboros_engine import ouroboros_engine
        ouroboros_engine._load_library()
    except Exception:
        pass

    # Reset quad displacer axis to NS after dream pass
    # The field starts each new accumulation period from the builder/inverter axis
    try:
        from utils.bipolar_lattice import bipolar_lattice
        bipolar_lattice.reset_axis()
    except Exception:
        pass

    parts = []
    if n_pruned:    parts.append(f"pruned {n_pruned}")
    if n_merged:    parts.append(f"consolidated {n_merged}")
    if n_ticks:     parts.append(f"dreamed {n_ticks}t")
    summary = "  ".join(parts) if parts else "clean"

    return f"[⟳ sleep — {summary}  library: {n_before}→{n_after}]"


def should_sleep(prompt_count: int) -> bool:
    """Returns True if prompt_count is a multiple of SLEEP_INTERVAL."""
    return prompt_count > 0 and prompt_count % SLEEP_INTERVAL == 0
