"""
tools/clean_truth_library.py
============================
Cleans ouro_truth_library.json in-place.

Run from repo root:
    python tools/clean_truth_library.py           # apply
    python tools/clean_truth_library.py --dry-run  # preview only

What it fixes:
  1. Truncated session entries  (no terminal punctuation)
  2. Incomplete projected arrays (len < 3)
  3. Punctuation attached to word keys  'insects?' -> 'insects'
  4. Encoding artifacts             'eartha€™s' -> dropped
  5. Structural words that should never be named invariants
  6. Deduplicates word entries after all fixes applied

Creates ouro_truth_library.backup.json before writing.
"""

import json, re, sys, shutil
from pathlib import Path

LIBRARY_FILE = Path("ouro_truth_library.json")
BACKUP_FILE  = Path("ouro_truth_library.backup.json")

# Words that should never be named invariants — structural/function words
_STRUCTURAL_WORDS = {
    # Question words
    'how','why','what','where','when','which','who',
    # Auxiliaries
    'do','does','did','is','are','was','were','will','would',
    'can','could','should','may','might','must','shall',
    'have','has','had','be','been','being',
    # Articles / determiners
    'the','a','an','this','that','these','those',
    # Conjunctions / prepositions
    'and','or','but','so','if','in','on','at','to','of',
    'from','by','for','with','as','not','nor',
    # Pronouns
    'it','its','they','them','their','we','our','us',
    'he','she','his','her','i','me','my','you','your',
    # Common adverbs / short structure words
    'very','also','just','only','even','still','already',
    'always','never','often','usually','ever','truly',
    'like','such','some','any','each','one','two','more',
    'well','then','than','now','here','there',
    'while','though','although','because','since',
}


def _has_encoding_artifact(word: str) -> bool:
    return bool(re.search(r'â€|Ã|Â', word))


def clean(dry_run: bool = False) -> None:
    if not LIBRARY_FILE.exists():
        print(f"ERROR: {LIBRARY_FILE} not found — run from repo root")
        sys.exit(1)

    with open(LIBRARY_FILE, encoding="utf-8") as f:
        raw = f.read()

    try:
        entries = json.loads(raw)
        print(f"Loaded {len(entries)} entries")
    except json.JSONDecodeError as e:
        print(f"JSON error: {e} — attempting partial recovery...")
        objects = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
        entries = []
        for obj in objects:
            try:
                entries.append(json.loads(obj))
            except Exception:
                pass
        print(f"Recovered {len(entries)} entries")

    counters = {"truncated":0,"incomplete":0,"punct":0,
                "encoding":0,"structural":0,"dupes":0}
    kept       = []
    seen_words = set()
    changes    = []

    for e in entries:
        desc = e.get("desc", "")
        proj = e.get("projected", [])

        if not isinstance(proj, list) or len(proj) < 3:
            counters["incomplete"] += 1
            changes.append(("DROP incomplete", desc[:50], ""))
            continue

        if desc.startswith("session::"):
            text = desc[9:].rstrip()
            if not text.endswith(('.', '?', '!')):
                counters["truncated"] += 1
                changes.append(("DROP truncated", desc[:50], ""))
                continue
            kept.append(e)
            continue

        if desc.startswith("word::"):
            word = desc[6:]

            if _has_encoding_artifact(word):
                counters["encoding"] += 1
                changes.append(("DROP encoding", word, ""))
                continue

            clean_word = word.strip().rstrip('.,?!;:\'"').lstrip('\'"(')
            if clean_word != word:
                counters["punct"] += 1
                changes.append(("FIX punct", word, clean_word))
                e = dict(e)
                e["desc"] = f"word::{clean_word}"
                word = clean_word

            if word.lower() in _STRUCTURAL_WORDS:
                counters["structural"] += 1
                changes.append(("DROP structural", word, ""))
                continue

            if word.lower() in seen_words:
                counters["dupes"] += 1
                continue

            seen_words.add(word.lower())
            kept.append(e)
            continue

        kept.append(e)

    # Report
    print()
    if changes:
        print("Changes:")
        for kind, before, after in changes:
            if after:
                print(f"  {kind:20s} {before!r:30s} -> {after!r}")
            else:
                print(f"  {kind:20s} {before!r}")
        print()

    word_entries = [e for e in kept if e.get("desc","").startswith("word::")]
    word_list    = sorted(e["desc"][6:] for e in word_entries)

    print(f"Results:")
    print(f"  Original entries  : {len(entries)}")
    print(f"  Truncated removed : {counters['truncated']}")
    print(f"  Incomplete removed: {counters['incomplete']}")
    print(f"  Punct fixed       : {counters['punct']}")
    print(f"  Encoding dropped  : {counters['encoding']}")
    print(f"  Structural dropped: {counters['structural']}")
    print(f"  Dupes removed     : {counters['dupes']}")
    print(f"  Final entries     : {len(kept)}")
    print(f"  Word invariants   : {len(word_entries)}")
    print()
    print(f"Surviving word invariants ({len(word_list)}):")
    for i in range(0, len(word_list), 4):
        row = word_list[i:i+4]
        print("  " + "  ".join(f"{w:<22s}" for w in row))

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    shutil.copy(LIBRARY_FILE, BACKUP_FILE)
    print(f"\nBacked up to {BACKUP_FILE}")
    with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"Written: {LIBRARY_FILE}")


if __name__ == "__main__":
    clean(dry_run="--dry-run" in sys.argv)
