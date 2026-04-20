"""
language/output_translator.py
==============================
Minimal grammar scaffold for geometric output translation.

Takes the raw word list the geometric field resolved and applies
three deterministic rules to produce natural English:

  1. Word form normalization — conjugate verb to agree with subject
  2. Connective insertion — find one preposition from pkt=1 if useful
  3. Sentence assembly — capitalize first word, add period

No semantic inference. No language model. No external dependencies.
Every decision is made from what the geometry already found.
"""

from typing import List, Dict, Any, Optional


# ── Irregular verb forms ──────────────────────────────────────────────────────
# Only the verbs the field actually produces. Extend as needed.
_THIRD_PERSON_SINGULAR = {
    "carry":       "carries",
    "transmit":    "transmits",
    "transport":   "transports",
    "produce":     "produces",
    "contain":     "contains",
    "store":       "stores",
    "absorb":      "absorbs",
    "release":     "releases",
    "attract":     "attracts",
    "protect":     "protects",
    "fight":       "fights",
    "kill":        "kills",
    "form":        "forms",
    "learn":       "learns",
    "teach":       "teaches",
    "create":      "creates",
    "drive":       "drives",
    "generate":    "generates",
    "convert":     "converts",
    "reduce":      "reduces",
    "freeze":      "freezes",
    "bend":        "bends",
    "use":         "uses",
    "need":        "needs",
    "shine":       "shines",
    "consolidate": "consolidates",
    "communicate": "communicates",
    "neutralize":  "neutralizes",
    "recognize":   "recognizes",
    "control":     "controls",
    "maintain":    "maintains",
    "improve":     "improves",
    "develop":     "develops",
    "require":     "requires",
}

# Verbs that already end in 's' in base form — don't double-add
_ALREADY_S = {"uses", "carries", "produces", "contains", "stores",
              "absorbs", "releases", "attracts", "protects", "fights",
              "kills", "forms", "learns", "teaches", "creates", "drives",
              "generates", "converts", "reduces", "freezes", "bends",
              "communicates", "recognizes", "controls", "maintains",
              "improves", "develops", "requires", "consolidates",
              "transports", "transmits", "neutralizes"}

# Structural words that should never appear in output
_BLOCKED = {
    "than", "rather", "each", "every", "both", "such", "while",
    "since", "once", "even", "just", "also", "only", "more", "most",
    "very", "well", "still", "yet", "too", "then", "thus", "hence",
    "though", "during", "approach", "important",
    # Pronouns
    "they", "them", "their", "those", "these", "there", "here",
    "who", "whom", "whose", "where", "he", "she", "we", "you",
    # Quantifiers/determiners that slip into output
    "major", "minor", "previous", "around", "beyond", "against",
    "whether", "either", "neither", "another", "instead", "nearly",
    # Auxiliaries
    "might", "cannot",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so",
    # Prepositions that produce awkward output when leading a slot
    "with", "for", "at", "on", "of",
    # Spatial/temporal words that lead outputs incorrectly
    "around", "beyond", "during", "above", "below", "across",
    "along", "within", "between", "toward", "towards", "through",
}

# Connective prepositions worth inserting between verb and object
# Only directional/relational prepositions that make clean bridges
# 'with' removed — too ambiguous as a connective (produces 'X with Y' noise)
_CONNECTIVES = {
    "to", "through", "from", "by", "into", "via",
    "across", "within", "between",
}


def _conjugate(verb: str, subject: str) -> str:
    """
    Return third-person singular present of verb if subject is a
    singular noun. If verb already conjugated or subject is plural
    (ends in 's' but not 'ss'), return as-is.
    """
    v = verb.lower().rstrip(".,!?;:")
    s = subject.lower().rstrip(".,!?;:")

    # Already conjugated
    if v in _ALREADY_S:
        return verb

    # Subject is plural (ends in s, not ss) → use base form
    if s.endswith("s") and not s.endswith("ss"):
        return verb  # plural subject — base form

    # Subject is singular — apply third-person singular
    if v in _THIRD_PERSON_SINGULAR:
        # Preserve original capitalisation pattern
        conjugated = _THIRD_PERSON_SINGULAR[v]
        if verb[0].isupper():
            conjugated = conjugated.capitalize()
        return conjugated

    # Default: add 's' for regular verbs
    if not v.endswith("s"):
        return verb + "s"

    return verb


def _find_connective(per_word: List[Dict]) -> Optional[str]:
    """
    Find the highest-tension preposition from pkt=1 words.
    Returns None if none useful found.
    """
    candidates = []
    for w in per_word:
        wl = w.get("word", "").lower().rstrip(".,!?;:")
        if (w.get("pocket", 0) == 1
                and wl in _CONNECTIVES
                and abs(w.get("mean_tension", 0)) > 0.05):
            candidates.append((abs(w.get("mean_tension", 0)), wl))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def translate(
    words: List[Dict],
    fingerprint: Optional[Dict] = None,
    insert_connective: bool = True,
) -> str:
    """
    Translate a list of geometric word dicts into a natural English sentence.

    words: list of {'word': str, 'pool': str, ...} from geometric output
    fingerprint: full fingerprint dict (for connective extraction from pkt=1)
    insert_connective: whether to insert a preposition between verb and object

    Returns a clean English sentence string.
    """
    if not words:
        return ""

    # Filter blocked words
    clean = [w for w in words
             if w.get("word", "").lower().rstrip(".,!?;:") not in _BLOCKED]

    if not clean:
        return ""

    # Single word — capitalize and period
    if len(clean) == 1:
        word = clean[0].get("word", "").strip(".,!?;:")
        return word.capitalize() + "."

    # Identify subject (position 0), verb (pool='verb'/'action'), objects (rest)
    subject_dict = clean[0]
    subject = subject_dict.get("word", "").strip(".,!?;:")

    verb_idx = next(
        (i for i, w in enumerate(clean)
         if w.get("pool") in ("verb", "action") and i > 0),
        None
    )

    if verb_idx is not None:
        verb_dict = clean[verb_idx]
        verb = verb_dict.get("word", "").strip(".,!?;:")
        objects = [w for i, w in enumerate(clean)
                   if i != 0 and i != verb_idx]
    else:
        # No explicit verb slot — treat second word as verb
        if len(clean) >= 2:
            verb = clean[1].get("word", "").strip(".,!?;:")
            objects = [w for w in clean[2:]]
        else:
            verb = ""
            objects = []

    # Conjugate verb — only if it looks like a verb, not a noun
    if verb:
        _v = verb.lower().rstrip(".,!?;:")
        _VERB_LIKE = set(_THIRD_PERSON_SINGULAR.keys()) | set(_ALREADY_S)
        _NOUN_ENDS = ("tion","ness","ment","ity","ance","ence","ism","ist",
                      "ey","ia","ics","ogy","ium","tem","ney","ria")
        _looks_verb = (_v in _VERB_LIKE
                       or (any(_v.endswith(e) for e in ("ate","ize","ise","fy","en","ect","ort","mit","end","orm"))
                           and not any(_v.endswith(e) for e in _NOUN_ENDS)))
        if _looks_verb:
            verb = _conjugate(verb, subject)

    # Find connective from fingerprint pkt=1
    connective = None
    if insert_connective and fingerprint and objects:
        per_word = fingerprint.get("per_word", [])
        connective = _find_connective(per_word)

    # Assemble
    parts = [subject.capitalize()]
    if verb:
        parts.append(verb)
    if connective and objects:
        parts.append(connective)
        parts.extend(w.get("word", "").strip(".,!?;:") for w in objects)
    elif objects:
        parts.extend(w.get("word", "").strip(".,!?;:") for w in objects)

    # Clean and join
    result = " ".join(p for p in parts if p)
    if not result.endswith("."):
        result += "."

    return result


def translate_raw(
    raw_text: str,
    fingerprint: Optional[Dict] = None,
) -> str:
    """
    Translate a raw output string (already assembled) through the
    normalization layer only — conjugation, blocked word removal.

    Used when word dicts are not available, only the string output.
    """
    if not raw_text or raw_text.strip() in (".", ""):
        return raw_text

    words = raw_text.rstrip(".").split()
    if not words:
        return raw_text

    # Filter blocked words and known conjugation artifacts
    # 'ands', 'boths', 'ors' etc come from structural words getting conjugated
    _ARTIFACTS = {"ands", "buts", "ors", "bys", "tos", "boths",
                  "arounds", "beyonds", "theys", "eachs", "mights",
                  "durings", "abouts", "agains", "intos", "overs",
                  "unders", "alongs", "acrosss", "withins", "betweens"}
    words = [w for w in words
             if w.lower().rstrip(".,!?;:") not in _BLOCKED
             and w.lower().rstrip(".,!?;:") not in _ARTIFACTS]

    if not words:
        return raw_text

    # Conjugate word[1] as verb ONLY if it looks like a verb.
    # Nouns (money, system, bacteria) should not be conjugated.
    # A word is verb-like if:
    #   a) it's in our known verb list, OR
    #   b) it ends in a verb suffix and is not a known noun pattern
    if len(words) >= 2:
        _w1 = words[1].lower().rstrip(".,!?;:")
        _VERB_LIKE = set(_THIRD_PERSON_SINGULAR.keys()) | set(_ALREADY_S)
        _NOUN_ENDINGS = ("tion", "ness", "ment", "ity", "ance", "ence",
                         "ism", "ist", "er", "or", "ey", "ia", "ics",
                         "ogy", "ium", "tem", "ney", "ney")
        _is_verb = (_w1 in _VERB_LIKE
                    or (any(_w1.endswith(e) for e in ("ate","ize","ise","fy","en","ect","ort","mit"))
                        and not any(_w1.endswith(e) for e in _NOUN_ENDINGS)))
        if _is_verb:
            words[1] = _conjugate(words[1], words[0])

    # Capitalize first word
    words[0] = words[0].capitalize()

    # Find and insert connective
    if fingerprint and len(words) >= 3:
        per_word = fingerprint.get("per_word", [])
        conn = _find_connective(per_word)
        if conn and conn not in words:
            # Insert after verb (position 2)
            words = words[:2] + [conn] + words[2:]

    return " ".join(words) + "."
