"""
language/conversation_field.py
===============================
Conversation field — synthesizes virtual context for question-only input.

When the user provides only a question (no context sentence before it),
this module assembles a minimal context sentence from the conversation
window stored in field_state.json. The virtual context is prepended to
the question before triangulation, producing a valid pocket split and
allowing the geometric field to operate normally.

The assembled context is intentionally minimal — only the highest-charge
named words from recent exchanges, ordered by recency. This is not
retrieval by similarity; it is geometric field priming from the
accumulated attractor state of prior conversation.

Architecture role:
  processor.process(sentence)
    ↓ detects question-only input
    ↓ calls conversation_field.prime(sentence)
    ↓ returns primed_sentence with virtual pkt=0
    ↓ continues normal triangulate() → fingerprint → output pipeline

Usage:
    from language.conversation_field import conversation_field
    primed = conversation_field.prime("How do ravens think?")
    # Returns: "ravens demonstrate advanced abilities. How do ravens think?"
"""

import re
from typing import List, Optional
from core.field_state import field_state_manager


# ── Detection constants ───────────────────────────────────────────────────────

# Structural words that don't contribute to context quality
_STRUCTURAL = {
    'the','and','is','in','of','a','to','it','as','that','this',
    'was','be','are','for','on','or','from','by','at','an','not',
    'but','so','if','its','has','had','have','with','how','what',
    'why','where','when','which','who','do','does','did','would',
    'could','should','will','can','may','might','must','shall',
    'about','even','some','any','each','one','two','three',
}

# Max words to use in synthesized context sentence
_MAX_CONTEXT_WORDS = 8

# Minimum conversation window entries needed to prime
_MIN_WINDOW_ENTRIES = 1


class ConversationField:
    """
    Synthesizes a virtual context sentence from the conversation window
    when the user provides a question without an explicit context sentence.
    """

    # ── Detection ─────────────────────────────────────────────────────────────

    def is_question_only(self, sentence: str) -> bool:
        """
        Returns True if input is a question with no preceding context sentence.

        A context sentence is identified by a sentence-ending punctuation
        (. or !) appearing before the question mark. Without this, the
        _insert_pockets() in symbolic_wave produces an empty pkt=0 —
        the question becomes the entire field with no content to select from.

        Examples:
            "How do ravens think?"                    → True  (no context)
            "Ravens are smart. How do ravens think?"  → False (has context)
            "Describe photosynthesis."                → False (not a question)
            "What is ATP used for?"                   → True
        """
        s = sentence.strip()
        if '?' not in s:
            return False  # not a question at all

        # Find last '?' position
        q_pos = s.rfind('?')
        before_q = s[:q_pos]

        # Check for sentence-ending punctuation before the '?'
        has_context_sentence = bool(re.search(r'[.!]', before_q))
        return not has_context_sentence

    # ── Priming ───────────────────────────────────────────────────────────────

    def prime(self, sentence: str) -> tuple:
        """
        Prepend a virtual context sentence to a question-only input.

        Returns (primed_sentence, was_primed, context_words_used)

        Two-path priming strategy:
        1. CONTEXTUAL PATH — question overlaps with recent window topics.
           Use window context words. Good for follow-up questions like
           "Why does it do that?" or "What about neurons?"
        2. REFLECTIVE PATH — question is a fresh topic with no window overlap.
           Reflect the question's own high-charge nouns into pkt=0.
           "How does supply and demand set prices?" →
           pkt=0 = "supply demand prices." pkt=1 = the question.
           This gives the field a real pkt=0/pkt=1 split without
           contaminating it with irrelevant prior context.
        """
        window = field_state_manager.get_conversation_window()

        # Extract content words from the question
        q_words = [
            w.lower().rstrip('?!.,;:')
            for w in sentence.split()
            if w.lower().rstrip('?!.,;:') not in _STRUCTURAL
            and len(w.rstrip('?!.,;:')) > 3
        ]

        # Check topical overlap with conversation window
        window_words = set()
        if window:
            for exchange in window[-4:]:  # last 4 exchanges
                anchor = exchange.get("anchor","").lower()
                if anchor:
                    window_words.add(anchor)
                for w in exchange.get("top_words",[]):
                    window_words.add(w.lower().rstrip('.,!?;:'))

        overlap = sum(1 for w in q_words if w in window_words)
        overlap_ratio = overlap / max(len(q_words), 1)

        # CONTEXTUAL PATH: significant overlap → use window context
        # Threshold 0.4: at least 2 of 5 question words must match window
        # to be considered a genuine follow-up, not a fresh topic.
        if overlap_ratio >= 0.40 and len(window) >= _MIN_WINDOW_ENTRIES:
            context_words = self._collect_context_words(window, sentence)
            if context_words:
                context_sentence = " ".join(context_words) + "."
                primed = f"{context_sentence} {sentence}"
                return primed, True, context_words

        # FRESH TOPIC: no overlap with window — do not prime at all.
        # The geometric library query in the output pipeline handles
        # candidate enrichment for question-only prompts properly.
        # Fake context contaminated the field. Clean question is better.
        return sentence, False, []

    # ── Context word collection ───────────────────────────────────────────────

    def _collect_context_words(
        self,
        window:    List[dict],
        question:  str,
    ) -> List[str]:
        """
        Collect the best context words from the conversation window.

        Strategy:
        1. Start with the anchor words from recent exchanges (subjects)
        2. Add highest-charge content words from recent outputs
        3. Deduplicate and filter structural/question words
        4. Remove words already in the question (they're already pkt=1)
        5. Cap at _MAX_CONTEXT_WORDS

        The question words are excluded because they'll appear in pkt=1
        anyway — including them in pkt=0 would dilute the pocket scoring.
        """
        # Extract question words to exclude from context
        q_words = set(
            w.lower().rstrip('?!.,;:')
            for w in question.split()
            if w.lower().rstrip('?!.,;:') not in _STRUCTURAL
        )

        seen:  set       = set()
        words: List[str] = []

        # Walk window most-recent first
        for exchange in reversed(window):
            # Anchor word — the subject of the exchange
            anchor = exchange.get("anchor", "").lower().strip()
            if (anchor and anchor not in seen
                    and anchor not in _STRUCTURAL
                    and anchor not in q_words
                    and len(anchor) > 2):
                seen.add(anchor)
                words.append(anchor)

            # Top content words from the exchange output
            for w in exchange.get("top_words", []):
                wl = w.lower().strip().rstrip('.,!?;:')
                if (wl and wl not in seen
                        and wl not in _STRUCTURAL
                        and wl not in q_words
                        and len(wl) > 3):
                    seen.add(wl)
                    words.append(wl)

            if len(words) >= _MAX_CONTEXT_WORDS:
                break

        return words[:_MAX_CONTEXT_WORDS]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def window_summary(self) -> str:
        """One-line summary of available conversation context."""
        window = field_state_manager.get_conversation_window()
        if not window:
            return "conversation window: empty"
        words = field_state_manager.get_context_words(n=8)
        return (f"conversation window: {len(window)} exchanges  "
                f"context_words: {words}")


# Module-level singleton
conversation_field = ConversationField()
