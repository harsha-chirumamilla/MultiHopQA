"""A simple proper-name extractor for the first graph experiment."""

import re

from multihopqa.graph.entity_normalization import normalize_entity


CAPITALIZED_PHRASE = re.compile(
    r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b"
)
ALL_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")

COMMON_SENTENCE_STARTS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "he",
    "her",
    "his",
    "in",
    "it",
    "its",
    "she",
    "the",
    "they",
    "this",
    "that",
    "these",
    "those",
    "we",
    "when",
    "where",
}


def extract_entities(text):
    """Find capitalized names and acronyms, then normalize and de-duplicate them."""
    found = set()

    for match in CAPITALIZED_PHRASE.finditer(text):
        phrase = match.group(0)
        first_word = phrase.split()[0].casefold()

        at_sentence_start = not text[: match.start()].strip()
        is_one_word_opener = len(phrase.split()) == 1
        if at_sentence_start and is_one_word_opener and first_word in COMMON_SENTENCE_STARTS:
            continue

        normalized = normalize_entity(phrase)
        if normalized:
            found.add(normalized)

    for match in ALL_CAPS_WORD.finditer(text):
        normalized = normalize_entity(match.group(0))
        if normalized:
            found.add(normalized)

    return found
