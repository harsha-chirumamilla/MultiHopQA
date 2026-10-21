"""Small, explainable normalization for entity strings."""

import re


def normalize_entity(entity):
    """Lowercase an entity and replace punctuation with spaces."""
    entity = entity.casefold()
    entity = re.sub(r"\W+", " ", entity)
    return " ".join(entity.split())
