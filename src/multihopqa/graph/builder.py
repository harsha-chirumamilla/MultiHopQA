"""Build sentence graphs for one HotpotQA question."""

from multihopqa.graph.entity_extraction import LEGACY_COMMON_SENTENCE_STARTS
from multihopqa.graph.entity_extraction import extract_entities
from multihopqa.graph.entity_normalization import normalize_entity


GRAPH_MODES = ("current", "legacy")


def _set_undirected_edge(adjacency, first, second):
    adjacency[first][second] = True
    adjacency[second][first] = True


def _article_key(node, node_index):
    paragraph_index = node.get("paragraph_index")
    if paragraph_index is not None:
        return "paragraph", paragraph_index

    title = normalize_entity(node.get("title", ""))
    if title:
        return "title", title
    return "node", node_index


def _sentence_order(node, node_index):
    try:
        sentence_index = int(node.get("sentence_index", node_index))
    except (TypeError, ValueError):
        sentence_index = node_index
    return sentence_index, node_index


def _contains_phrase(tokens, phrase_tokens):
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return False

    phrase_length = len(phrase_tokens)
    return any(
        tokens[start : start + phrase_length] == phrase_tokens
        for start in range(len(tokens) - phrase_length + 1)
    )


def _build_legacy_adjacency(sentence_nodes):
    """Reproduce the original title/entity clique graph for comparisons."""
    node_count = len(sentence_nodes)
    adjacency = [[False for _ in range(node_count)] for _ in range(node_count)]
    entity_to_nodes = {}

    for node_index, node in enumerate(sentence_nodes):
        entities = extract_entities(
            node["text"],
            sentence_start_stopwords=LEGACY_COMMON_SENTENCE_STARTS,
            filter_acronym_openers=False,
        )
        title = normalize_entity(node.get("title", ""))
        if title:
            entities.add(title)

        adjacency[node_index][node_index] = True
        for entity in entities:
            entity_to_nodes.setdefault(entity, []).append(node_index)

    for node_indexes in entity_to_nodes.values():
        for first in node_indexes:
            for second in node_indexes:
                adjacency[first][second] = True

    return adjacency


def _build_current_adjacency(sentence_nodes):
    node_count = len(sentence_nodes)
    adjacency = [[False for _ in range(node_count)] for _ in range(node_count)]
    article_nodes = {}

    for node_index, node in enumerate(sentence_nodes):
        adjacency[node_index][node_index] = True
        article_nodes.setdefault(_article_key(node, node_index), []).append(node_index)

    article_anchors = {}
    article_titles = {}
    for article_key, node_indexes in article_nodes.items():
        anchor = min(
            node_indexes,
            key=lambda index: _sentence_order(sentence_nodes[index], index),
        )
        article_anchors[article_key] = anchor
        article_titles[article_key] = normalize_entity(
            sentence_nodes[anchor].get("title", "")
        ).split()

        # Preserve article-level context with O(n) star edges instead of an
        # all-to-all clique among the article's sentences.
        for node_index in node_indexes:
            if node_index != anchor:
                _set_undirected_edge(adjacency, anchor, node_index)

    title_targets = [
        (article_key, title_tokens, article_anchors[article_key])
        for article_key, title_tokens in article_titles.items()
        if title_tokens
    ]
    for node_index, node in enumerate(sentence_nodes):
        sentence_tokens = normalize_entity(node.get("text", "")).split()
        source_article = _article_key(node, node_index)
        for target_article, title_tokens, target_anchor in title_targets:
            if target_article == source_article:
                continue
            if _contains_phrase(sentence_tokens, title_tokens):
                _set_undirected_edge(adjacency, node_index, target_anchor)

    entity_to_nodes = {}
    for node_index, node in enumerate(sentence_nodes):
        for entity in extract_entities(node.get("text", "")):
            normalized_entity = normalize_entity(entity)
            if normalized_entity:
                entity_to_nodes.setdefault(normalized_entity, []).append(node_index)

    for node_indexes in entity_to_nodes.values():
        if len(node_indexes) < 2:
            continue
        representative = node_indexes[0]
        for node_index in node_indexes[1:]:
            _set_undirected_edge(adjacency, representative, node_index)

    return adjacency


def build_adjacency(sentence_nodes, graph_mode="current"):
    """Return a symmetric adjacency matrix for one question's sentence nodes.

    The current graph uses the lowest-index sentence in each article as a
    context hub, links title mentions to the referenced article hub, and links
    repeated normalized name phrases through a representative sentence.
    ``legacy`` is available for controlled comparisons with the original graph.
    """
    if graph_mode == "current":
        return _build_current_adjacency(sentence_nodes)
    if graph_mode == "legacy":
        return _build_legacy_adjacency(sentence_nodes)
    raise ValueError(
        f"Unknown graph mode {graph_mode!r}; expected one of {GRAPH_MODES}."
    )
