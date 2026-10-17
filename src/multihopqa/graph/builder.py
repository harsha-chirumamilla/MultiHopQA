"""Build one sentence graph using self-loops and shared entity names."""

from multihopqa.graph.entity_extraction import extract_entities
from multihopqa.graph.entity_normalization import normalize_entity


def build_adjacency(sentence_nodes):
    """Return a dense adjacency matrix for one question's sentence nodes.

    Every node gets a self-loop. Sentences are also connected when they share
    a detected name, acronym, or paragraph title.
    """
    node_count = len(sentence_nodes)
    adjacency = [[False for _ in range(node_count)] for _ in range(node_count)]
    entity_to_nodes = {}

    for node_index, node in enumerate(sentence_nodes):
        entities = extract_entities(node["text"])
        title = normalize_entity(node["title"])
        if title:
            entities.add(title)

        adjacency[node_index][node_index] = True

        for entity in entities:
            if entity not in entity_to_nodes:
                entity_to_nodes[entity] = []
            entity_to_nodes[entity].append(node_index)

    for node_indexes in entity_to_nodes.values():
        for first in node_indexes:
            for second in node_indexes:
                adjacency[first][second] = True

    return adjacency
