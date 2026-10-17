"""Turn a batch of questions into padded text and graph tensors."""

import torch

from multihopqa.graph.builder import build_adjacency


ANSWER_TYPES = {"yes": 0, "no": 1, "span": 2}


def find_answer_tokens(answer, sentence, offsets, token_type_ids):
    """Find the answer's first token span inside one sentence, if it is present."""
    answer_text = answer.strip().casefold()
    if not answer_text:
        return None

    char_start = sentence.casefold().find(answer_text)
    if char_start < 0:
        return None
    char_end = char_start + len(answer_text)

    start_token = None
    end_token = None

    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_type_ids[token_index] != 1 or token_end <= token_start:
            continue

        if token_start <= char_start < token_end:
            start_token = token_index
        if token_start < char_end <= token_end:
            end_token = token_index

    if start_token is None or end_token is None or end_token < start_token:
        return None

    return start_token, end_token


class GraphQADataCollator:
    """Tokenize sentence nodes and pad their separate question graphs."""

    def __init__(self, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, examples):
        questions = []
        sentence_texts = []
        locations = []
        graph_adjacencies = []
        node_counts = []

        for graph_index, example in enumerate(examples):
            nodes = example["sentences"]
            if not nodes:
                raise ValueError(f"Example {example['id']} has no sentence nodes.")

            graph_adjacencies.append(build_adjacency(nodes))
            node_counts.append(len(nodes))

            for node_index, node in enumerate(nodes):
                questions.append(example["question"])
                sentence_texts.append(node["text"])
                locations.append((graph_index, node_index))

        encoded = self.tokenizer(
            questions,
            sentence_texts,
            max_length=self.max_sequence_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        batch_size = len(examples)
        max_nodes = max(node_counts)
        sequence_length = encoded["input_ids"].shape[1]
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full(
            (batch_size, max_nodes, sequence_length),
            pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        adjacency = torch.zeros(
            (batch_size, max_nodes, max_nodes),
            dtype=torch.bool,
        )
        support_labels = torch.zeros((batch_size, max_nodes), dtype=torch.float)

        for graph_index, example in enumerate(examples):
            node_count = node_counts[graph_index]
            node_mask[graph_index, :node_count] = True
            adjacency[graph_index, :node_count, :node_count] = torch.tensor(
                graph_adjacencies[graph_index],
                dtype=torch.bool,
            )
            for node_index, node in enumerate(example["sentences"]):
                support_labels[graph_index, node_index] = float(
                    node["is_supporting_fact"]
                )

        for flat_index, (graph_index, node_index) in enumerate(locations):
            input_ids[graph_index, node_index] = encoded["input_ids"][flat_index]
            attention_mask[graph_index, node_index] = encoded["attention_mask"][flat_index]
            token_type_ids[graph_index, node_index] = encoded["token_type_ids"][flat_index]
            offsets = encoded["offset_mapping"][flat_index]
            context_mask[graph_index, node_index] = (
                (encoded["token_type_ids"][flat_index] == 1)
                & (encoded["attention_mask"][flat_index] == 1)
                & (offsets[:, 1] > offsets[:, 0])
            )

        answer_type_labels = []
        span_node = torch.zeros(batch_size, dtype=torch.long)
        span_start = torch.zeros(batch_size, dtype=torch.long)
        span_end = torch.zeros(batch_size, dtype=torch.long)
        span_available = torch.zeros(batch_size, dtype=torch.bool)

        next_flat_index = 0
        gold_answers = []
        gold_supporting_facts = []
        node_metadata = []

        for graph_index, example in enumerate(examples):
            answer = example["answer"]
            answer_kind = example["answer_kind"]

            if answer_kind == "yes_no":
                answer_type_labels.append(ANSWER_TYPES[answer.strip().casefold()])
            else:
                answer_type_labels.append(ANSWER_TYPES["span"])

                for node_index, node in enumerate(example["sentences"]):
                    flat_index = next_flat_index + node_index
                    token_span = find_answer_tokens(
                        answer,
                        node["text"],
                        encoded["offset_mapping"][flat_index].tolist(),
                        encoded["token_type_ids"][flat_index].tolist(),
                    )
                    if token_span is not None:
                        span_node[graph_index] = node_index
                        span_start[graph_index] = token_span[0]
                        span_end[graph_index] = token_span[1]
                        span_available[graph_index] = True
                        break

            next_flat_index += node_counts[graph_index]
            gold_answers.append(answer)
            gold_supporting_facts.append(example["supporting_facts"])
            node_metadata.append(
                [
                    {
                        "title": node["title"],
                        "sentence_index": node["sentence_index"],
                    }
                    for node in example["sentences"]
                ]
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "context_mask": context_mask,
            "node_mask": node_mask,
            "adjacency": adjacency,
            "support_labels": support_labels,
            "answer_type_labels": torch.tensor(answer_type_labels, dtype=torch.long),
            "span_node": span_node,
            "span_start": span_start,
            "span_end": span_end,
            "span_available": span_available,
            "example_ids": [example["id"] for example in examples],
            "gold_answers": gold_answers,
            "gold_supporting_facts": gold_supporting_facts,
            "node_metadata": node_metadata,
        }
