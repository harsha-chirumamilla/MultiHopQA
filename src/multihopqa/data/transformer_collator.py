"""Collate HotpotQA examples as one question-context transformer sequence."""

import torch


class TransformerSupportCollator:
    """Serialize all evidence sentences and map tokens back to their sentences."""

    def __init__(self, tokenizer, max_sequence_length=512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("The transformer support baseline needs a fast tokenizer.")
        if max_sequence_length > tokenizer.model_max_length:
            raise ValueError(
                f"max_sequence_length ({max_sequence_length}) exceeds tokenizer limit "
                f"({tokenizer.model_max_length})."
            )
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, examples):
        questions = []
        contexts = []
        sentence_ranges = []
        metadata = []
        labels = []

        for example in examples:
            nodes = example["sentences"]
            if not nodes:
                raise ValueError(f"Example {example['id']} has no sentence nodes.")

            parts = []
            offsets = []
            context_length = 0
            for node_index, node in enumerate(nodes):
                if node_index:
                    separator = " [SEP] "
                    parts.append(separator)
                    context_length += len(separator)

                part = f"{node['title']}: {node['text']}"
                parts.append(part)
                offsets.append((context_length, context_length + len(part)))
                context_length += len(part)

            questions.append(example["question"])
            contexts.append("".join(parts))
            sentence_ranges.append(offsets)
            labels.append([float(node["is_supporting_fact"]) for node in nodes])
            metadata.append(
                [
                    {
                        "title": node["title"],
                        "sentence_index": node["sentence_index"],
                    }
                    for node in nodes
                ]
            )

        encoded = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_sequence_length,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        batch_size, sequence_length = encoded["input_ids"].shape
        # Batch tensors only need room for the largest example in this batch.
        # HotpotQA contexts can contain more sentences than any fixed cap, even
        # though most are later hidden by BERT's sequence-length truncation.
        max_nodes = max(len(nodes) for nodes in labels)
        sentence_token_mask = torch.zeros(
            (batch_size, max_nodes, sequence_length), dtype=torch.bool
        )
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        support_labels = torch.zeros((batch_size, max_nodes), dtype=torch.float)

        for batch_index in range(batch_size):
            offsets = encoded["offset_mapping"][batch_index]
            context_tokens = encoded["token_type_ids"][batch_index] == 1
            nonempty_tokens = offsets[:, 1] > offsets[:, 0]
            for node_index, (char_start, char_end) in enumerate(
                sentence_ranges[batch_index]
            ):
                overlaps_sentence = (
                    (offsets[:, 0] < char_end) & (offsets[:, 1] > char_start)
                )
                token_mask = context_tokens & nonempty_tokens & overlaps_sentence
                if token_mask.any():
                    sentence_token_mask[batch_index, node_index] = token_mask
                    node_mask[batch_index, node_index] = True
                    support_labels[batch_index, node_index] = labels[batch_index][
                        node_index
                    ]

        encoded.pop("offset_mapping")
        return {
            **encoded,
            "sentence_token_mask": sentence_token_mask,
            "node_mask": node_mask,
            "support_labels": support_labels,
            "example_ids": [example["id"] for example in examples],
            "gold_supporting_facts": [
                example["supporting_facts"] for example in examples
            ],
            "node_metadata": metadata,
        }
