"""Build full-context, overlapping BERT windows and align QA labels."""

import torch

from multihopqa.graph.builder import build_adjacency


ANSWER_TYPES = {"yes": 0, "no": 1, "span": 2}


class HotpotQAReaderCollator:
    """Tokenize each question with all titled evidence, using overlapping windows.

    Windows cover the whole supplied context, so long examples are not silently
    cut off at BERT's sequence limit. Sentence IDs remain global across windows
    and let the reader aggregate evidence and score supporting facts.
    """

    def __init__(
        self,
        tokenizer,
        max_sequence_length=512,
        stride=128,
        graph_mode="current",
        static_shapes=False,
        static_node_count=256,
        window_padding_multiple=4,
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("The reader needs a fast tokenizer for offset alignment.")
        if tokenizer.pad_token_id is None:
            raise ValueError("The reader tokenizer must define a padding token.")
        if max_sequence_length > tokenizer.model_max_length:
            raise ValueError(
                f"max_sequence_length ({max_sequence_length}) exceeds tokenizer limit "
                f"({tokenizer.model_max_length})."
            )
        if stride < 0 or stride >= max_sequence_length:
            raise ValueError("stride must be non-negative and smaller than max_sequence_length.")
        if static_node_count < 1 or window_padding_multiple < 1:
            raise ValueError("Static shape sizes must be positive.")
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.stride = stride
        self.graph_mode = graph_mode
        self.static_shapes = static_shapes
        self.static_node_count = static_node_count
        self.window_padding_multiple = window_padding_multiple

    def __call__(self, examples):
        questions = []
        contexts = []
        sentence_ranges = []
        answer_ranges = []
        metadata = []
        labels = []
        adjacencies = []

        for example in examples:
            nodes = example["sentences"]
            if not nodes:
                raise ValueError(f"Example {example['id']} has no sentence nodes.")

            parts = []
            ranges = []
            answers = []
            char_position = 0
            for node_index, node in enumerate(nodes):
                if node_index:
                    separator = " [SEP] "
                    parts.append(separator)
                    char_position += len(separator)
                title_prefix = f"{node['title']}: "
                parts.append(title_prefix)
                char_position += len(title_prefix)
                text_start = char_position
                sentence = str(node["text"])
                parts.append(sentence)
                text_end = text_start + len(sentence)
                ranges.append((text_start, text_end))
                char_position = text_end

                answer = str(example["answer"]).strip()
                answer_start = sentence.casefold().find(answer.casefold())
                if answer_start >= 0:
                    answers.append(
                        (
                            node_index,
                            text_start + answer_start,
                            text_start + answer_start + len(answer),
                            bool(node["is_supporting_fact"]),
                        )
                    )

            questions.append(example["question"])
            contexts.append("".join(parts))
            sentence_ranges.append(ranges)
            answer_ranges.append(answers)
            labels.append([float(node["is_supporting_fact"]) for node in nodes])
            metadata.append(
                [
                    {"title": node["title"], "sentence_index": node["sentence_index"]}
                    for node in nodes
                ]
            )
            adjacencies.append(build_adjacency(nodes, self.graph_mode))

        encoded = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_sequence_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        window_to_example = encoded.pop("overflow_to_sample_mapping").tolist()
        offsets = encoded.pop("offset_mapping")
        batch_size = len(examples)
        sequence_length = encoded["input_ids"].shape[1]
        window_counts = [window_to_example.count(index) for index in range(batch_size)]
        max_windows = max(window_counts)
        max_nodes = max(len(example["sentences"]) for example in examples)
        if self.static_shapes:
            if max_nodes > self.static_node_count:
                raise ValueError(
                    f"An example has {max_nodes} sentences, exceeding the configured "
                    f"static TPU shape of {self.static_node_count}. Increase "
                    "static_node_count to retain all sentence nodes."
                )
            multiple = self.window_padding_multiple
            max_windows = (max_windows + multiple - 1) // multiple * multiple
            max_nodes = self.static_node_count
        device = encoded["input_ids"].device

        input_ids = torch.zeros(
            (batch_size, max_windows, sequence_length), dtype=torch.long, device=device
        )
        input_ids.fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        token_sentence_ids = torch.full_like(input_ids, -1)
        window_mask = torch.zeros((batch_size, max_windows), dtype=torch.bool, device=device)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
        support_labels = torch.zeros((batch_size, max_nodes), dtype=torch.float, device=device)
        adjacency = torch.zeros(
            (batch_size, max_nodes, max_nodes), dtype=torch.bool, device=device
        )
        answer_type_labels = torch.empty(batch_size, dtype=torch.long, device=device)
        span_window = torch.zeros(batch_size, dtype=torch.long, device=device)
        span_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        span_end = torch.zeros(batch_size, dtype=torch.long, device=device)
        span_available = torch.zeros(batch_size, dtype=torch.bool, device=device)
        local_window_index = [0] * batch_size
        window_indices = [[] for _ in range(batch_size)]

        for flat_window_index, example_index in enumerate(window_to_example):
            local_index = local_window_index[example_index]
            local_window_index[example_index] += 1
            window_indices[example_index].append(flat_window_index)
            window_mask[example_index, local_index] = True
            input_ids[example_index, local_index] = encoded["input_ids"][flat_window_index]
            attention_mask[example_index, local_index] = encoded["attention_mask"][flat_window_index]
            token_type_ids[example_index, local_index] = encoded["token_type_ids"][flat_window_index]

            token_offsets = offsets[flat_window_index]
            is_context = (
                (encoded["token_type_ids"][flat_window_index] == 1)
                & (encoded["attention_mask"][flat_window_index] == 1)
                & (token_offsets[:, 1] > token_offsets[:, 0])
            )
            context_mask[example_index, local_index] = is_context
            for node_index, (char_start, char_end) in enumerate(sentence_ranges[example_index]):
                overlaps = (
                    (token_offsets[:, 0] < char_end)
                    & (token_offsets[:, 1] > char_start)
                    & is_context
                )
                token_sentence_ids[example_index, local_index, overlaps] = node_index

        for example_index, example in enumerate(examples):
            node_count = len(example["sentences"])
            for node_index in range(node_count):
                visible = (token_sentence_ids[example_index] == node_index).any()
                node_mask[example_index, node_index] = visible
                support_labels[example_index, node_index] = labels[example_index][node_index]
            adjacency[example_index, :node_count, :node_count] = torch.tensor(
                adjacencies[example_index], dtype=torch.bool, device=device
            )

            answer = str(example["answer"]).strip().casefold()
            if answer in ANSWER_TYPES and example["answer_kind"] == "yes_no":
                answer_type_labels[example_index] = ANSWER_TYPES[answer]
            else:
                answer_type_labels[example_index] = ANSWER_TYPES["span"]
                ordered_answer_ranges = sorted(
                    answer_ranges[example_index],
                    key=lambda item: not item[3],
                )
                for _, answer_start_char, answer_end_char, _ in ordered_answer_ranges:
                    for local_index, flat_window_index in enumerate(
                        window_indices[example_index]
                    ):
                        token_offsets = offsets[flat_window_index]
                        token_type = encoded["token_type_ids"][flat_window_index]
                        attention = encoded["attention_mask"][flat_window_index]
                        start_matches = (
                            (token_offsets[:, 0] <= answer_start_char)
                            & (token_offsets[:, 1] > answer_start_char)
                            & (token_type == 1)
                            & (attention == 1)
                        ).nonzero(as_tuple=False)
                        end_matches = (
                            (token_offsets[:, 0] < answer_end_char)
                            & (token_offsets[:, 1] >= answer_end_char)
                            & (token_type == 1)
                            & (attention == 1)
                        ).nonzero(as_tuple=False)
                        if start_matches.numel() and end_matches.numel():
                            start_index = int(start_matches[0].item())
                            end_index = int(end_matches[-1].item())
                            if end_index >= start_index:
                                span_window[example_index] = local_index
                                span_start[example_index] = start_index
                                span_end[example_index] = end_index
                                span_available[example_index] = True
                                break
                    if span_available[example_index]:
                        break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "context_mask": context_mask,
            "token_sentence_ids": token_sentence_ids,
            "window_mask": window_mask,
            "node_mask": node_mask,
            "adjacency": adjacency,
            "support_labels": support_labels,
            "answer_type_labels": answer_type_labels,
            "span_window": span_window,
            "span_start": span_start,
            "span_end": span_end,
            "span_available": span_available,
            "example_ids": [example["id"] for example in examples],
            "gold_answers": [example["answer"] for example in examples],
            "gold_supporting_facts": [example["supporting_facts"] for example in examples],
            "node_metadata": metadata,
        }
