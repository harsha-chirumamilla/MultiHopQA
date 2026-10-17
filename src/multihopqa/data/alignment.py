"""Helpers that keep HotpotQA support labels aligned with its sentences."""


def read_context(context):
    """Return context as [(title, [sentence, ...]), ...].

    Hugging Face can represent a sequence of records as a dictionary of lists;
    accepting that form and a list of records makes the expected data shape clear.
    """
    if isinstance(context, dict):
        titles = context.get("title")
        sentence_lists = context.get("sentences")

        if titles is None or sentence_lists is None:
            raise ValueError("Context must contain 'title' and 'sentences'.")
        if len(titles) != len(sentence_lists):
            raise ValueError("Context title and sentence counts do not match.")

        paragraphs = []
        for title, sentences in zip(titles, sentence_lists):
            paragraphs.append((str(title), list(sentences)))
        return paragraphs

    if isinstance(context, list):
        paragraphs = []
        for paragraph in context:
            if not isinstance(paragraph, dict):
                raise ValueError("Each context paragraph must be a dictionary.")
            if "title" not in paragraph or "sentences" not in paragraph:
                raise ValueError("Each paragraph needs a title and sentences.")
            paragraphs.append((str(paragraph["title"]), list(paragraph["sentences"])))
        return paragraphs

    raise ValueError("Context must be a dictionary or a list of paragraphs.")


def read_supporting_facts(supporting_facts):
    """Return supporting facts as [(title, zero_based_sentence_index), ...]."""
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title")
        sentence_ids = supporting_facts.get("sent_id")

        if titles is None or sentence_ids is None:
            raise ValueError("Supporting facts need 'title' and 'sent_id'.")
        if len(titles) != len(sentence_ids):
            raise ValueError("Supporting-fact title and sentence counts do not match.")

        return [(str(title), int(sentence_id)) for title, sentence_id in zip(titles, sentence_ids)]

    if isinstance(supporting_facts, list):
        facts = []
        for fact in supporting_facts:
            if isinstance(fact, dict):
                title = fact.get("title")
                sentence_id = fact.get("sent_id", fact.get("sentence_index"))
            elif isinstance(fact, (list, tuple)) and len(fact) == 2:
                title, sentence_id = fact
            else:
                raise ValueError("Each supporting fact must contain a title and sentence index.")

            if title is None or sentence_id is None:
                raise ValueError("A supporting fact is missing its title or sentence index.")
            facts.append((str(title), int(sentence_id)))
        return facts

    raise ValueError("Supporting facts must be a dictionary or a list.")


def make_sentence_nodes(paragraphs, supporting_facts):
    """Flatten paragraph sentences without changing their original positions."""
    support_keys = set(supporting_facts)
    available_keys = set()
    sentence_nodes = []

    for paragraph_index, (title, sentences) in enumerate(paragraphs):
        for sentence_index, sentence in enumerate(sentences):
            key = (title, sentence_index)
            available_keys.add(key)
            sentence_nodes.append(
                {
                    "node_index": len(sentence_nodes),
                    "paragraph_index": paragraph_index,
                    "title": title,
                    "sentence_index": sentence_index,
                    "text": str(sentence),
                    "is_supporting_fact": key in support_keys,
                }
            )

    support_records = []
    unmatched_records = []
    for title, sentence_index in supporting_facts:
        fact = {"title": title, "sentence_index": sentence_index}
        support_records.append(fact)
        if (title, sentence_index) not in available_keys:
            unmatched_records.append(fact)

    return sentence_nodes, support_records, unmatched_records
