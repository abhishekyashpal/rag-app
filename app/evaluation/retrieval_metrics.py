def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    retrieved_ids:
        IDs returned by the retriever.

    relevant_ids:
        Ground-truth relevant document/chunk IDs.

    k:
        Number of top results to evaluate.
    """

    if not relevant_ids:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])

    relevant_retrieved = (
        retrieved_top_k & relevant_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_ids)
    )

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.
    """

    if k <= 0:
        return 0.0

    retrieved_top_k = retrieved_ids[:k]

    relevant_count = sum(
        1
        for document_id in retrieved_top_k
        if document_id in relevant_ids
    )

    return relevant_count / k

def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Return 1 if at least one relevant result
    appears in the top K, otherwise 0.
    """

    retrieved_top_k = retrieved_ids[:k]

    return float(
        any(
            document_id in relevant_ids
            for document_id in retrieved_top_k
        )
    )

def mean_reciprocal_rank(
    rankings: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """
    Calculate MRR across multiple queries.
    """

    if not rankings:
        return 0.0

    scores = [
        reciprocal_rank(
            retrieved_ids=ranking,
            relevant_ids=relevant,
        )
        for ranking, relevant in zip(
            rankings,
            relevant_sets,
        )
    ]

    return sum(scores) / len(scores)