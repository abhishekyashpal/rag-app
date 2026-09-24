import re


CITATION_PATTERN = re.compile(
    r"\[CTX-(\d+)\]"
)


def extract_citation_ids(
    answer: str,
) -> list[str]:
    """
    Extract citation IDs from an answer.
    """

    matches = CITATION_PATTERN.findall(
        answer
    )

    return [
        f"CTX-{number}"
        for number in matches
    ]


def citation_validity(
    answer: str,
    valid_citation_ids: set[str],
) -> float:
    """
    Return the fraction of citations that
    point to valid context IDs.
    """

    citation_ids = extract_citation_ids(
        answer
    )

    if not citation_ids:
        return 1.0

    valid_count = sum(
        citation_id in valid_citation_ids
        for citation_id in citation_ids
    )

    return valid_count / len(citation_ids)