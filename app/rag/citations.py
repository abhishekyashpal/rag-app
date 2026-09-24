import re

from langchain_core.documents import Document


CITATION_PATTERN = re.compile(
    r"\[CTX-(\d+)\]"
)


def build_citation_map(
    documents: list[Document],
) -> dict[str, dict]:
    """
    Create a mapping between citation IDs and
    document metadata.
    """

    citation_map = {}

    for index, document in enumerate(
        documents,
        start=1,
    ):

        citation_id = f"CTX-{index}"

        citation_map[citation_id] = {
            "source": document.metadata.get(
                "source",
                "Unknown source",
            ),
            "page": document.metadata.get(
                "page_label",
                document.metadata.get(
                    "page",
                    "Unknown page",
                ),
            ),
            "chunk_id": document.metadata.get(
                "chunk_id",
                "Unknown chunk",
            ),
        }

    return citation_map


def extract_citation_ids(
    answer: str,
) -> list[str]:
    """
    Extract [CTX-N] references from the answer.
    """

    matches = CITATION_PATTERN.findall(answer)

    return [
        f"CTX-{number}"
        for number in matches
    ]


def validate_citations(
    answer: str,
    citation_map: dict[str, dict],
) -> list[str]:
    """
    Return citation IDs that appear in the answer
    but do not exist in the citation map.
    """

    citation_ids = extract_citation_ids(answer)

    invalid_citations = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in citation_map
    ]

    return invalid_citations