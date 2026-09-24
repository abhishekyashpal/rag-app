from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document


DEFAULT_MAX_CONTEXT_TOKENS = 6000

# Tokenizer used for OpenAI-style token counting.
ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class ContextChunk:
    """
    A retrieved document prepared for context injection.
    """

    citation_id: str
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    """
    Count tokens in a string.
    """

    return len(ENCODING.encode(text))


def build_context_chunks(
    documents: list[Document],
) -> list[ContextChunk]:
    """
    Convert retrieved documents into individually formatted
    context chunks.

    Each chunk is token-counted independently.
    """

    context_chunks = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page_label",
            document.metadata.get(
                "page",
                "Unknown page",
            ),
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown chunk",
        )

        text = document.page_content.strip()

        if not text:
            continue

        citation_id = f"CTX-{index}"

        context_text = (
            f"[{citation_id}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk_id}\n\n"
            f"{text}"
        )

        token_count = count_tokens(context_text)

        context_chunks.append(
            ContextChunk(
                citation_id=citation_id,
                text=context_text,
                token_count=token_count,
            )
        )

    return context_chunks


def select_context_chunks(
    context_chunks: list[ContextChunk],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[ContextChunk]:
    """
    Select complete context chunks until the token budget
    is exhausted.

    A chunk is either included completely or excluded.
    We never cut a chunk in the middle.
    """

    selected_chunks = []

    total_tokens = 0

    for chunk in context_chunks:

        # If this chunk would exceed the budget,
        # skip it rather than cutting it.
        if total_tokens + chunk.token_count > max_tokens:
            continue

        selected_chunks.append(chunk)

        total_tokens += chunk.token_count

    return selected_chunks


def build_context(
    documents: list[Document],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    """
    Build final context from retrieved documents while
    respecting a token budget.

    Only complete chunks are included.
    """

    if not documents:
        return "No relevant context was retrieved."

    context_chunks = build_context_chunks(documents)

    selected_chunks = select_context_chunks(
        context_chunks=context_chunks,
        max_tokens=max_tokens,
    )

    if not selected_chunks:
        return (
            "No retrieved context could fit within "
            "the context token budget."
        )

    return "\n\n".join(
        chunk.text
        for chunk in selected_chunks
    )

def get_context_token_count(
    context: str,
) -> int:
    """
    Return the token count of the final context.
    """

    return count_tokens(context)