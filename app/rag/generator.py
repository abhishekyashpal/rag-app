import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.rag.context import build_context
from app.rag.prompts import (
    PROMPT_VERSION,
    RAG_PROMPT,
)


load_dotenv()


MODEL_NAME = os.getenv(
    "OPENAI_CHAT_MODEL",
    "YOUR_MODEL_NAME",
)

TEMPERATURE = 0

MAX_OUTPUT_TOKENS = 500

MAX_CONTEXT_TOKENS = 6000


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_OUTPUT_TOKENS,
)


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """
    Generate an answer using retrieved documents.
    """

    context = build_context(
        documents=documents,
        max_tokens=MAX_CONTEXT_TOKENS,
    )

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content