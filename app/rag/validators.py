def validate_answer(answer: str) -> str:
    """
    Basic validation of the generated answer.
    """

    if not answer:
        raise ValueError(
            "LLM returned an empty response."
        )

    answer = answer.strip()

    if not answer:
        raise ValueError(
            "LLM returned an empty response."
        )

    return answer