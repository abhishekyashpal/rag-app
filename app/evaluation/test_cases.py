from deepeval.test_case import LLMTestCase

from app.rag.run_rag import run_rag


def build_test_cases(
    dataset: list[dict],
) -> list[LLMTestCase]:
    """
    Run the RAG pipeline for every evaluation question
    and convert the results into DeepEval test cases.
    """

    test_cases = []

    for item in dataset:

        question = item["question"]
        expected_answer = item["expected_answer"]

        result = run_rag(question)

        retrieval_context = []

        for citation in result["citations"]:
            retrieval_context.append(
                (
                    f"Source: {citation['source']}\n"
                    f"Page: {citation['page']}\n"
                    f"Chunk ID: {citation['chunk_id']}"
                )
            )

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

    return test_cases