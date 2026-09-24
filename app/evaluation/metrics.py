from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)


def create_metrics():
    """
    Create the metrics used to evaluate our RAG system.
    """

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        include_reason=True,
    )

    contextual_precision = ContextualPrecisionMetric(
        threshold=0.7,
        include_reason=True,
    )

    contextual_recall = ContextualRecallMetric(
        threshold=0.7,
        include_reason=True,
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
    )

    return [
        contextual_relevancy,
        contextual_precision,
        contextual_recall,
        faithfulness,
        answer_relevancy,
    ]