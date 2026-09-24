import json
from pathlib import Path


DATASET_PATH = Path(
    "evaluation/datasets/rag_questions.json"
)


def load_evaluation_dataset() -> list[dict]:
    """
    Load evaluation questions from JSON.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    return dataset