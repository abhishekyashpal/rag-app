from datetime import datetime
from pathlib import Path
import json


RESULTS_DIRECTORY = Path(
    "evaluation/results"
)


def save_json_report(
    report: dict,
    filename: str = "evaluation_report.json",
) -> Path:
    """
    Save an evaluation report as JSON.
    """

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIRECTORY / filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def build_report(
    dataset_size: int,
    prompt_version: str,
    metrics: dict,
) -> dict:
    """
    Build a structured evaluation report.
    """

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": dataset_size,
        "prompt_version": prompt_version,
        "metrics": metrics,
    }


def print_report(
    report: dict,
) -> None:
    """
    Print a human-readable evaluation report.
    """

    print()
    print("=" * 80)
    print("RAG EVALUATION REPORT")
    print("=" * 80)

    print(
        f"Timestamp: {report['timestamp']}"
    )

    print(
        f"Dataset size: {report['dataset_size']}"
    )

    print(
        f"Prompt version: {report['prompt_version']}"
    )

    print()
    print("-" * 80)
    print("METRICS")
    print("-" * 80)

    for metric_name, metric_value in (
        report["metrics"].items()
    ):
        if isinstance(metric_value, float):
            print(
                f"{metric_name:<30} "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name:<30} "
                f"{metric_value}"
            )

    print("=" * 80)