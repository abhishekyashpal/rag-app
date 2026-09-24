from app.evaluation.report import (
    build_report,
    print_report,
    save_json_report,
)


metrics = {
    "contextual_recall": 0.91,
    "contextual_precision": 0.87,
    "contextual_relevancy": 0.90,
    "faithfulness": 0.94,
    "answer_relevancy": 0.92,
}


report = build_report(
    dataset_size=5,
    prompt_version="v1",
    metrics=metrics,
)

print_report(report)

save_json_report(report)