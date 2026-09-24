from dataclasses import dataclass
import time


@dataclass
class SystemMetrics:
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def milliseconds(
    start_time: float,
    end_time: float,
) -> float:
    """
    Convert elapsed time to milliseconds.
    """

    return (
        end_time - start_time
    ) * 1000    