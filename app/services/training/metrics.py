from __future__ import annotations

import math


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def binary_metrics(y_true, scores, threshold: float = 0.4) -> dict[str, float]:
    accepted = [index for index, score in enumerate(scores) if score >= threshold]
    correct = sum(int(y_true[index]) for index in accepted)
    return {"accepted": len(accepted), "precision": correct / len(accepted) if accepted else 0.0, "coverage": len(accepted) / len(y_true) if len(y_true) else 0.0}