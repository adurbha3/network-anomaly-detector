"""Evaluates detection quality against the synthetic ground truth label,
comparing the small detected-candidate set back against the full stream's
true anomaly count (so a miss that never even became a z-score candidate
still counts against recall)."""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def confusion_counts(true_positives: int, false_positives: int, false_negatives: int) -> dict:
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_detection(full_df: DataFrame, detected_pdf: pd.DataFrame) -> dict:
    total_true_anomalies = full_df.filter(F.col("is_anomaly")).count()

    detected = detected_pdf[detected_pdf["is_detected_anomaly"]]
    true_positives = int((detected["is_anomaly"]).sum())
    false_positives = int((~detected["is_anomaly"]).sum())
    false_negatives = total_true_anomalies - true_positives

    return confusion_counts(true_positives, false_positives, false_negatives)
