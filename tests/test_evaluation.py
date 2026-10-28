import pandas as pd

from app.evaluation import confusion_counts, evaluate_detection


def test_confusion_counts_known_case():
    result = confusion_counts(true_positives=8, false_positives=2, false_negatives=2)
    assert result["precision"] == 0.8
    assert result["recall"] == 0.8
    assert abs(result["f1"] - 0.8) < 1e-9


def test_confusion_counts_handles_zero_division():
    result = confusion_counts(true_positives=0, false_positives=0, false_negatives=0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_evaluate_detection_against_ground_truth(spark):
    full_df = spark.createDataFrame(
        [
            (0, 0, True), (0, 1, True), (0, 2, False),
            (0, 3, False), (0, 4, True),
        ],
        schema=["device_id", "minute_idx", "is_anomaly"],
    )
    detected_pdf = pd.DataFrame([
        {"device_id": 0, "minute_idx": 0, "is_anomaly": True, "is_detected_anomaly": True},
        {"device_id": 0, "minute_idx": 2, "is_anomaly": False, "is_detected_anomaly": True},
    ])

    result = evaluate_detection(full_df, detected_pdf)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 2
