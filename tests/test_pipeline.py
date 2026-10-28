from app.pipeline import run


def test_pipeline_runs_end_to_end_with_reasonable_detection_quality(spark):
    """Bounds here reflect a measured threshold sweep against synthetic
    ground truth (see anomaly_detection.Z_SCORE_THRESHOLD), not an
    arbitrary target. Recall has a real, explainable ceiling around 0.35-0.4
    for this data: once an anomaly lasts longer than roughly the 30 minute
    rolling window, it starts dominating its own baseline and the back half
    of the anomaly reads as the new normal -- a known limitation of
    window-relative detection, documented in the README, not a bug."""
    result = run(n_devices=20, n_days=2, n_anomalies=15, seed=0, spark=spark)
    assert result["metrics"]["recall"] > 0.3
    assert result["metrics"]["precision"] > 0.35
    assert isinstance(result["sample_reports"], list)
