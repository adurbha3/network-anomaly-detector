"""Runs the full pipeline end to end: generate telemetry at scale in
Spark, compute rolling event-stream features, detect anomalies in two
stages, confirm the detections are statistically real, and investigate
each one with the root cause agent."""
from __future__ import annotations

from pyspark.sql import SparkSession

from app.anomaly_detection import detect_anomalies
from app.data_generation import generate_telemetry
from app.evaluation import evaluate_detection
from app.features import METRICS, add_rolling_features
from app.hypothesis_testing import compare_anomaly_vs_normal
from app.root_cause_agent import investigate


def run(
    n_devices: int = 200,
    n_days: int = 14,
    n_anomalies: int = 150,
    seed: int = 0,
    spark: SparkSession | None = None,
) -> dict:
    """If `spark` is provided (e.g. a shared session in tests), it is used
    as-is and left running for the caller to manage. Otherwise a session is
    created here and stopped before returning."""
    owns_session = spark is None
    if spark is None:
        spark = SparkSession.builder.appName("network-anomaly-detector").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        raw = generate_telemetry(spark, n_devices=n_devices, n_days=n_days, n_anomalies=n_anomalies, seed=seed)
        featured = add_rolling_features(raw)
        detected = detect_anomalies(featured)
        metrics = evaluate_detection(featured, detected)

        confirmed_df = featured.join(
            spark.createDataFrame(detected[detected["is_detected_anomaly"]][["device_id", "minute_idx"]]),
            on=["device_id", "minute_idx"],
        )
        normal_df = featured.filter("NOT is_anomaly")
        significance = {}
        for metric in METRICS:
            anomaly_values = [r[metric] for r in confirmed_df.select(metric).collect()]
            normal_values = [r[metric] for r in normal_df.sample(fraction=0.01, seed=seed).select(metric).collect()]
            significance[metric] = compare_anomaly_vs_normal(anomaly_values, normal_values)

        reports = []
        flagged = detected[detected["is_detected_anomaly"]].head(10)
        for _, row in flagged.iterrows():
            report = investigate(detected, int(row["device_id"]), int(row["minute_idx"]))
            reports.append(report)

        return {"metrics": metrics, "significance": significance, "sample_reports": reports}
    finally:
        if owns_session:
            spark.stop()


if __name__ == "__main__":
    result = run()
    print("evaluation:", result["metrics"])
    print("sample root cause reports:")
    for report in result["sample_reports"]:
        print(" ", report)
