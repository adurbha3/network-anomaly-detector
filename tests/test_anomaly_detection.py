import numpy as np
import pandas as pd

from app.anomaly_detection import (
    fit_normal_baseline_model,
    flag_candidates_by_zscore,
    refine_with_isolation_forest,
)
from app.features import METRICS


def _zscore_df(spark, zscores: dict):
    row = {"device_id": 0, "minute_idx": 0}
    for metric in METRICS:
        row[f"{metric}_zscore"] = zscores.get(metric, 0.0)
    return spark.createDataFrame([row])


def test_flag_candidates_by_zscore_flags_any_metric_over_threshold(spark):
    anomalous = _zscore_df(spark, {"latency_ms": 4.0})
    normal = _zscore_df(spark, {"latency_ms": 0.5})
    combined = anomalous.unionByName(normal)

    result = flag_candidates_by_zscore(combined, threshold=3.0).orderBy(
        combined["latency_ms_zscore"].desc()
    ).collect()

    assert result[0]["zscore_candidate"] is True
    assert result[1]["zscore_candidate"] is False


def test_refine_with_isolation_forest_keeps_clear_outlier_relative_to_normal_baseline():
    rng = np.random.default_rng(0)
    z_cols = [f"{m}_zscore" for m in METRICS]
    normal_pdf = pd.DataFrame(rng.normal(0, 0.5, size=(200, len(z_cols))), columns=z_cols)

    candidates = pd.DataFrame(
        [{c: 10.0 for c in z_cols}, {c: 0.4 for c in z_cols}],
        columns=z_cols,
    )

    model = fit_normal_baseline_model(normal_pdf, seed=0)
    result = refine_with_isolation_forest(model, candidates)

    assert result.iloc[0]["is_detected_anomaly"]


def test_refine_with_isolation_forest_does_not_flip_majority_logic_when_most_candidates_are_real():
    """Regression test for the bug where fitting Isolation Forest directly
    on the candidate set (mostly real anomalies) caused it to treat those
    anomalies as the normal majority and reject them."""
    rng = np.random.default_rng(1)
    z_cols = [f"{m}_zscore" for m in METRICS]
    normal_pdf = pd.DataFrame(rng.normal(0, 0.5, size=(200, len(z_cols))), columns=z_cols)

    many_real_anomalies = pd.DataFrame(
        rng.normal(8.0, 0.3, size=(20, len(z_cols))), columns=z_cols
    )

    model = fit_normal_baseline_model(normal_pdf, seed=1)
    result = refine_with_isolation_forest(model, many_real_anomalies)

    assert result["is_detected_anomaly"].mean() > 0.8


def test_refine_with_isolation_forest_handles_empty_candidates():
    z_cols = [f"{m}_zscore" for m in METRICS]
    rng = np.random.default_rng(0)
    normal_pdf = pd.DataFrame(rng.normal(0, 0.5, size=(50, len(z_cols))), columns=z_cols)
    model = fit_normal_baseline_model(normal_pdf, seed=0)

    empty = pd.DataFrame(columns=z_cols)
    empty_result = refine_with_isolation_forest(model, empty)
    assert len(empty_result) == 0
