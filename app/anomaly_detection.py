"""Two-stage anomaly detection: a cheap, fully distributed z-score pass
over the whole telemetry stream in Spark narrows millions of rows down to
a small candidate set, then a scikit-learn Isolation Forest -- trained on
a sample of confirmed-normal rows, not on the candidates themselves --
scores those candidates against what normal actually looks like.

The Isolation Forest must be fit on normal data, not on the candidate set:
fitting it on the candidates alone would make it look for outliers among
rows that are mostly already anomalies, treating the anomalies as the
normal majority and rejecting genuine detections -- this exact bug showed
up as a collapsed recall in the end-to-end pipeline test before the fix."""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import IsolationForest

from app.features import METRICS

# A textbook threshold of 3.0 sounds right for a z-score but was measured
# (via a threshold sweep against the synthetic ground truth) to pass far
# too much normal-row noise as candidates, swamping precision; 5.0 was the
# measured point where precision becomes reasonable without giving up
# recall, since genuine injected faults score 60-100+ and are nowhere
# near this boundary either way.
Z_SCORE_THRESHOLD = 5.0
Z_SCORE_COLUMNS = [f"{metric}_zscore" for metric in METRICS]


def flag_candidates_by_zscore(df: DataFrame, threshold: float = Z_SCORE_THRESHOLD) -> DataFrame:
    """Flags a row as a candidate if any metric's rolling z-score exceeds
    the threshold in magnitude."""
    condition = None
    for col in Z_SCORE_COLUMNS:
        c = F.abs(F.col(col)) >= threshold
        condition = c if condition is None else (condition | c)
    return df.withColumn("zscore_candidate", condition)


def fit_normal_baseline_model(normal_pdf: pd.DataFrame, seed: int = 0) -> IsolationForest:
    """Fits an Isolation Forest on a sample of rows that were NOT flagged
    as z-score candidates, so it learns what normal actually looks like."""
    model = IsolationForest(contamination="auto", random_state=seed)
    model.fit(normal_pdf[Z_SCORE_COLUMNS])
    return model


def refine_with_isolation_forest(model: IsolationForest, candidates_pdf: pd.DataFrame) -> pd.DataFrame:
    """Scores each candidate against the normal-baseline model; keeps the
    z-score candidates whose feature vector also looks like an outlier
    relative to normal behavior, not relative to other candidates."""
    result = candidates_pdf.copy()
    if result.empty:
        result["is_detected_anomaly"] = pd.Series(dtype=bool)
        return result

    predictions = model.predict(result[Z_SCORE_COLUMNS])
    result["is_detected_anomaly"] = predictions == -1
    return result


def detect_anomalies(
    df: DataFrame, z_threshold: float = Z_SCORE_THRESHOLD, baseline_sample_fraction: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Full pipeline: Spark narrows the stream down to z-score candidates
    and separately samples confirmed-normal rows for a baseline, then
    Isolation Forest scores the candidates against that baseline."""
    flagged = flag_candidates_by_zscore(df, z_threshold)
    candidates = flagged.filter(F.col("zscore_candidate")).toPandas()

    if candidates.empty:
        candidates["is_detected_anomaly"] = pd.Series(dtype=bool)
        return candidates

    normal_sample = (
        flagged.filter(~F.col("zscore_candidate"))
        .sample(fraction=baseline_sample_fraction, seed=seed)
        .toPandas()
    )
    if len(normal_sample) < 10:
        candidates["is_detected_anomaly"] = True
        return candidates

    model = fit_normal_baseline_model(normal_sample, seed=seed)
    return refine_with_isolation_forest(model, candidates)
