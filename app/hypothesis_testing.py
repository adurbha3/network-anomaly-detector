"""Formal hypothesis testing: confirms that a detected anomaly window is
actually statistically distinct from normal baseline behavior for a given
metric, rather than trusting the detector's label alone."""
from __future__ import annotations

import numpy as np
from scipy import stats


def compare_anomaly_vs_normal(anomaly_values: list[float] | np.ndarray, normal_values: list[float] | np.ndarray) -> dict:
    """Two-sample Mann-Whitney U test (no assumption of normal residuals,
    appropriate for skewed metrics like packet loss or error counts).
    Returns the test statistic, the p-value, and whether the difference is
    significant at alpha = 0.05."""
    anomaly_values = np.asarray(anomaly_values, dtype=float)
    normal_values = np.asarray(normal_values, dtype=float)

    if len(anomaly_values) < 2 or len(normal_values) < 2:
        return {"statistic": None, "p_value": None, "significant": False}

    statistic, p_value = stats.mannwhitneyu(anomaly_values, normal_values, alternative="two-sided")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
