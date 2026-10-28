from app.hypothesis_testing import compare_anomaly_vs_normal


def test_detects_significant_difference_between_clearly_different_groups():
    anomaly_values = [50, 55, 52, 58, 60, 53, 57]
    normal_values = [10, 12, 9, 11, 10, 13, 8]
    result = compare_anomaly_vs_normal(anomaly_values, normal_values)
    assert result["significant"] is True
    assert result["p_value"] < 0.05


def test_does_not_flag_identical_distributions_as_significant():
    values = [10, 11, 9, 12, 10, 11, 9, 10]
    result = compare_anomaly_vs_normal(values, values)
    assert result["significant"] is False


def test_handles_too_few_samples_gracefully():
    result = compare_anomaly_vs_normal([1.0], [2.0])
    assert result["p_value"] is None
    assert result["significant"] is False
