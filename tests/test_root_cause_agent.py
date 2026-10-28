import pandas as pd
import pytest

from app.root_cause_agent import (
    find_co_occurring_anomalies,
    identify_likely_cause,
    investigate,
    suggest_remediation,
)


def test_identify_likely_cause_device_fault_signature():
    cause = identify_likely_cause({
        "latency_ms_zscore": 5.0, "error_count_zscore": 4.0,
        "bandwidth_mbps_zscore": 0.0, "packet_loss_pct_zscore": 0.0,
    })
    assert cause == "device_hardware_fault"


def test_identify_likely_cause_congestion_signature():
    cause = identify_likely_cause({
        "latency_ms_zscore": 0.5, "error_count_zscore": 0.0,
        "bandwidth_mbps_zscore": -4.0, "packet_loss_pct_zscore": 5.0,
    })
    assert cause == "link_congestion_or_physical_layer"


def test_identify_likely_cause_undetermined_for_mild_signals():
    cause = identify_likely_cause({
        "latency_ms_zscore": 0.2, "error_count_zscore": 0.1,
        "bandwidth_mbps_zscore": -0.3, "packet_loss_pct_zscore": 0.1,
    })
    assert cause == "undetermined"


def test_suggest_remediation_matches_known_cause():
    remediation = suggest_remediation("device_hardware_fault")
    assert "hardware" in remediation.lower() or "interface" in remediation.lower()


def _detected_pdf():
    return pd.DataFrame([
        {"device_id": 0, "minute_idx": 100, "is_detected_anomaly": True,
         "latency_ms_zscore": 5.0, "bandwidth_mbps_zscore": 0.0,
         "packet_loss_pct_zscore": 0.0, "error_count_zscore": 5.0},
        {"device_id": 1, "minute_idx": 102, "is_detected_anomaly": True,
         "latency_ms_zscore": 4.5, "bandwidth_mbps_zscore": -3.5,
         "packet_loss_pct_zscore": 4.0, "error_count_zscore": 0.0},
        {"device_id": 2, "minute_idx": 500, "is_detected_anomaly": True,
         "latency_ms_zscore": 4.0, "bandwidth_mbps_zscore": 0.0,
         "packet_loss_pct_zscore": 0.0, "error_count_zscore": 0.0},
    ])


def test_find_co_occurring_anomalies_detects_nearby_device():
    detected = _detected_pdf()
    co_occurring = find_co_occurring_anomalies(detected, device_id=0, minute_idx=100, window_minutes=5)
    assert co_occurring == [1]


def test_find_co_occurring_anomalies_empty_when_isolated():
    detected = _detected_pdf()
    co_occurring = find_co_occurring_anomalies(detected, device_id=2, minute_idx=500, window_minutes=5)
    assert co_occurring == []


def test_investigate_returns_full_report_for_isolated_fault():
    detected = _detected_pdf()
    report = investigate(detected, device_id=2, minute_idx=500)
    assert report.scope == "isolated"
    assert report.likely_cause == "routing_or_queuing_delay"
    assert report.co_occurring_devices == []


def test_investigate_raises_for_unknown_row():
    detected = _detected_pdf()
    with pytest.raises(ValueError):
        investigate(detected, device_id=99, minute_idx=0)
