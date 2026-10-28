"""A small, deterministic multi-step investigation agent: given one
detected anomaly, it checks whether other devices are anomalous in the
same time window (isolated fault vs. network-wide issue), matches the
metric signature against a rule set of known fault patterns, and proposes
a remediation action -- the root-causing and remediation step the posting
describes, built as an auditable tool-calling pipeline rather than an
opaque model."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

Z_SIGNATURE_THRESHOLD = 3.0

REMEDIATIONS = {
    "link_congestion_or_physical_layer": "Escalate to network operations for physical link inspection and failover the affected path if available.",
    "routing_or_queuing_delay": "Check routing table changes and queuing configuration on the affected path; consider traffic rebalancing.",
    "device_hardware_fault": "Flag the device for hardware diagnostics and restart the affected interface as a first mitigation.",
    "bandwidth_exhaustion": "Review recent traffic growth on the link and consider capacity upgrade or rate limiting noisy clients.",
    "undetermined": "Insufficient signal to automatically diagnose; route to a network engineer for manual review.",
}


@dataclass
class RootCauseReport:
    device_id: int
    minute_idx: int
    scope: str
    likely_cause: str
    suggested_remediation: str
    co_occurring_devices: list[int]


def find_co_occurring_anomalies(
    detected_pdf: pd.DataFrame, device_id: int, minute_idx: int, window_minutes: int = 5
) -> list[int]:
    """Other devices flagged as anomalous within the same time window,
    used to distinguish an isolated device fault from a shared network
    issue affecting several devices at once."""
    nearby = detected_pdf[
        (detected_pdf["device_id"] != device_id)
        & detected_pdf["is_detected_anomaly"]
        & (detected_pdf["minute_idx"] >= minute_idx - window_minutes)
        & (detected_pdf["minute_idx"] <= minute_idx + window_minutes)
    ]
    return sorted(nearby["device_id"].unique().tolist())


def identify_likely_cause(metrics_snapshot: dict) -> str:
    """Matches a row's z-score signature against known fault patterns."""
    latency_z = metrics_snapshot.get("latency_ms_zscore", 0.0)
    bandwidth_z = metrics_snapshot.get("bandwidth_mbps_zscore", 0.0)
    packet_loss_z = metrics_snapshot.get("packet_loss_pct_zscore", 0.0)
    error_z = metrics_snapshot.get("error_count_zscore", 0.0)

    if error_z >= Z_SIGNATURE_THRESHOLD and latency_z >= Z_SIGNATURE_THRESHOLD:
        return "device_hardware_fault"
    if packet_loss_z >= Z_SIGNATURE_THRESHOLD and bandwidth_z <= -Z_SIGNATURE_THRESHOLD:
        return "link_congestion_or_physical_layer"
    if bandwidth_z <= -Z_SIGNATURE_THRESHOLD:
        return "bandwidth_exhaustion"
    if latency_z >= Z_SIGNATURE_THRESHOLD:
        return "routing_or_queuing_delay"
    return "undetermined"


def suggest_remediation(likely_cause: str) -> str:
    return REMEDIATIONS.get(likely_cause, REMEDIATIONS["undetermined"])


def investigate(detected_pdf: pd.DataFrame, device_id: int, minute_idx: int) -> RootCauseReport:
    row_matches = detected_pdf[
        (detected_pdf["device_id"] == device_id) & (detected_pdf["minute_idx"] == minute_idx)
    ]
    if row_matches.empty:
        raise ValueError(f"no detected row for device {device_id} at minute {minute_idx}")
    row = row_matches.iloc[0]

    co_occurring = find_co_occurring_anomalies(detected_pdf, device_id, minute_idx)
    scope = "network_wide" if co_occurring else "isolated"
    likely_cause = identify_likely_cause(row.to_dict())
    remediation = suggest_remediation(likely_cause)

    return RootCauseReport(
        device_id=device_id,
        minute_idx=minute_idx,
        scope=scope,
        likely_cause=likely_cause,
        suggested_remediation=remediation,
        co_occurring_devices=co_occurring,
    )
