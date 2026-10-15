from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AlarmPolicy:
    confidence_threshold: float = 0.75
    consecutive_windows: int = 1
    cooldown_seconds: int = 300


class AlarmEngine:
    """把模型诊断转换为可审计告警；模型与业务告警策略保持解耦。"""

    def __init__(self, policy: AlarmPolicy) -> None:
        self.policy = policy
        self._streaks: dict[tuple[str, str], int] = {}
        self._last_alarm: dict[tuple[str, str], datetime] = {}

    def process(self, diagnosis: dict, observed_at: datetime, sample_id: str) -> dict | None:
        component = str(diagnosis.get("root_cause_component") or "unknown")
        fault_type = str(diagnosis.get("fault_type") or "unknown")
        key = (component, fault_type)
        qualifies = (
            bool(diagnosis.get("is_anomaly"))
            and bool(diagnosis.get("should_alert"))
            and float(diagnosis.get("confidence", 0.0)) >= self.policy.confidence_threshold
        )
        if not qualifies:
            self._streaks[key] = 0
            return None
        self._streaks[key] = self._streaks.get(key, 0) + 1
        if self._streaks[key] < self.policy.consecutive_windows:
            return None
        previous = self._last_alarm.get(key)
        if previous and observed_at - previous < timedelta(seconds=self.policy.cooldown_seconds):
            return None
        self._last_alarm[key] = observed_at
        return {
            "alarm_id": f"alarm-{sample_id}",
            "observed_at": observed_at.isoformat(),
            "component": component,
            "fault_category": diagnosis.get("fault_category", "unknown"),
            "fault_type": fault_type,
            "severity": diagnosis.get("severity", "major"),
            "confidence": float(diagnosis.get("confidence", 0.0)),
            "evidence": diagnosis.get("evidence", []),
            "recommendation": diagnosis.get("recommendation", ""),
            "source_sample_id": sample_id,
        }

