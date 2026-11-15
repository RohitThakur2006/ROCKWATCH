"""
forecaster_agent.py - Calibrated Horizon Risk Forecaster Agent.

Role:
Computes zone-level rockfall probability over a 6-hour horizon,
modulates risk via approved sensitivity proposals, and quantifies sensor uncertainty bounds.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.forecast_tools import compute_zone_risk, create_provenance_record


class ForecasterAgent(BaseAgent):
    """Agent 4: Calibrated Horizon Risk Forecaster."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_FORECASTER",
            role="Calibrated Risk Forecaster & Uncertainty Assessor",
            plugin_name="ForecasterPlugin",
        )

    @kernel_function(name="estimate_risk", description="Estimates 6-hour rockfall probability and uncertainty interval.")
    def estimate_risk(
        self,
        zone_id: str,
        velocity_mm_hr: float,
        rainfall_3h_mm: float = 0.0,
        vibration_ppv: float = 0.0,
        sensitivity_multiplier: float = 1.0,
        sensor_uncertainty_penalty: float = 0.05,
    ) -> Dict[str, Any]:
        return compute_zone_risk(
            zone_id,
            velocity_mm_hr,
            rainfall_3h_mm,
            vibration_ppv,
            sensitivity_multiplier,
            sensor_uncertainty_penalty,
        )

    @kernel_function(name="log_provenance", description="Creates tamper-evident provenance audit record.")
    def log_provenance(self, zone_id: str, risk_payload: Dict[str, Any]) -> Dict[str, Any]:
        return create_provenance_record(zone_id, risk_payload)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'clean_telemetry' and 'sensor_health', queries fail-closed sensitivity,
        invokes tools, and writes 'forecast' to the shared blackboard.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        clean_telemetry = blackboard.read("clean_telemetry").get(zone_id, {})
        sensor_health = blackboard.read("sensor_health").get(zone_id, {})

        v = clean_telemetry.get("effective_velocity_mm_hr", 0.0)
        rain = clean_telemetry.get("rainfall_3h_mm", 0.0)
        vib = clean_telemetry.get("vibration_ppv_mms", 0.0)
        penalty = sensor_health.get("uncertainty_penalty", 0.05)

        # Fail-closed sensitivity lookup: reverts to baseline (1.0) on expiry or error
        current_time = context.get("timestamp")
        active_sensitivity = blackboard.get_effective_forecaster_sensitivity(zone_id, current_time)

        # Tool 1: Compute calibrated horizon risk
        risk_data = self.estimate_risk(zone_id, v, rain, vib, active_sensitivity, penalty)

        # Tool 2: Build provenance record
        provenance = self.log_provenance(zone_id, risk_data)

        combined_forecast = {
            **risk_data,
            "provenance": provenance,
        }

        # Explicit Blackboard Write
        blackboard.write("forecast", {zone_id: combined_forecast}, self.agent_id)

        return combined_forecast
