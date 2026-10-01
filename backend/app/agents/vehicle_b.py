"""B차량측 에이전트 — B차량 과실 최소화 근거 논증 (공통 로직은 AdvocateAgent)."""
from __future__ import annotations

from app.agents.base import AdvocateAgent


class VehicleBAgent(AdvocateAgent):
    name = "vehicle_b_agent"
    side = "B"
