"""A차량측 에이전트 — A차량 과실 최소화 근거 논증 (공통 로직은 AdvocateAgent)."""
from __future__ import annotations

from app.agents.base import AdvocateAgent


class VehicleAAgent(AdvocateAgent):
    name = "vehicle_a_agent"
    side = "A"
