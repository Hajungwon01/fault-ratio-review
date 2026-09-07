"""사실검증 에이전트 — 양측이 인용한 기준표·법조항 적용의 정확성 검증.

TODO(5주차): 실제 기준표 데이터(app/data/fault_ratio_reference.json)와
대조해 인용이 맞는지 검증하는 로직/프롬프트 작성.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import FactCheckResult


class FactCheckerAgent(BaseAgent):
    name = "fact_checker_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 A/B차량측 에이전트가 인용한 과실비율 기준표 및 도로교통법 조항이 "
            "실제로 정확하게 적용되었는지 검증하는 사실검증 에이전트입니다.\n"
            "TODO: 기준표 참조 데이터와 대조하는 절차를 구체화."
        )

    async def run(self, context: dict[str, Any]) -> FactCheckResult:
        user_prompt = (
            f"A차량측 주장: {context.get('argument_a')}\n"
            f"B차량측 주장: {context.get('argument_b')}\n"
        )
        notes = await self.llm.complete(self.system_prompt, user_prompt)
        return FactCheckResult(is_valid=True, notes=notes)
