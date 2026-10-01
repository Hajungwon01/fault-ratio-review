"""사실검증 에이전트 — 양측이 인용한 기준표·법조항 적용의 정확성 검증.

TODO(5주차): 인용 오류를 구조화해 FactCheckResult.flagged_citations 로 채우기.
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
            "[절차] [참조 기준표]와 대조해 각 주장의 (1) 도표번호가 실제 목록에 있는지 (2) 기본 과실비율이 맞는지 "
            "(3) 인용한 가감요소가 해당 도표에 존재하고 사고 정황에 근거가 있는지를 항목별로 확인하세요. "
            "틀린 인용은 `오류: <주체> <내용>` 형식으로 나열하세요."
        )

    async def run(self, context: dict[str, Any]) -> FactCheckResult:
        user_prompt = (
            f"A차량측 주장: {context.get('argument_a')}\n"
            f"B차량측 주장: {context.get('argument_b')}\n"
        )
        notes = await self.llm.complete(self.system_prompt, user_prompt)
        return FactCheckResult(is_valid=True, notes=notes)
