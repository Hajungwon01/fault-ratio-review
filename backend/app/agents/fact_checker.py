"""사실검증 에이전트 — 양측이 인용한 기준표·법조항 적용의 정확성 검증.

TODO(5주차): 규칙 기반(도표번호·비율 존재 여부) 사전 검증을 LLM 검증 앞에 추가.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.chart_tools import build_chart_tools
from app.models.schemas import FactCheckResult


class FactCheckerAgent(BaseAgent):
    name = "fact_checker_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 A/B차량측 에이전트가 인용한 과실비율 기준표 및 도로교통법 조항이 "
            "실제로 정확하게 적용되었는지 검증하는 사실검증 에이전트입니다.\n"
            "[절차] 각 주장이 인용한 도표번호마다 get_chart 도구로 실제 내용을 조회하고, 그 결과와 대조해 각 주장의 (1) 도표번호가 실제 목록에 있는지 (2) 기본 과실비율이 맞는지 "
            "(3) 인용한 가감요소가 해당 도표에 존재하고 사고 정황에 근거가 있는지를 항목별로 확인하세요. "
            "조회 결과와 다른 인용은 한 줄에 하나씩 `오류: <A측|B측> <내용>` 형식으로 나열하고, 오류가 없으면 `오류 없음`이라고 쓰세요."
        )

    async def run(self, context: dict[str, Any]) -> FactCheckResult:
        user_prompt = (
            f"A차량측 주장: {context.get('argument_a')}\n"
            f"B차량측 주장: {context.get('argument_b')}\n"
        )
        trace: list[str] = []
        notes = await self.llm.run_agent(self.system_prompt, user_prompt, build_chart_tools(trace))
        errors = [line.split(":", 1)[1].strip() for line in notes.splitlines() if line.strip().startswith("오류:")]
        return FactCheckResult(is_valid=not errors, notes=notes, flagged_citations=errors)
