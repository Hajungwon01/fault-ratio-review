"""진행 에이전트(오케스트레이터) — 라운드 진행 관리 + 위원 의견 취합 + 심사 리포트 생성.

이 에이전트 자체는 "판정"을 내리지 않고, 검토위원 3인의 결과를 집계해
과실비율 평균과 편차를 계산하고, 편차가 크면 정밀 심사가 필요하다고 플래그만 남긴다.
실제 라운드 순서 제어(누구를 언제 부를지)는 core/pipeline.py 가 담당하고,
이 클래스는 "이미 다 끝난 결과를 받아서 리포트로 정리하는" 마지막 단계만 맡는다.

TODO(8주차): 편차 계산 방식(표준편차 vs 최대-최소 등)과 임계값을 확정.
"""
from __future__ import annotations

import statistics
from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import FaultReport

# TODO(8주차): 실측 테스트 후 적절한 임계값으로 조정
DEVIATION_THRESHOLD = 15.0


class ProceedingAgent(BaseAgent):
    name = "proceeding_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 검토위원 3인의 평가를 취합해 최종 과실비율 심사 리포트를 작성하는 진행 에이전트입니다.\n"
            "직접 판정을 내리지 않고, 위원들의 평가를 집계·요약하는 역할만 합니다."
        )

    async def run(self, context: dict[str, Any]) -> FaultReport:
        reviews = context["reviews"]  # list[ReviewScore]
        ratios_a = [r.fault_ratio_a for r in reviews]

        avg_a = round(statistics.mean(ratios_a))
        avg_b = 100 - avg_a
        deviation = round(statistics.pstdev(ratios_a), 1) if len(ratios_a) > 1 else 0.0

        basis = await self.llm.complete(
            self.system_prompt,
            f"검토위원 평가: {reviews}\n평균 A과실: {avg_a}, 편차: {deviation}",
        )

        return FaultReport(
            fault_ratio_a=avg_a,
            fault_ratio_b=avg_b,
            basis=basis,
            deviation=deviation,
            needs_precise_review=deviation > DEVIATION_THRESHOLD,
            rounds=context.get("rounds", []),
            fact_check=context.get("fact_check"),
            reviews=reviews,
        )
