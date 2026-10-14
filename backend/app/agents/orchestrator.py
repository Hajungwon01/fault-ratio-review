"""진행 에이전트(오케스트레이터) — 라운드 진행 관리 + 위원 의견 취합 + 심사 리포트 생성.

이 에이전트 자체는 "판정"을 내리지 않고, 검토위원 3인의 결과를 집계해
과실비율 중앙값과 편차를 계산하고, 편차가 크면 정밀 심사가 필요하다고 플래그만 남긴다.
실제 라운드 순서 제어(누구를 언제 부를지)는 core/pipeline.py 가 담당하고,
이 클래스는 "이미 다 끝난 결과를 받아서 리포트로 정리하는" 마지막 단계만 맡는다.

TODO(8주차): 편차 계산 방식(표준편차 vs 최대-최소 등)과 임계값을 확정.
"""
from __future__ import annotations

import statistics
from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import FaultReport

# "정밀 심사 필요" 판정 기준 — 둘 중 하나라도 넘으면 플래그. 둘 다 평가셋 실측 후 조정할 초기값이다 (#13).
#
# 표준편차만으로는 위원이 3명일 때 이상치 한 명을 놓친다: 한 명이 나머지와 d점 벗어나면 표준편차는 약 0.47*d 라서
# 임계값 15 는 d >= 32점이어야 켜진다. 실제 첫 측정에서 위원 A과실이 30/0/0 (30점 차)인데 편차 14.1 로 플래그가 안 켜졌다.
# 그래서 사람이 직관적으로 이해하는 "위원 간 최대-최소 차이(spread)"를 함께 쓴다.
SPREAD_THRESHOLD = 20      # 위원 A과실의 최대-최소 차이(%p)
DEVIATION_THRESHOLD = 15.0  # 위원 A과실의 모집단 표준편차
# 위원이 숫자는 일치해도 스스로 "확신 낮음"이라고 답한 경우: 애매한 사례(예: 도표가 둘 다 가능한 D09)는 위원 숫자가 우연히 일치하면
# 위 두 기준으로는 드러나지 않았다. 몇 명 이상이 low 일 때 켤지는 초기값이며 평가셋 실측 후 조정한다 (#13).
# (위원 평가가 도표 기본 비율과 어긋나 신뢰도가 low 로 내려간 경우는 pipeline 이 별도로 플래그를 켠다)
LOW_CONFIDENCE_REVIEWERS = 2


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

        # 위원이 3명이면 한 명의 이탈값(역할 오류 등)에 평균은 크게 끌려가지만 중앙값은 영향을 받지 않는다. 2명이면 평균과 같다.
        avg_a = round(statistics.median(ratios_a))
        avg_b = 100 - avg_a
        deviation = round(statistics.pstdev(ratios_a), 1) if len(ratios_a) > 1 else 0.0
        spread = (max(ratios_a) - min(ratios_a)) if ratios_a else 0

        basis = await self.llm.complete(
            self.system_prompt,
            f"검토위원 평가: {reviews}\n중앙값 A과실: {avg_a}, 편차: {deviation}",
        )

        reasons: list[str] = []
        if spread >= SPREAD_THRESHOLD:
            reasons.append(f"위원 간 과실비율 차이가 큼 (최대-최소 {spread}%p)")
        if deviation > DEVIATION_THRESHOLD:
            reasons.append(f"위원 간 과실비율 편차가 큼 (표준편차 {deviation})")
        # 위원이 서로 다른 도표를 적용했다 = 정황으로 도표가 하나로 정해지지 않는 애매한 사고. 숫자가 우연히 일치해도 알린다.
        # (저장된 실행 64건 중 불일치는 애매 사례 D09 3건뿐이고 명백한 사례에서는 0건이었다)
        charts = sorted({r.chart_no for r in reviews if r.chart_no})
        if len(charts) > 1:
            reasons.append(f"위원이 서로 다른 도표를 적용함 ({', '.join(charts)})")
        low = [r.perspective for r in reviews if r.confidence == "low"]
        if len(low) >= LOW_CONFIDENCE_REVIEWERS:
            reasons.append(f"위원 {len(low)}명이 확신 낮음으로 답함 ({', '.join(low)})")

        return FaultReport(
            fault_ratio_a=avg_a,
            fault_ratio_b=avg_b,
            basis=basis,
            deviation=deviation,
            spread=spread,
            flag_reasons=reasons,
            needs_precise_review=bool(reasons),
            rounds=context.get("rounds", []),
            fact_check=context.get("fact_check"),
            reviews=reviews,
        )
