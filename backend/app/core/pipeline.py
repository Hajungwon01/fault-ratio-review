"""FaultAssessmentPipeline — 7개 에이전트의 라운드 흐름을 실제로 진행시키는 오케스트레이션 로직.

흐름: A/B 1차 논증(병렬) → 사실검증 → A/B 2차 반박(병렬) → 검토위원 3종(병렬) → 진행 에이전트 집계

지금은 모든 에이전트가 LLMClient의 mock 모드로 동작하므로, API 키 없이도
이 파이프라인이 끝까지 도는지 오늘(1주차) 바로 확인할 수 있다.
"""
from __future__ import annotations

import asyncio

from app.agents.fact_checker import FactCheckerAgent
from app.agents.orchestrator import ProceedingAgent
from app.agents.reviewer_precedent import ReviewerPrecedentAgent
from app.agents.reviewer_scene import ReviewerSceneAgent
from app.agents.reviewer_standard import ReviewerStandardAgent
from app.agents.vehicle_a import VehicleAAgent
from app.agents.vehicle_b import VehicleBAgent
from app.core.chart_lookup import find_candidate_charts, format_candidates
from app.core.llm_client import LLMClient
from app.core.scope_check import OutOfScopeError, check_scope
from app.models.schemas import AccidentInput, FaultReport


class FaultAssessmentPipeline:
    def __init__(self) -> None:
        llm = LLMClient()  # 에이전트들이 LLMClient 인스턴스를 공유
        self.vehicle_a = VehicleAAgent(llm)
        self.vehicle_b = VehicleBAgent(llm)
        self.fact_checker = FactCheckerAgent(llm)
        self.reviewers = [
            ReviewerStandardAgent(llm),
            ReviewerSceneAgent(llm),
            ReviewerPrecedentAgent(llm),
        ]
        self.proceeding = ProceedingAgent(llm)

    async def run(self, accident_input: AccidentInput) -> FaultReport:
        # 지원 범위 밖(보행자/자전거/PM/단독사고)이면 LLM 호출 전에 중단한다. uncertain 은 경고만 남기고 진행.
        scope = check_scope(accident_input)
        if scope.status == "unsupported":
            raise OutOfScopeError(scope)
        warnings = [scope.message] if scope.status == "uncertain" else []

        rounds = []
        # 사고 정황으로 후보 기준표를 한 번 조회해 논증/검증 에이전트에 공통 주입
        reference_charts = format_candidates(find_candidate_charts(accident_input))
        # 에이전트에는 파이썬 repr 이 아니라 읽기 쉬운 텍스트를 준다
        accident_text = accident_input.to_prompt_text()
        base_ctx = {"accident_input": accident_text, "reference_charts": reference_charts}

        # 1라운드: A/B 병렬 1차 논증
        arg_a1, arg_b1 = await asyncio.gather(
            self.vehicle_a.run({**base_ctx, "round": 1, "opponent_argument": None}),
            self.vehicle_b.run({**base_ctx, "round": 1, "opponent_argument": None}),
        )
        rounds += [arg_a1, arg_b1]

        # 사실검증
        fact_check = await self.fact_checker.run(
            {**base_ctx, "argument_a": arg_a1.content, "argument_b": arg_b1.content}
        )

        # 2라운드: 서로의 1차 주장을 받아 반박 (병렬)
        arg_a2, arg_b2 = await asyncio.gather(
            self.vehicle_a.run(
                {**base_ctx, "round": 2, "opponent_argument": arg_b1.content, "fact_check": fact_check}
            ),
            self.vehicle_b.run(
                {**base_ctx, "round": 2, "opponent_argument": arg_a1.content, "fact_check": fact_check}
            ),
        )
        rounds += [arg_a2, arg_b2]

        # 검토위원 3종 병렬 평가
        reviewer_context = {
            "accident_input": accident_text,
            "rounds": rounds,
            "fact_check": fact_check,
        }
        reviews = await asyncio.gather(*(r.run(reviewer_context) for r in self.reviewers))

        # 진행 에이전트가 취합해 최종 리포트 생성
        report = await self.proceeding.run(
            {"reviews": list(reviews), "rounds": rounds, "fact_check": fact_check}
        )
        report.warnings = warnings
        return report
