"""FaultAssessmentPipeline — 7개 에이전트의 라운드 흐름을 실제로 진행시키는 오케스트레이션 로직.

흐름: A/B 1차 논증(병렬) → 사실검증 → A/B 2차 반박(병렬) → 검토위원 3종(병렬) → 진행 에이전트 집계

지금은 모든 에이전트가 LLMClient의 mock 모드로 동작하므로, API 키 없이도
이 파이프라인이 끝까지 도는지 오늘(1주차) 바로 확인할 수 있다.
"""
from __future__ import annotations

import asyncio
import logging

from app.agents.fact_checker import FactCheckerAgent
from app.agents.orchestrator import ProceedingAgent, plan_clarification
from app.agents.reviewer_precedent import ReviewerPrecedentAgent
from app.agents.reviewer_scene import ReviewerSceneAgent
from app.agents.reviewer_standard import ReviewerStandardAgent
from app.agents.vehicle_a import VehicleAAgent
from app.agents.vehicle_b import VehicleBAgent
from app.core.chart_lookup import find_candidate_charts, format_candidates
from app.core.llm_client import LLMClient
from app.core.reviewer_context import (
    cited_chart_numbers,
    format_cited_charts,
    format_precedents,
)
from app.core.scope_check import OutOfScopeError, check_scope
from app.models.schemas import AccidentInput, FaultReport

logger = logging.getLogger(__name__)

# 독립 평가가 이보다 적으면 편차·합의의 의미가 없어 심사를 중단한다
MIN_REVIEWS = 2


class FaultAssessmentPipeline:
    def __init__(self) -> None:
        llm = LLMClient()  # 에이전트들이 LLMClient 인스턴스를 공유
        self.llm = llm  # 평가 도구가 파이프라인 1회 실행의 호출 수·토큰(llm.usage)을 읽는다
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
        candidates = find_candidate_charts(accident_input)
        reference_charts = format_candidates(candidates)
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
        # 위원에게는 A/B 가 실제로 인용한 (존재하는) 도표의 상세와, 판례 근거 등급을 함께 준다
        cited = cited_chart_numbers(rounds)
        reviewer_context = {
            "accident_input": accident_text,
            "rounds": rounds,
            "fact_check": fact_check,
            "cited_charts": format_cited_charts(cited),
            "cited_chart_numbers": list(cited),
            # 질문 단계가 보기에 쓸 수 있는 도표: 인용된 도표 + 검색 후보 (후보의 상세 내용도 함께 준다)
            "candidate_chart_numbers": [m.chart_no for m in candidates],
            "reference_charts": reference_charts,
            "precedents": format_precedents(cited),
        }
        results = await asyncio.gather(*(r.run(reviewer_context) for r in self.reviewers), return_exceptions=True)
        reviews = []
        for reviewer, res in zip(self.reviewers, results):
            if isinstance(res, BaseException) and not isinstance(res, Exception):
                raise res  # CancelledError 등은 삼키지 않는다
            if isinstance(res, Exception):
                logger.error("검토위원 %s 평가 실패: %s", reviewer.name, res)
                warnings.append(f"검토위원 '{reviewer.perspective}' 평가에 실패해 제외했습니다 ({type(res).__name__}).")
            else:
                reviews.append(res)
        if len(reviews) < MIN_REVIEWS:
            raise RuntimeError(f"유효한 검토위원 평가가 {len(reviews)}건뿐이라 심사를 완료할 수 없습니다 (최소 {MIN_REVIEWS}건)")

        # 진행 에이전트가 취합해 최종 리포트 생성
        report = await self.proceeding.run(
            {"reviews": reviews, "rounds": rounds, "fact_check": fact_check}
        )
        if len(reviews) < len(self.reviewers):
            # 독립 평가가 줄어 신뢰도가 낮으므로 보수적으로 정밀 심사 대상으로 표시한다
            report.needs_precise_review = True
            report.flag_reasons.append(f"유효한 위원 평가가 {len(reviews)}건뿐임 (정상 {len(self.reviewers)}건)")
        # 위원의 숫자가 도표 기본 비율과 앞뒤가 안 맞으면(값/역할 오류 의심) 위원 3명이 일치해도 정밀 심사 대상으로 표시한다
        for rv in reviews:
            for w in rv.warnings:
                warnings.append(f"위원 '{rv.perspective}' 평가 의심: {w}")
        if any(rv.warnings for rv in reviews):
            report.needs_precise_review = True
            report.flag_reasons.append("위원 평가가 도표 기본 비율 등과 어긋남 (경고 참고)")
        # 질문 단계: 정황에 없는 사실 때문에 결과가 크게 갈리면 사용자에게 물을 질문을 만든다 (실패해도 심사 결과는 유지)
        try:
            trace: list[str] = []
            questions, note, q_warnings = await plan_clarification(
                self.llm, {**reviewer_context, "reviews": reviews, "trace": trace}
            )
            report.clarification_trace = " | ".join(trace)
            report.clarifications = questions
            report.needs_clarification = bool(questions)
            report.clarification_note = note
            warnings += q_warnings
        except Exception as e:  # noqa: BLE001 - 보조 단계라 실패해도 본 결과를 막지 않는다
            logger.warning("질문 생성 실패: %s", e)
            warnings.append(f"질문 단계를 건너뛰었습니다 ({type(e).__name__}).")
        report.warnings = warnings
        return report
