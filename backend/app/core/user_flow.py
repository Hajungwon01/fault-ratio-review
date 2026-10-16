"""사용자용 흐름: 파이프라인 결과(내부 FaultReport)를 사용자에게 안전한 응답으로 바꾼다.

원칙(docs/USER_API_DESIGN.md):
- 질문 단계에서는 질문 문구와 보기 라벨만 내보낸다 (도표번호·과실·증거·플래그 없음).
- 결과는 단정하지 않고 범위로, 질문에 답했다면 "사용자 진술 기준"으로 표기하고 한계 고지를 붙인다.
- 답변은 세션당 1회.
- 설명(§9): 사실(도표 내용·기본 비율)은 코드가 도표 데이터에서 만들고, 모델이 쓴 문장은 검증해서 실패하면 코드 문장을 쓴다.
  "모름"을 고르면 보기별 도표·결과를 보여 주지 않는다 (모름으로 먼저 확인하고 유리한 답으로 다시 시작하는 우회 방지).
"""
from __future__ import annotations

import logging

from app.core.clarify import OTHER_ID, UNKNOWN_ID, ClarifiedResult, resolve, user_view
from app.core.pipeline import FaultAssessmentPipeline
from app.core.sessions import Session, SessionStore
from app.core.user_summary import basis_for, reviewers_range, unconfirmed_facts, write_user_summary
from app.core.user_text import OTHER_SUMMARY, UNKNOWN_SUMMARY, choose_summary
from app.models.schemas import AccidentInput, FaultReport
from app.models.user_schemas import AssessResponse, ChartBasis, UserOption, UserQuestion, UserResult

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "이 결과는 참고용이며 법적 효력이 없습니다. 최종 판단은 보험사, 과실비율분쟁심의위원회, 법원에 있습니다."
)
NOTICE = "정확한 검토를 위한 질문입니다. 답변은 사실로 확인되지 않은 진술로 취급되며, 모르면 '모름'을 고르셔도 됩니다."

class InvalidOption(Exception):
    pass


def build_result(report: FaultReport, resolved: ClarifiedResult | None = None, model_summary: str = "") -> UserResult:
    """질문 없음: 위원 3명의 최소~최대 / 보기를 고름: 그 보기의 값 / 모름: 모든 보기의 범위.

    model_summary 는 모델이 쓴 설명 후보 — 검증을 통과하면 쓰고 아니면 코드 문장을 쓴다 (모름은 항상 고정 문장).
    """
    if resolved is not None:
        low, high = resolved.scene_a_low, resolved.scene_a_high
        assumption, evidence = resolved.statement, [resolved.evidence] if resolved.evidence else []
    else:
        low, high = reviewers_range(report)
        assumption, evidence = "", []

    basis = basis_for(report, resolved)
    if resolved is not None and resolved.chart_no is None:
        summary = OTHER_SUMMARY if resolved.kind == "other" else UNKNOWN_SUMMARY
    else:
        summary = choose_summary(model_summary, basis, low, high, answered=resolved is not None)

    return UserResult(
        my_fault_low=low, my_fault_high=high, other_fault_low=100 - high, other_fault_high=100 - low,
        assumption=assumption,
        summary=summary,
        chart_basis=basis,
        evidence_to_check=evidence,
        other_possibilities=report.clarification_note,
        uncertain=report.needs_precise_review or (resolved is not None and resolved.kind == "other"),
        disclaimer=DISCLAIMER,
    )


class AssessmentService:
    def __init__(self, pipeline: FaultAssessmentPipeline, store: SessionStore) -> None:
        self.pipeline = pipeline
        self.store = store

    async def start(self, accident_input: AccidentInput) -> AssessResponse:
        report = await self.pipeline.run(accident_input)  # OutOfScopeError 는 호출 측이 처리한다
        if not (report.needs_clarification and report.clarifications):
            return AssessResponse(status="done", result=build_result(report, model_summary=report.user_summary))

        question = report.clarifications[0]
        view = user_view(question)
        session_id = self.store.create(report, question, view["seed"], accident_input.to_prompt_text())
        return AssessResponse(
            status="needs_answer",
            session_id=session_id,
            expires_in_sec=self.store.ttl_sec,
            question=UserQuestion(
                seed=view["seed"], text=view["question"], options=[UserOption(**o) for o in view["options"]]
            ),
            notice=NOTICE,
        )

    async def answer(self, session_id: str, option_id: int | None, seed: int) -> AssessResponse:
        session = self.store.get(session_id)  # SessionNotFound / AlreadyAnswered
        n = len(session.question.options)
        if seed != session.seed or (option_id is not None and option_id not in (UNKNOWN_ID, OTHER_ID) and not 0 <= option_id < n):
            raise InvalidOption()
        resolved = resolve(session.question, option_id, session.seed)
        model_summary = await self._answer_summary(session, resolved)
        result = build_result(session.report, resolved, model_summary)
        self.store.mark_answered(session_id)
        return AssessResponse(status="done", result=result)

    async def _answer_summary(self, session: Session, resolved: ClarifiedResult) -> str:
        """답변을 반영한 설명을 모델에게 한 번 쓰게 한다. 모름·mock·실패면 빈 문자열(코드 문장 사용)."""
        if resolved.chart_no is None:
            return ""
        basis = basis_for(session.report, resolved)
        if not basis:
            return ""
        return await write_user_summary(
            self.pipeline.llm, session.accident_text, basis, resolved.scene_a_low, resolved.scene_a_high, statement=resolved.statement
        )
