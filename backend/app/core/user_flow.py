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

from app.core.clarify import UNKNOWN_ID, ClarifiedResult, resolve, user_view
from app.core.pipeline import FaultAssessmentPipeline
from app.core.sessions import Session, SessionStore
from app.core.user_text import UNKNOWN_SUMMARY, choose_summary, chart_basis_entry, unique_basis
from app.models.schemas import AccidentInput, FaultReport
from app.models.user_schemas import AssessResponse, ChartBasis, UserOption, UserQuestion, UserResult

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "이 결과는 참고용이며 법적 효력이 없습니다. 최종 판단은 보험사, 과실비율분쟁심의위원회, 법원에 있습니다."
)
NOTICE = "정확한 검토를 위한 질문입니다. 답변은 사실로 확인되지 않은 진술로 취급되며, 모르면 '모름'을 고르셔도 됩니다."

ANSWER_SUMMARY_PROMPT = (
    "당신은 교통사고 과실비율 확인 서비스에서 사고 당사자에게 결과를 설명하는 도우미입니다. 판정을 바꾸지 말고 주어진 결과를 쉽게 풀어 쓰세요.\n\n"
    "[규칙]\n"
    "- 3~5문장의 한국어 문단 하나.\n"
    "- 사용자의 답변은 확인된 사실이 아니라 진술입니다. \"고르신 답변을 기준으로\"처럼 진술 기준임을 밝히고 사실처럼 단정하지 마세요.\n"
    "- 이 정황에 [적용한 기준표]가 왜 맞는지 정황의 내용과 연결해 설명하세요.\n"
    "- 숫자는 [적용한 기준표]와 [최종 결론]에 있는 값만 쓰세요. 도표번호도 주어진 것만 쓰세요. 판례나 법조항을 새로 만들어 쓰지 마세요.\n"
    "- '내 차량', '상대 차량'으로 쓰고 '위원', '정황 A/B', '도표 A/B', '확신', '플래그', '가정' 같은 내부 용어를 쓰지 마세요."
)


class InvalidOption(Exception):
    pass


def _range_text(low: int, high: int) -> str:
    return f"{low}%" if low == high else f"{low}~{high}%"


def basis_for(report: FaultReport, resolved: ClarifiedResult | None) -> list[ChartBasis]:
    """도표 근거 블록. 모름이면 비운다(보기별 결과 비공개), 답변함이면 고른 도표만, 질문 없음이면 위원이 적용한 도표."""
    if resolved is not None:
        if resolved.chart_no is None:
            return []
        return unique_basis([chart_basis_entry(resolved.chart_no, resolved.scene_a_low)])
    return unique_basis(chart_basis_entry(r.chart_no, report.fault_ratio_a, r.chart_a_is) for r in report.reviews if r.chart_no)


def build_result(report: FaultReport, resolved: ClarifiedResult | None = None, model_summary: str = "") -> UserResult:
    """질문 없음: 위원 3명의 최소~최대 / 보기를 고름: 그 보기의 값 / 모름: 모든 보기의 범위.

    model_summary 는 모델이 쓴 설명 후보 — 검증을 통과하면 쓰고 아니면 코드 문장을 쓴다 (모름은 항상 고정 문장).
    """
    if resolved is not None:
        low, high = resolved.scene_a_low, resolved.scene_a_high
        assumption, evidence = resolved.statement, [resolved.evidence] if resolved.evidence else []
    else:
        faults = [r.fault_ratio_a for r in report.reviews] or [report.fault_ratio_a]
        low, high = min(faults), max(faults)
        assumption, evidence = "", []

    basis = basis_for(report, resolved)
    if resolved is not None and resolved.chart_no is None:
        summary = UNKNOWN_SUMMARY
    else:
        summary = choose_summary(model_summary, basis, low, high, answered=resolved is not None)

    return UserResult(
        my_fault_low=low, my_fault_high=high, other_fault_low=100 - high, other_fault_high=100 - low,
        assumption=assumption,
        summary=summary,
        chart_basis=basis,
        evidence_to_check=evidence,
        other_possibilities=report.clarification_note,
        uncertain=report.needs_precise_review,
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
        if seed != session.seed or (option_id is not None and option_id != UNKNOWN_ID and not 0 <= option_id < n):
            raise InvalidOption()
        resolved = resolve(session.question, option_id, session.seed)
        model_summary = await self._answer_summary(session, resolved)
        result = build_result(session.report, resolved, model_summary)
        self.store.mark_answered(session_id)
        return AssessResponse(status="done", result=result)

    async def _answer_summary(self, session: Session, resolved: ClarifiedResult) -> str:
        """답변을 반영한 설명을 모델에게 한 번 쓰게 한다. 모름·mock·실패면 빈 문자열(코드 문장 사용)."""
        if resolved.chart_no is None or self.pipeline.llm.mock_mode:
            return ""
        basis = basis_for(session.report, resolved)
        if not basis:
            return ""
        b = basis[0]
        laws = " / ".join(b.laws) or "(없음)"
        user_prompt = (
            f"[사고 정황]\n{session.accident_text}\n\n"
            f"[사용자 진술]\n{resolved.statement}\n\n"
            f"[적용한 기준표]\n{b.chart_no} {b.label}\n도표 상황: {b.situation}\n"
            f"기본 과실비율: 나 {b.my_base} : 상대 {b.other_base}\n관련 법규: {laws}\n\n"
            f"[최종 결론]\n나의 과실 {_range_text(resolved.scene_a_low, resolved.scene_a_high)}, "
            f"상대 {_range_text(100 - resolved.scene_a_high, 100 - resolved.scene_a_low)}\n"
        )
        try:
            return await self.pipeline.llm.complete(ANSWER_SUMMARY_PROMPT, user_prompt)
        except Exception as e:  # noqa: BLE001 - 설명은 보조 산출물, 실패하면 코드 문장
            logger.warning("답변 반영 설명 생성 실패: %s", e)
            return ""
