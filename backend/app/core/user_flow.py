"""사용자용 흐름: 파이프라인 결과(내부 FaultReport)를 사용자에게 안전한 응답으로 바꾼다.

원칙(docs/USER_API_DESIGN.md):
- 질문 단계에서는 질문 문구와 보기 라벨만 내보낸다 (도표번호·과실·증거·플래그 없음).
- 결과는 단정하지 않고 범위로, 질문에 답했다면 "사용자 진술 기준"으로 표기하고 한계 고지를 붙인다.
- 답변은 세션당 1회.
"""
from __future__ import annotations

from app.core.chart_lookup import get_catalog
from app.core.clarify import UNKNOWN_ID, ClarifiedResult, resolve, user_view
from app.core.pipeline import FaultAssessmentPipeline
from app.core.sessions import SessionStore
from app.models.schemas import AccidentInput, FaultReport
from app.models.user_schemas import AssessResponse, ChartRef, UserOption, UserQuestion, UserResult

DISCLAIMER = (
    "이 결과는 참고용이며 법적 효력이 없습니다. 최종 판단은 보험사, 과실비율분쟁심의위원회, 법원에 있습니다."
)
NOTICE = "정확한 검토를 위한 질문입니다. 답변은 사실로 확인되지 않은 진술로 취급되며, 모르면 '모름'을 고르셔도 됩니다."


class InvalidOption(Exception):
    pass


def _chart_refs(chart_nos: list[str]) -> list[ChartRef]:
    catalog = get_catalog()
    refs, seen = [], set()
    for no in chart_nos:
        if no and no not in seen and (chart := catalog.get(no)):
            seen.add(no)
            refs.append(ChartRef(chart_no=no, label=chart["label"]))
    return refs


def _explanation(report: FaultReport) -> str:
    """사용자에게 보여 줄 근거: 기준표 적용 위원의 근거를 우선 쓴다 (없으면 첫 위원)."""
    if not report.reviews:
        return report.basis
    pick = next((r for r in report.reviews if r.perspective == "기준표 적용"), report.reviews[0])
    return pick.rationale


def build_result(report: FaultReport, question=None, resolved: ClarifiedResult | None = None) -> UserResult:
    """질문 없음: 위원 3명의 최소~최대 / 보기를 고름: 그 보기의 값 / 모름: 모든 보기의 범위."""
    if resolved is not None:
        low, high = resolved.scene_a_low, resolved.scene_a_high
        assumption, evidence = resolved.statement, [resolved.evidence] if resolved.evidence else []
        chart_nos = [resolved.chart_no] if resolved.chart_no else [r.chart_no for r in report.reviews]
    else:
        faults = [r.fault_ratio_a for r in report.reviews] or [report.fault_ratio_a]
        low, high = min(faults), max(faults)
        assumption, evidence = "", []
        chart_nos = [r.chart_no for r in report.reviews]
    return UserResult(
        my_fault_low=low, my_fault_high=high, other_fault_low=100 - high, other_fault_high=100 - low,
        assumption=assumption,
        explanation=_explanation(report),
        charts=_chart_refs(chart_nos),
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
            return AssessResponse(status="done", result=build_result(report))

        question = report.clarifications[0]
        view = user_view(question)
        session_id = self.store.create(report, question, view["seed"])
        return AssessResponse(
            status="needs_answer",
            session_id=session_id,
            expires_in_sec=self.store.ttl_sec,
            question=UserQuestion(
                seed=view["seed"], text=view["question"], options=[UserOption(**o) for o in view["options"]]
            ),
            notice=NOTICE,
        )

    def answer(self, session_id: str, option_id: int | None, seed: int) -> AssessResponse:
        session = self.store.get(session_id)  # SessionNotFound / AlreadyAnswered
        n = len(session.question.options)
        if seed != session.seed or (option_id is not None and option_id != UNKNOWN_ID and not 0 <= option_id < n):
            raise InvalidOption()
        resolved = resolve(session.question, option_id, session.seed)
        result = build_result(session.report, session.question, resolved)
        self.store.mark_answered(session_id)
        return AssessResponse(status="done", result=result)
