"""사용자용 설명을 쓰는 별도 호출 — 내부 정보(위원·사실검증)를 주지 않는 깨끗한 프롬프트."""
import asyncio

from app.core.llm_client import LLMClient
from app.core.pipeline import FaultAssessmentPipeline
from app.core.user_summary import (
    USER_SUMMARY_PROMPT,
    basis_for,
    build_prompt,
    reviewers_range,
    unconfirmed_facts,
    write_user_summary,
)
from app.core.user_text import chart_basis_entry
from app.models.schemas import AccidentInput, FaultReport, ReviewScore

DESC = "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다"
B = [chart_basis_entry("차2-2", 0, "A")]


class Spy(LLMClient):
    def __init__(self, reply="내 차량의 과실은 0%로 보았습니다.", fail=False, mock=False):
        super().__init__()
        self.reply, self.fail, self._m, self.calls = reply, fail, mock, []

    @property
    def mock_mode(self):
        return self._m

    async def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


def review(name, fault, assumed=()):
    return ReviewScore(reviewer_name=name, perspective=name, fault_ratio_a=fault, fault_ratio_b=100 - fault, rationale=f"{name} 근거",
                       chart_no="차2-2", chart_a_is="A", scene_a_fault=fault, confidence="high", assumed_facts=list(assumed))


def report(reviews):
    return FaultReport(fault_ratio_a=reviews[0].fault_ratio_a, fault_ratio_b=100 - reviews[0].fault_ratio_a, basis="b",
                       deviation=0.0, needs_precise_review=False, reviews=reviews)


# ---------- 프롬프트: 내부 정보 없음 ----------
def test_prompt_has_only_scene_chart_range_and_unconfirmed_facts():
    p = build_prompt("사고 상황 설명: 직진 중 충돌", B, 0, 10, unconfirmed=["상대 방향 신호가 녹색이었다"])
    for must in ("[사고 정황]", "[적용한 기준표]", "차2-2", "기본 과실비율: 나 0 : 상대 100", "[확인되지 않은 사실]", "- 상대 방향 신호가 녹색이었다", "나의 과실 0~10%", "상대 90~100%"):
        assert must in p
    for internal in ("위원", "사실검증", "인용 오류", "확신", "플래그", "reviewer"):
        assert internal not in p
    assert "[사용자 진술]" not in p and "[확인되지 않은 사실]" in p


def test_prompt_with_statement_and_without_unconfirmed():
    p = build_prompt("정황", B, 10, 10, statement="사용자 진술 기준: 질문 → 녹색 신호였다")
    assert "[사용자 진술]\n사용자 진술 기준: 질문 → 녹색 신호였다" in p and "[확인되지 않은 사실]" not in p and "나의 과실 10%" in p


def test_system_prompt_rules():
    for must in ("3~5문장", "확인되지 않은 사실", "진술", "있는 값만 쓰세요", "검토 과정이나 내부 절차", "'위원'"):
        assert must in USER_SUMMARY_PROMPT


# ---------- 호출 ----------
def test_write_user_summary_returns_text_and_uses_the_clean_prompt():
    llm = Spy()
    out = asyncio.run(write_user_summary(llm, "정황", B, 0, 0))
    assert out == "내 차량의 과실은 0%로 보았습니다." and llm.calls[0][0] == USER_SUMMARY_PROMPT


def test_mock_and_failure_return_empty_without_raising():
    mock = Spy(mock=True)
    assert asyncio.run(write_user_summary(mock, "정황", B, 0, 0)) == "" and mock.calls == []
    assert asyncio.run(write_user_summary(Spy(fail=True), "정황", B, 0, 0)) == ""


# ---------- 보조 함수 ----------
def test_unconfirmed_facts_are_deduplicated_and_range_comes_from_reviewers():
    rep = report([review("가", 0, ["신호 상태 불명", " 신호 상태 불명 "]), review("나", 10, ["신호 상태 불명", "속도 불명", ""]), review("다", 0)])
    assert unconfirmed_facts(rep) == ["신호 상태 불명", "속도 불명"]
    assert reviewers_range(rep) == (0, 10) and [b.chart_no for b in basis_for(rep, None)] == ["차2-2"]


# ---------- 파이프라인 ----------
def test_pipeline_writes_the_summary_only_when_no_question_is_needed(monkeypatch):
    import app.core.pipeline as pl

    calls = []

    async def fake_write(llm, accident_text, basis, low, high, **kw):
        calls.append((accident_text, [b.chart_no for b in basis], low, high, kw))
        return "별도 호출로 쓴 설명"

    monkeypatch.setattr(pl, "write_user_summary", fake_write)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.user_summary == "별도 호출로 쓴 설명" and len(calls) == 1
    assert DESC in calls[0][0] and calls[0][2] == calls[0][3] == 50 and "unconfirmed" in calls[0][4]    # mock 위원 50 (가정한 사실 없음)

    async def question_plan(llm, context):
        from tests.test_clarify import question

        return [question()], "", []

    calls.clear()
    monkeypatch.setattr(pl, "plan_clarification", question_plan)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.needs_clarification and r.user_summary == "" and calls == []                       # 질문이 필요하면 답변 뒤에 쓴다
