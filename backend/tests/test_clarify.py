"""질문 단계 — 사용자에게는 결과를 숨기고(user_view), 답변은 진술로 반영하며(resolve), 질문은 결과가 크게 갈릴 때만 만든다."""
import asyncio
import json

import pytest

from app.agents.orchestrator import CLARIFY_MIN_DIFF, plan_clarification
from app.core.clarify import UNKNOWN_ID, UNKNOWN_LABEL, resolve, user_view
from app.core.llm_client import LLMClient
from app.core.pipeline import FaultAssessmentPipeline
from app.models.schemas import AccidentInput, ClarifyingQuestion, QuestionOption, ReviewScore

DESC = "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다"


def opt(label, chart, fault):
    return QuestionOption(label=label, chart_no=chart, scene_a_fault=fault)


def question(options=None):
    return ClarifyingQuestion(
        question="상대 차량이 좌회전하기 직전, 상대 방향 신호는 무엇이었나요?",
        options=options or [
            opt("녹색 직진 신호 또는 적색 신호였다", "차2-2", 0),       # 도표 A = 직진 = 나
            opt("좌회전 표지가 있는 곳에서 녹색 신호였다", "차2-6", 10),  # 도표 B = 직진 = 나
            opt("정지선 앞 점멸 신호였다", "차2-5", 50),
        ],
        evidence="블랙박스 영상, 교차로 신호 주기 기록",
    )


# ---------- 사용자에게 보이는 형태 ----------
def test_user_view_hides_charts_and_faults():
    v = user_view(question(), seed=1)
    text = json.dumps(v, ensure_ascii=False)
    for secret in ("차2-2", "차2-6", "차2-5", "chart", "scene", "fault", "evidence", "블랙박스", "50"):
        assert secret not in text
    assert set(v) == {"seed", "question", "options"}
    assert all(set(o) == {"id", "label"} for o in v["options"])


def test_user_view_always_ends_with_unknown_and_shuffles_by_seed():
    q = question()
    assert user_view(q, seed=7) == user_view(q, seed=7)           # 같은 seed 면 같은 순서
    v = user_view(q, seed=7)
    assert v["options"][-1] == {"id": UNKNOWN_ID, "label": UNKNOWN_LABEL}
    assert [o["id"] for o in v["options"][:-1]] == [0, 1, 2]      # id 는 화면 위치
    orders = {tuple(o["label"] for o in user_view(q, seed=s)["options"][:-1]) for s in range(30)}
    assert len(orders) > 1                                         # seed 에 따라 순서가 바뀐다
    assert isinstance(user_view(q)["seed"], int)                   # seed 미지정이면 무작위로 만든다


# ---------- 답변 반영 ----------
@pytest.mark.parametrize("seed", [1, 2, 3, 99])
def test_resolve_maps_displayed_position_back_to_the_chosen_option(seed):
    q = question()
    v = user_view(q, seed=seed)
    for shown in v["options"][:-1]:
        r = resolve(q, shown["id"], seed)
        chosen = next(o for o in q.options if o.label == shown["label"])
        assert r.scene_a_low == r.scene_a_high == chosen.scene_a_fault and r.chart_no == chosen.chart_no
        assert shown["label"] in r.statement and "사실로 확인된 것이 아닙니다" in r.statement
        assert r.evidence == q.evidence


def test_unknown_answer_returns_range_over_all_options():
    q = question()
    for ans in (UNKNOWN_ID, None):
        r = resolve(q, ans, seed=5)
        assert (r.scene_a_low, r.scene_a_high, r.chart_no) == (0, 50, None)
        assert "확실하지 않다고" in r.statement


def test_resolve_rejects_out_of_range_choice():
    with pytest.raises(ValueError):
        resolve(question(), 9, seed=1)


# ---------- 질문 생성 (모델 응답은 가짜) ----------
class PlanLLM(LLMClient):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    @property
    def mock_mode(self):
        return False

    async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        return model_cls.model_validate(self.plan)


def plan_for(q):
    return {"questions": [] if q is None else [q.model_dump()]}


def review(chart="차2-6", fault=10, conf="high"):
    return ReviewScore(reviewer_name="r", perspective="기준표 적용", fault_ratio_a=fault, fault_ratio_b=100 - fault, rationale="r",
                       chart_no=chart, chart_a_is="B", scene_a_fault=fault, confidence=conf)


def ctx(reviews=None):
    # 기본: 위원들이 서로 다른 도표를 골라 질문 생성이 열리는 상황 (실제 D09)
    reviews = reviews or [review("차2-6"), review("차2-2", 0), review("차2-6")]
    return {"accident_input": "x", "reviews": reviews, "cited_charts": "■ 차2-2 ■ 차2-6"}


def run_plan(q, reviews=None):
    return asyncio.run(plan_clarification(PlanLLM(plan_for(q)), ctx(reviews)))


def test_question_is_kept_when_outcomes_differ_a_lot():
    questions, note, warnings = run_plan(question())            # 0 / 10 / 50 → 차이 50
    assert len(questions) == 1 and note == "" and warnings == []


def test_small_difference_is_a_note_not_a_question():
    q = question([opt("가", "차2-2", 0), opt("나", "차2-6", 10)])    # 실제 D09: 차이 10
    questions, note, warnings = run_plan(q)
    assert questions == [] and "차이는 작습니다" in note and "차2-2 적용 시 A 0" in note and "차2-6 적용 시 A 10" in note
    assert CLARIFY_MIN_DIFF == 10


def test_no_question_needed():
    assert run_plan(None) == ([], "", [])


def test_options_that_contradict_chart_data_are_dropped_with_warning():
    bad = question([opt("가", "차2-2", 50), opt("나", "차2-6", 10)])   # 차2-2 는 기본 0/100 — 어느 역할로 봐도 50 은 40%p 넘게 어긋남
    questions, note, warnings = run_plan(bad)
    assert questions == [] and warnings and "질문을 생략" in warnings[0]
    nonexistent = question([opt("가", "차99-9", 0), opt("나", "차2-6", 60)])
    assert run_plan(nonexistent)[0] == [] and "존재하지 않는" in run_plan(nonexistent)[2][0]


def test_duplicate_options_are_dropped():
    dup = question([opt("같은 보기", "차2-2", 0), opt("같은 보기", "차2-6", 10)])
    questions, _, warnings = run_plan(dup)
    assert questions == [] and "중복" in warnings[0]


def test_prompt_forbids_leaking_outcomes():
    from app.agents.orchestrator import CLARIFY_PROMPT

    for must in ("법적 결론", "과실비율·도표번호", "정황의 A(나의 차량)", "evidence", "[인용 도표 상세] 밖"):
        assert must in CLARIFY_PROMPT


# ---------- 파이프라인 ----------
def test_pipeline_mock_mode_makes_no_question_and_planner_failure_keeps_result(monkeypatch):
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.clarifications == [] and r.needs_clarification is False and r.clarification_note == ""

    import app.core.pipeline as pl

    async def boom(llm, context):
        raise RuntimeError("fail")

    monkeypatch.setattr(pl, "plan_clarification", boom)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.fault_ratio_a == 50 and any("질문 단계를 건너뛰었습니다" in w for w in r.warnings)


def test_pipeline_report_carries_planned_question(monkeypatch):
    import app.core.pipeline as pl

    async def fake_plan(llm, context):
        return [question()], "", []

    monkeypatch.setattr(pl, "plan_clarification", fake_plan)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.needs_clarification is True and len(r.clarifications) == 1
    from app.evaluation.scenarios import format_clarification

    assert "차2-6" in format_clarification(r) and "A 10" in format_clarification(r)


# ---------- 질문 생성 조건: 명백한 사례에서는 모델을 부르지 않는다 (D01 오탐 방지) ----------
class CountingLLM(PlanLLM):
    calls = 0

    async def complete_json(self, *a, **k):
        CountingLLM.calls += 1
        return await super().complete_json(*a, **k)


def test_no_planner_call_when_reviewers_agree_with_high_confidence():
    CountingLLM.calls = 0
    unanimous = [review("차43-2", 20), review("차43-2", 20), review("차43-2", 20)]
    out = asyncio.run(plan_clarification(CountingLLM(plan_for(question())), ctx(unanimous)))
    assert out == ([], "", []) and CountingLLM.calls == 0         # 질문 계획이 있어도 호출 자체를 건너뛴다


@pytest.mark.parametrize("reviews,expected", [
    ([review("차2-6"), review("차2-2"), review("차2-6")], "서로 다른 도표"),
    ([review("차2-6"), review("차2-6"), review("차2-6", conf="medium")], "high 가 아닌"),
    ([review("차2-6", 10), review("차2-6", 30), review("차2-6", 10)], "과실비율 차이"),
])
def test_ambiguity_signals(reviews, expected):
    from app.agents.orchestrator import ambiguity_signals

    assert any(expected in x for x in ambiguity_signals(reviews))


def test_reviewer_warning_is_an_ambiguity_signal():
    from app.agents.orchestrator import ambiguity_signals

    w = review("차2-6")
    w.warnings = ["평가 의심"]
    assert any("경고" in x for x in ambiguity_signals([w, review("차2-6"), review("차2-6")]))
    assert ambiguity_signals([review("차2-6")] * 3) == []


# ---------- 보기 검증은 도표 역할 표시 없이 값만 본다 ----------
def test_any_role_check_accepts_either_orientation():
    from app.core.verdict_check import check_scene_vs_chart_any_role

    assert check_scene_vs_chart_any_role("차2-2", 0) == [] and check_scene_vs_chart_any_role("차2-2", 100) == []   # 내가 도표 A / B
    assert check_scene_vs_chart_any_role("차2-6", 10) == [] and check_scene_vs_chart_any_role("차2-6", 90) == []
    assert check_scene_vs_chart_any_role("차2-2", 50)                                                              # 어느 쪽도 아님
    assert "존재하지 않는" in check_scene_vs_chart_any_role("차99-9", 0)[0]
