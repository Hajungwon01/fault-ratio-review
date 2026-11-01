"""질문 단계 — 사용자에게는 결과를 숨기고(user_view), 답변은 진술로 반영하며(resolve), 질문은 결과가 크게 갈릴 때만 만든다."""
import asyncio
import json

import pytest

from app.agents.orchestrator import CLARIFY_MIN_DIFF, plan_clarification
from app.core.clarify import OTHER_ID, OTHER_LABEL, UNKNOWN_ID, UNKNOWN_LABEL, resolve, user_view
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
    assert v["options"][-2] == {"id": UNKNOWN_ID, "label": UNKNOWN_LABEL}
    assert v["options"][-1] == {"id": OTHER_ID, "label": OTHER_LABEL}          # 시스템이 붙이는 두 선택지는 항상 맨 끝
    assert [o["id"] for o in v["options"][:-2]] == [0, 1, 2]      # id 는 화면 위치
    orders = {tuple(o["label"] for o in user_view(q, seed=s)["options"][:-2]) for s in range(30)}
    assert len(orders) > 1                                         # seed 에 따라 순서가 바뀐다
    assert isinstance(user_view(q)["seed"], int)                   # seed 미지정이면 무작위로 만든다


# ---------- 답변 반영 ----------
@pytest.mark.parametrize("seed", [1, 2, 3, 99])
def test_resolve_maps_displayed_position_back_to_the_chosen_option(seed):
    q = question()
    v = user_view(q, seed=seed)
    for shown in v["options"][:-2]:
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
    def __init__(self, plan, conflicting=(), check_error=False):
        super().__init__()
        self.plan = plan
        self.conflicting = list(conflicting)  # 보기 점검 호출이 "정황과 모순"이라고 답할 보기 번호
        self.check_error = check_error

    @property
    def mock_mode(self):
        return False

    async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        if model_cls.__name__ == "OptionCheck":
            if self.check_error:
                raise RuntimeError("boom")
            return model_cls.model_validate({"conflicting": self.conflicting, "reason": "정황과 다른 동작"})
        return model_cls.model_validate(self.plan)


def plan_for(q):
    return {"questions": [] if q is None else [q.model_dump()]}


def review(chart="차2-6", fault=10, conf="high"):
    return ReviewScore(reviewer_name="r", perspective="기준표 적용", fault_ratio_a=fault, fault_ratio_b=100 - fault, rationale="r",
                       chart_no=chart, chart_a_is="B", scene_a_fault=fault, confidence=conf)


def ctx(reviews=None, cited=("차2-2", "차2-5", "차2-6")):
    # 기본: 위원들이 서로 다른 도표를 골라 질문 생성이 열리는 상황 (실제 D09)
    reviews = reviews or [review("차2-6"), review("차2-2", 0), review("차2-6")]
    return {"accident_input": "x", "reviews": reviews, "cited_charts": "■ 차2-2 ■ 차2-6", "cited_chart_numbers": list(cited)}


def run_plan(q, reviews=None, cited=("차2-2", "차2-5", "차2-6"), candidates=()):
    context = ctx(reviews, cited)
    context["candidate_chart_numbers"] = list(candidates)
    return asyncio.run(plan_clarification(PlanLLM(plan_for(q)), context))


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
    cited = ("차2-2", "차2-6", "차99-9")                # 인용 목록에 있어도 데이터에 없는 도표는 검증이 막는다
    assert run_plan(nonexistent, cited=cited)[0] == [] and "존재하지 않는" in run_plan(nonexistent, cited=cited)[2][0]


def test_duplicate_options_are_dropped():
    dup = question([opt("같은 보기", "차2-2", 0), opt("같은 보기", "차2-6", 10)])
    questions, _, warnings = run_plan(dup)
    assert questions == [] and "중복" in warnings[0]


def test_prompt_forbids_leaking_outcomes():
    from app.agents.orchestrator import CLARIFY_PROMPT

    for must in ("법적 결론", "과실비율·도표번호", "정황의 A(나의 차량)", "evidence", "[보기에 쓸 수 있는 도표번호] 밖"):
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


# ---------- 보기의 도표는 A/B 에이전트·위원이 인용한 것만 ----------
def test_option_with_uncited_chart_is_removed_and_may_drop_the_question():
    # 실제 D09: 정황과 맞지 않는 차2-1 이 보기로 끼어들었던 사례
    q = question([opt("가", "차2-2", 0), opt("나", "차2-6", 60), opt("다", "차2-1", 0)])
    questions, note, warnings = run_plan(q, cited=("차2-2", "차2-6"))
    assert len(questions) == 1 and [o.chart_no for o in questions[0].options] == ["차2-2", "차2-6"]
    assert warnings and "차2-1" in warnings[0] and "인용되지 않은" in warnings[0]

    only_one_left = question([opt("가", "차2-2", 0), opt("나", "차2-1", 60)])
    questions, note, warnings = run_plan(only_one_left, cited=("차2-2",))
    assert questions == [] and note == "" and "인용되지 않은" in warnings[0]


# ---------- 위원이 가정한 사실이 있으면 질문 생성을 연다 (D15: 전원 high 확신으로 조용히 가정) ----------
def test_assumed_facts_open_the_gate_even_when_everything_agrees():
    from app.agents.orchestrator import ambiguity_signals

    agree = [review("차6-1", 80) for _ in range(3)]
    assert ambiguity_signals(agree) == []
    agree[0].assumed_facts = ["상대 방향 신호가 녹색이었다"]
    assert any("가정" in x for x in ambiguity_signals(agree))


def test_assumed_facts_reach_the_planner_prompt_and_reviewer_rules():
    from app.agents.orchestrator import _format_review_summary
    from app.agents.reviewer_standard import ReviewerStandardAgent
    from app.agents.reviewer_scene import ReviewerSceneAgent
    from app.agents.reviewer_precedent import ReviewerPrecedentAgent

    r = review("차6-1", 80)
    r.assumed_facts = ["상대 방향 신호가 녹색이었다"]
    assert "상대 방향 신호가 녹색이었다" in _format_review_summary([r])
    for cls in (ReviewerStandardAgent, ReviewerSceneAgent, ReviewerPrecedentAgent):
        assert "assumed_facts" in cls.OUTPUT_RULES and "가정을 숨기지 마세요" in cls.OUTPUT_RULES


def test_reviewer_score_carries_assumed_facts():
    import asyncio as _a
    from app.agents.reviewer_standard import ReviewerStandardAgent
    from tests.test_role_conversion import FakeJsonLLM

    payload = {"checklist": [], "rationale": "r", "assumed_facts": [" 상대 신호는 녹색 ", ""], "scene_a_fault": 80,
               "chart_no": "차6-1", "chart_a_is": "A", "confidence": "high", "cited_rules": []}
    ctx_ = {"accident_input": "x", "rounds": [], "fact_check": None, "cited_charts": "■ 차6-1"}
    s = _a.run(ReviewerStandardAgent(FakeJsonLLM(payload)).run(ctx_))
    assert s.assumed_facts == ["상대 신호는 녹색"]               # 공백/빈 항목은 정리한다


# ---------- 사용자에게 보이는 문장에 내부 용어가 새면 질문을 버린다 ----------
@pytest.mark.parametrize("label", [
    "나(정황 A)의 진행로는 적색 신호였다",       # 실제 D14 보기에 섞였던 내부 용어
    "차2-2 가 적용되는 신호였다",
    "과실이 큰 쪽의 신호였다",
    "녹색 신호였고 비율은 10% 였다",
])
def test_internal_terms_in_option_label_drop_the_question(label):
    q = question([opt(label, "차2-2", 0), opt("정지선 앞 점멸 신호였다", "차2-6", 90)])
    questions, note, warnings = run_plan(q)
    assert questions == [] and "내부 용어" in warnings[0]


def test_internal_terms_in_question_text_drop_the_question():
    q = question()
    q = q.model_copy(update={"question": "도표 차2-2 를 적용하려는데 상대 신호는?"})
    assert "내부 용어" in run_plan(q)[2][0]
    ok = question().model_copy(update={"question": "충돌 직전 내 방향과 상대 방향의 신호는 각각 무엇이었나요?"})
    assert run_plan(ok)[0]                                          # 일반 문장은 통과


def test_prompt_requires_complete_options_and_plain_words():
    from app.agents.orchestrator import CLARIFY_PROMPT

    assert "빠짐없이" in CLARIFY_PROMPT and "유리한 경우와 불리한 경우" in CLARIFY_PROMPT
    assert "내부 용어" in CLARIFY_PROMPT and "가정한 사실" in CLARIFY_PROMPT


# ---------- 보기에 쓸 수 있는 도표 = 인용된 도표 + 검색 후보 + 위원이 고른 도표 ----------
def test_search_candidates_are_allowed_even_if_nobody_cited_them():
    # 실제 D14: 에이전트들이 엉뚱한 도표를 인용해도, 검색 후보에 있는 맞는 도표(차1-1, 차1-4)는 보기로 쓸 수 있어야 한다
    q = question([opt("내 방향 신호가 녹색이었다", "차1-1", 0), opt("양쪽 모두 적색이었다", "차1-4", 50),
                  opt("내 방향 신호가 적색이었다", "차1-1", 100)])
    questions, note, warnings = run_plan(q, cited=("차3-1",), candidates=("차1-1", "차1-4"))
    assert len(questions) == 1 and len(questions[0].options) == 3 and warnings == []

    # 후보에도 인용에도 없는 도표는 여전히 제외된다
    questions, note, warnings = run_plan(q, cited=("차3-1",), candidates=("차1-1",))
    assert [o.chart_no for o in questions[0].options] == ["차1-1", "차1-1"] and "차1-4" in warnings[0]


def test_planner_prompt_lists_allowed_numbers_and_candidate_details():
    seen = {}

    class Spy(PlanLLM):
        async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
            seen["prompt"] = user_prompt
            return model_cls.model_validate({"questions": []} if model_cls.__name__ == "ClarificationPlan" else {})

    c = ctx()
    c["candidate_chart_numbers"] = ["차1-1", "차1-4"]
    c["reference_charts"] = "■ 차1-1 후보 상세"
    asyncio.run(plan_clarification(Spy({}), c))
    p = seen["prompt"]
    assert "[검색 후보 도표 상세]" in p and "차1-1 후보 상세" in p
    allowed_line = p.split("[보기에 쓸 수 있는 도표번호]")[1]
    assert "차1-1" in allowed_line and "차1-4" in allowed_line and "차2-6" in allowed_line


def test_pipeline_passes_candidates_to_planner(monkeypatch):
    import app.core.pipeline as pl

    got = {}

    async def spy(llm, context):
        got.update(context)
        return [], "", []

    monkeypatch.setattr(pl, "plan_clarification", spy)
    asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert got["candidate_chart_numbers"] and all(n.startswith("차") for n in got["candidate_chart_numbers"])
    assert "reference_charts" in got and "cited_chart_numbers" in got


# ---------- 진단 기록(clarification_trace): 질문이 왜 나왔는지/안 나왔는지 ----------
def trace_for(q, reviews=None, cited=("차2-2", "차2-5", "차2-6")):
    context = ctx(reviews, cited)
    context["candidate_chart_numbers"] = []
    context["trace"] = []
    asyncio.run(plan_clarification(PlanLLM(plan_for(q)), context))
    return context["trace"]


def test_trace_records_gate_closed_and_open():
    agree = [review("차43-2", 20), review("차43-2", 20), review("차43-2", 20)]
    assert trace_for(question(), agree) == ["게이트 닫힘: 모호함 신호 없음"]
    t = trace_for(question())
    assert t[0].startswith("게이트 열림: ") and "서로 다른 도표" in t[0] and t[-1] == "질문 채택: 보기별 결과 차이 50%p"


@pytest.mark.parametrize("q,expected", [
    (None, "질문 계획: 질문 불필요로 판단"),
    (question([opt("가", "차2-2", 0), opt("나", "차2-6", 10)]), "참고 문구로 대체: 보기별 결과 차이 10%p"),
    (question([opt("같은 보기", "차2-2", 0), opt("같은 보기", "차2-6", 10)]), "폐기: 보기 중복"),
    (question([opt("가", "차2-2", 50), opt("나", "차2-6", 10)]), "폐기: 보기 값이 도표 데이터와 불일치"),
    (question([opt("과실 큰 쪽", "차2-2", 0), opt("나", "차2-6", 90)]), "폐기: 질문·보기에 내부 용어/수치 누출"),
])
def test_trace_records_why_no_question(q, expected):
    assert expected in trace_for(q)


def test_pipeline_report_and_scenario_result_carry_the_trace_and_summary_check():
    from app.evaluation.scenarios import check_user_summary, evaluate_report, Scenario

    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert r.clarification_trace.startswith("게이트 ")              # mock 위원은 확신 medium 이라 게이트가 열리고 mock 계획은 질문 없음
    sc = Scenario.model_validate({"id": "T", "description": DESC, "expected_chart": ["차2-2"]})
    res = evaluate_report(sc, r)
    assert res.clarification_trace == r.clarification_trace and res.basis == r.basis
    assert res.summary_problems == ["설명 없음"] == check_user_summary(r)        # mock 은 사용자용 설명이 비어 있다
    assert set(res.reviewer_assumed) == {v.perspective for v in r.reviews}


# ---------- 질문 계획: 영향이 가장 큰 사실을 고르게 하고, 참고 문구는 같은 도표를 묶는다 ----------
def test_planner_prompt_asks_for_the_highest_impact_fact():
    from app.agents.orchestrator import CLARIFY_PROMPT

    for must in ("가장 크게 바뀌는 사실 하나", "사소한 경우 차이", "신호·표지의 유무와 신호 상태"):
        assert must in CLARIFY_PROMPT


def test_note_groups_options_of_the_same_chart():
    # 실측 D14: 같은 도표의 두 경우(A 40 / A 50)가 "차12-1 적용 시 A 40, 차12-1 적용 시 A 50" 으로 중복 표시됐다
    q = question([opt("가", "차2-6", 10), opt("나", "차2-6", 20)])
    questions, note, _ = run_plan(q)
    assert questions == [] and "차2-6 적용 시 A 10~20" in note and note.count("차2-6") == 1
    assert "경우에 따라 비율이 조금 달라질 수 있으나" in note and "적용 도표가 달라질 수 있으나" not in note


# ---------- '해당 없음': 보기가 모든 경우를 덮지 못할 때 모름의 낙관적 범위를 보여 주지 않는다 (실측 D14) ----------
def test_other_answer_does_not_narrow_the_range():
    q = question()                                          # 보기 값 0 / 10 / 50 — 가장 불리한 경우가 빠진 질문
    other = resolve(q, OTHER_ID, seed=3)
    assert (other.scene_a_low, other.scene_a_high, other.chart_no, other.kind) == (0, 100, None, "other")
    assert "해당하는 것이 없다고" in other.statement and other.evidence == q.evidence
    unknown = resolve(q, UNKNOWN_ID, seed=3)
    assert (unknown.scene_a_low, unknown.scene_a_high, unknown.kind) == (0, 50, "unknown")       # 모름은 보기 안의 범위만
    assert "보기에 있던 경우의 범위" in unknown.statement


# ---------- 보기 점검: 정황과 모순되는 보기 제외 (실측 D14) ----------
def run_plan_with(q, **kw):
    return asyncio.run(plan_clarification(PlanLLM(plan_for(q), **kw), ctx()))


def test_conflicting_option_is_dropped_but_question_kept():
    out, note, warnings = run_plan_with(question(), conflicting=[0])      # 보기 3개 중 첫째 제외 → 둘째·셋째(10, 50)가 남아 질문 유지
    assert len(out) == 1 and len(out[0].options) == 2
    assert "녹색 직진 신호 또는 적색 신호였다" not in [o.label for o in out[0].options]


def test_question_discarded_when_too_few_options_remain():
    out, note, warnings = run_plan_with(question(), conflicting=[0, 1])   # 3개 중 2개 제외 → 1개 남음
    assert out == [] and any("정황과 맞지 않아" in w for w in warnings)


def test_question_discarded_when_check_call_fails():
    out, note, warnings = run_plan_with(question(), check_error=True)      # 품질 확인 없는 질문은 내지 않는다
    assert out == [] and any("정황과 맞지 않아" in w for w in warnings)


def test_out_of_range_conflict_numbers_are_ignored():
    out, note, warnings = run_plan_with(question(), conflicting=[7, -1])
    assert len(out) == 1 and len(out[0].options) == len(question().options)


def test_check_prompt_contains_scene_and_numbered_options():
    seen = {}

    class Spy(PlanLLM):
        async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
            if model_cls.__name__ == "OptionCheck":
                seen["p"] = user_prompt
            return await super().complete_json(system_prompt, user_prompt, model_cls, mock=mock)

    c = ctx()
    c["accident_input"] = "내가 직진하다 충돌"
    asyncio.run(plan_clarification(Spy(plan_for(question())), c))
    assert "내가 직진하다 충돌" in seen["p"] and "0. " in seen["p"] and "1. " in seen["p"]
