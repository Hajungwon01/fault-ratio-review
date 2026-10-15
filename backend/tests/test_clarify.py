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
            return model_cls.model_validate({"questions": []})

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
