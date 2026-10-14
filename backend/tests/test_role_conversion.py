"""실제 실측에서 드러난 위원 숫자 오류를 고정하는 테스트.

- D03: 위원 3명이 근거 문장은 "정차한 A차량은 과실이 없다"인데 숫자는 도표 A 의 값(100)으로 써서 전원 일치 오답이 감지되지 않았다.
- D02·D06·D08·D10: 역할(도표 A=정황 B)은 맞게 답하고도 "도표 A 입장의 과실" 칸에 정황 값을 적어 코드 환산이 한 번 더 뒤집혔다.
  → 모델은 정황 A 기준 과실(scene_a_fault)을 직접 답하고, 코드는 도표 기본 비율(chart_a_is 로 정황 기준으로 바꾼 값)과 대조만 한다.
"""
import asyncio

import pytest

from app.agents.reviewer_standard import ReviewerStandardAgent
from app.core.llm_client import LLMClient, strict_json_schema
from app.core.pipeline import FaultAssessmentPipeline
from app.core.verdict_check import MAX_DEVIATION_FROM_BASE, check_verdict
from app.models.schemas import AccidentInput, ReviewerVerdict


def verdict(**kw):
    """기본값: 차41-1 (도표 A=뒤차 추돌 100 / 도표 B=앞차 0), 나(정황 A)는 앞차 = 도표 B → 정황 A 과실 0."""
    base = dict(checklist=[], rationale="근거", scene_a_fault=0, chart_no="차41-1", chart_a_is="B", confidence="high", cited_rules=[])
    return ReviewerVerdict(**{**base, **kw})


# ---------- 필드 순서: 근거가 숫자보다 먼저 ----------
def test_numbers_come_after_reasoning_in_generation_order():
    order = list(ReviewerVerdict.model_json_schema()["properties"])
    assert order.index("checklist") < order.index("rationale") < order.index("scene_a_fault")
    assert order.index("scene_a_fault") < order.index("chart_no") < order.index("chart_a_is")
    assert "chart_a_fault" not in order            # 도표 기준 과실 칸은 받지 않는다 (정황 값을 잘못 적는 오류의 원인)
    # OpenAI strict 스키마 변환 후에도 순서가 유지된다
    assert list(strict_json_schema(ReviewerVerdict)["properties"]) == order


def test_final_ratio_is_scene_a_fault_as_answered():
    assert verdict(scene_a_fault=0).fault_ratio_a == 0
    assert verdict(scene_a_fault=80, chart_a_is="A").fault_ratio_a == 80
    assert verdict(scene_a_fault=30, chart_no="", chart_a_is="none").fault_ratio_a == 30


class FakeJsonLLM(LLMClient):
    """complete_json 만 가로채 모델이 낸 JSON 을 그대로 돌려주는 가짜 (실제 호출 없음)."""

    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self._mock = False

    @property
    def mock_mode(self):
        return False

    async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        return model_cls.model_validate(self.payload)


def review_with(payload, cls=ReviewerStandardAgent):
    ctx = {"accident_input": "x", "rounds": [], "fact_check": None, "cited_charts": "■ 차41-1"}
    return asyncio.run(cls(FakeJsonLLM(payload)).run(ctx))


def payload(**kw):
    base = {"checklist": [], "rationale": "r", "scene_a_fault": 0, "chart_no": "차41-1", "chart_a_is": "B",
            "confidence": "high", "cited_rules": []}
    return {**base, **kw}


def test_d03_reviewer_front_car_gets_a_zero():
    s = review_with(payload(rationale="정차 중인 나는 과실 없음", cited_rules=["차41-1"]))
    assert (s.fault_ratio_a, s.fault_ratio_b) == (0, 100)
    assert s.chart_no == "차41-1" and s.chart_a_is == "B" and s.scene_a_fault == 0
    assert s.confidence == "high" and s.warnings == []


# ---------- 도표 기본 비율과 대조 (chart_a_is 로 정황 기준으로 바꿔서) ----------
def test_check_passes_when_close_to_base_and_with_modifiers():
    assert check_verdict(verdict()) == []                                                   # 차41-1: 내가 도표 B → 기본 0
    assert check_verdict(verdict(scene_a_fault=100, chart_a_is="A")) == []                 # 내가 도표 A → 기본 100
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", scene_a_fault=20)) == []   # 도표 A 기본 30, 가감 -10
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", scene_a_fault=30 + MAX_DEVIATION_FROM_BASE)) == []
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="B", scene_a_fault=70)) == []   # 내가 도표 B → 기본 70


def test_check_flags_value_far_from_base():
    # 내가 도표 B(앞차, 기본 0)라고 답했는데 정황 A 과실을 100 으로 답함 → 값/역할 오류 의심
    p = check_verdict(verdict(scene_a_fault=100))
    assert len(p) == 1 and "100%p 차이" in p[0] and "정황 B" in p[0]
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", scene_a_fault=30 + MAX_DEVIATION_FROM_BASE + 1))


def test_check_uses_nearest_variant_for_multi_variant_chart():
    # 차12-1: 도표 A 기본 (가)40 (나)30 (다)70
    assert check_verdict(verdict(chart_no="차12-1", chart_a_is="A", scene_a_fault=70)) == []
    assert check_verdict(verdict(chart_no="차12-1", chart_a_is="B", scene_a_fault=30)) == []   # 뒤집으면 60/70/30


def test_check_flags_inconsistent_fields_and_unknown_chart():
    assert check_verdict(verdict(chart_no="", chart_a_is="A"))
    assert check_verdict(verdict(chart_no="차41-1", chart_a_is="none"))
    assert any("존재하지 않는" in x for x in check_verdict(verdict(chart_no="차99-9")))
    assert check_verdict(verdict(chart_no="", chart_a_is="none", scene_a_fault=55)) == []   # 도표 안 쓴 정상 경우


def test_suspicious_review_gets_low_confidence_and_warning():
    s = review_with(payload(chart_a_is="A", scene_a_fault=0))       # 내가 도표 A(기본 100)인데 0 이라고 답함
    assert s.confidence == "low" and s.warnings and "차41-1" in s.warnings[0]


def test_d10_inverted_value_is_detected_and_correct_value_is_not():
    # 4.1 D10: 차42-2 는 도표 A=정차 차량(0), 도표 B=추돌 차량(100). 나는 추돌한 쪽 = 도표 B 라서 정황 A 과실은 100 이다.
    wrong = review_with(payload(chart_no="차42-2", chart_a_is="B", scene_a_fault=0))
    assert wrong.confidence == "low" and wrong.warnings
    right = review_with(payload(chart_no="차42-2", chart_a_is="B", scene_a_fault=100))
    assert right.fault_ratio_a == 100 and right.warnings == [] and right.confidence == "high"


# ---------- 파이프라인: 전원 일치 오답도 정밀 심사로 ----------
def test_pipeline_flags_review_even_when_all_reviewers_agree(monkeypatch):
    bad = payload(chart_a_is="A", scene_a_fault=0)

    async def fake_complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        return model_cls.model_validate(bad)   # 위원 3명 모두 똑같이 틀린 값을 낸다

    monkeypatch.setattr(LLMClient, "complete_json", fake_complete_json)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description="신호 대기 중 정차한 제 차를 뒤차가 추돌했습니다 충분히")))
    assert len({v.fault_ratio_a for v in r.reviews}) == 1 and r.spread == 0 and r.deviation == 0  # 전원 일치
    assert r.needs_precise_review is True                                  # 그래도 도표 기본 비율과 안 맞아 플래그
    assert sum("평가 의심" in w for w in r.warnings) == 3


def test_pipeline_normal_mock_run_has_no_review_warnings():
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description="녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다")))
    assert r.warnings == [] and all(v.warnings == [] for v in r.reviews)


# ---------- 최종값은 위원 중앙값 (한 명의 이탈값에 강하다) ----------
def report_for(ratios):
    from app.agents.orchestrator import ProceedingAgent
    from app.models.schemas import ReviewScore

    reviews = [ReviewScore(reviewer_name=f"r{i}", perspective=f"p{i}", fault_ratio_a=a, fault_ratio_b=100 - a, rationale="r")
               for i, a in enumerate(ratios)]
    return asyncio.run(ProceedingAgent(LLMClient()).run({"reviews": reviews}))


@pytest.mark.parametrize("ratios,expected", [
    ([80, 80, 20], 80),    # 4.1 D08: 한 명이 틀려도 평균(60)에 끌려가지 않는다
    ([0, 100, 100], 100),  # 4.1 D10
    ([20, 20, 20], 20),
    ([30, 70], 50),        # 위원이 2명이면 평균과 같다
])
def test_final_ratio_is_median_of_reviewers(ratios, expected):
    r = report_for(ratios)
    assert r.fault_ratio_a == expected and r.fault_ratio_b == 100 - expected


def test_outlier_still_triggers_precise_review_flag():
    assert report_for([80, 80, 20]).needs_precise_review is True   # 중앙값으로 가려도 불일치는 알린다


# ---------- 평가 결과에 위원 원답(도표·역할·정황 A 과실) 저장 ----------
def test_scenario_result_stores_reviewer_raw_answers():
    from app.evaluation.scenarios import raw_answer

    assert raw_answer(review_with(payload())) == "차41-1 | 도표 A=정황 B | 정황 A 과실 0"
    none = review_with(payload(chart_no="", chart_a_is="none", scene_a_fault=30))
    assert raw_answer(none) == "- | 도표 A=도표 미사용 | 정황 A 과실 30"


# ---------- 확신 낮음(low) 위원이 여럿이면 숫자가 일치해도 정밀 심사 ----------
def report_with_confidence(levels, ratios=None):
    from app.agents.orchestrator import ProceedingAgent
    from app.models.schemas import ReviewScore

    ratios = ratios or [10] * len(levels)
    reviews = [ReviewScore(reviewer_name=f"r{i}", perspective=f"p{i}", fault_ratio_a=a, fault_ratio_b=100 - a,
                           rationale="r", confidence=c) for i, (a, c) in enumerate(zip(ratios, levels))]
    return asyncio.run(ProceedingAgent(LLMClient()).run({"reviews": reviews}))


def test_two_low_confidence_reviewers_flag_even_when_numbers_agree():
    r = report_with_confidence(["low", "low", "high"])           # 숫자는 10/10/10 으로 일치
    assert r.spread == 0 and r.needs_precise_review is True
    assert len(r.flag_reasons) == 1 and "확신 낮음" in r.flag_reasons[0] and "p0" in r.flag_reasons[0]


@pytest.mark.parametrize("levels", [["high", "high", "high"], ["low", "high", "high"], ["medium", "medium", "low"], ["medium"] * 3])
def test_fewer_than_two_low_confidence_do_not_flag(levels):
    r = report_with_confidence(levels)
    assert r.needs_precise_review is False and r.flag_reasons == []


def test_flag_reasons_list_every_cause():
    r = report_with_confidence(["low", "low", "low"], ratios=[0, 50, 80])
    assert r.needs_precise_review is True and len(r.flag_reasons) == 3     # spread, deviation, low confidence
    assert any("최대-최소" in x for x in r.flag_reasons) and any("표준편차" in x for x in r.flag_reasons)


def test_pipeline_adds_reasons_for_missing_reviewer_and_warnings(monkeypatch):
    from app.agents.reviewer_scene import ReviewerSceneAgent

    async def boom(self, context):
        raise RuntimeError("fail")

    monkeypatch.setattr(ReviewerSceneAgent, "run", boom)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description="녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다")))
    assert r.needs_precise_review is True and any("2건뿐" in x for x in r.flag_reasons)
