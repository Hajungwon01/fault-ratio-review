"""첫 실측(D03)에서 드러난 오류를 고정하는 테스트.

위원 3명이 근거 문장은 "정차한 A차량은 과실이 없다"인데 숫자는 A 100(도표 A 의 값)으로 써서, 3명이 일치해 편차 0·플래그 꺼짐으로
전원 일치 오답이 감지되지 않았다. 대응책: ① 숫자를 근거 뒤(맨 끝)에 생성 ② 도표 A→정황 A 변환을 코드가 수행
③ 모델 답을 도표 기본 비율과 대조해 이상 값은 경고·정밀 심사.
"""
import asyncio
import json

import pytest

from app.agents.reviewer_scene import ReviewerSceneAgent
from app.agents.reviewer_standard import ReviewerStandardAgent
from app.core.llm_client import LLMClient, strict_json_schema
from app.core.pipeline import FaultAssessmentPipeline
from app.core.verdict_check import MAX_DEVIATION_FROM_BASE, check_verdict
from app.models.schemas import AccidentInput, ReviewerVerdict


def verdict(**kw):
    """scene_a_fault 를 따로 안 주면 도표 기준 답과 일치하는 값으로 채운다 (일치하는 정상 답)."""
    base = dict(checklist=[], rationale="근거", chart_no="차41-1", chart_a_is="B", chart_a_fault=100, confidence="high", cited_rules=[])
    base.update(kw)
    if "scene_a_fault" not in base:
        base["scene_a_fault"] = 100 - base["chart_a_fault"] if base["chart_a_is"] == "B" else base["chart_a_fault"]
    return ReviewerVerdict(**base)


# ---------- 필드 순서: 근거가 숫자보다 먼저 ----------
def test_numbers_come_after_reasoning_in_generation_order():
    order = list(ReviewerVerdict.model_json_schema()["properties"])
    assert order.index("checklist") < order.index("rationale") < order.index("scene_a_fault")
    assert order.index("scene_a_fault") < order.index("chart_a_is") < order.index("chart_a_fault")
    # OpenAI strict 스키마 변환 후에도 순서가 유지된다
    assert list(strict_json_schema(ReviewerVerdict)["properties"]) == order


# ---------- 도표 기준 답의 환산값 (교차 검증용) ----------
@pytest.mark.parametrize("is_,fault,expected", [
    ("A", 100, 100),   # 사용자가 도표의 A(뒤차) → 그대로
    ("B", 100, 0),     # 사용자가 도표의 B(앞차) → 뒤집힘  (D03 의 정답)
    ("B", 30, 70),
    ("A", 30, 30),
    ("none", 40, 40),  # 도표를 안 쓴 경우는 정황 A 값 그대로
])
def test_converted_fault_a(is_, fault, expected):
    v = verdict(chart_a_is=is_, chart_a_fault=fault, chart_no="" if is_ == "none" else "차41-1")
    assert v.converted_fault_a == expected and v.fault_ratio_a == v.scene_a_fault == expected


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


def test_d03_reviewer_front_car_gets_a_zero():
    # 정차 중이던 나(정황 A)는 0, 도표 A(뒤차)는 상대 차량이라 도표 기준으로는 100 — 두 답이 일치한다
    s = review_with({"checklist": [], "rationale": "정차 중인 나는 과실 없음", "scene_a_fault": 0, "chart_no": "차41-1",
                     "chart_a_is": "B", "chart_a_fault": 100, "confidence": "high", "cited_rules": ["차41-1"]})
    assert (s.fault_ratio_a, s.fault_ratio_b) == (0, 100)
    assert s.chart_no == "차41-1" and s.chart_a_is == "B" and s.chart_a_fault == 100
    assert s.confidence == "high" and s.warnings == []


# ---------- 도표 기본 비율과 대조 ----------
def test_check_passes_when_close_to_base_and_with_modifiers():
    assert check_verdict(verdict()) == []                                   # 차41-1 기본 100
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", chart_a_fault=20)) == []   # 기본 30, 가감 -10
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", chart_a_fault=30 + MAX_DEVIATION_FROM_BASE)) == []


def test_check_flags_value_far_from_base():
    # 도표 A 기본 100 인 차41-1 인데 도표 A 과실을 0 으로 답함 → 값/역할 오류 의심
    p = check_verdict(verdict(chart_a_fault=0))
    assert len(p) == 1 and "100" in p[0] and "100%p 차이" in p[0]
    assert check_verdict(verdict(chart_no="차43-2", chart_a_is="A", chart_a_fault=30 + MAX_DEVIATION_FROM_BASE + 1))


def test_check_uses_nearest_variant_for_multi_variant_chart():
    # 차12-1: (가)40 (나)30 (다)70 — 70 은 (다)와 일치하므로 정상
    assert check_verdict(verdict(chart_no="차12-1", chart_a_is="A", chart_a_fault=70)) == []


def test_check_flags_inconsistent_fields_and_unknown_chart():
    assert check_verdict(verdict(chart_no="", chart_a_is="A"))
    assert check_verdict(verdict(chart_no="차41-1", chart_a_is="none"))
    assert any("존재하지 않는" in x for x in check_verdict(verdict(chart_no="차99-9")))
    assert check_verdict(verdict(chart_no="", chart_a_is="none", chart_a_fault=55)) == []   # 도표 안 쓴 정상 경우


def test_suspicious_review_gets_low_confidence_and_warning():
    s = review_with({"checklist": [], "rationale": "r", "scene_a_fault": 0, "chart_no": "차41-1", "chart_a_is": "A", "chart_a_fault": 0,
                     "confidence": "high", "cited_rules": []})
    assert s.confidence == "low" and s.warnings and "차41-1" in s.warnings[0]


# ---------- 파이프라인: 전원 일치 오답도 정밀 심사로 ----------
def test_pipeline_flags_review_even_when_all_reviewers_agree(monkeypatch):
    bad = {"checklist": [], "rationale": "r", "scene_a_fault": 0, "chart_no": "차41-1", "chart_a_is": "A", "chart_a_fault": 0,
           "confidence": "high", "cited_rules": []}

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
    ([80, 80, 20], 80),    # 4.1 D08: 한 명이 이중 변환으로 틀려도 평균(60)에 끌려가지 않는다
    ([0, 100, 100], 100),  # 4.1 D10
    ([20, 20, 20], 20),
    ([30, 70], 50),        # 위원이 2명이면 평균과 같다
])
def test_final_ratio_is_median_of_reviewers(ratios, expected):
    r = report_for(ratios)
    assert r.fault_ratio_a == expected and r.fault_ratio_b == 100 - expected


def test_outlier_still_triggers_precise_review_flag():
    assert report_for([80, 80, 20]).needs_precise_review is True   # 중앙값으로 가려도 불일치는 알린다


# ---------- 평가 결과에 위원 원답(도표·역할·도표 A 과실) 저장 ----------
def test_scenario_result_stores_reviewer_raw_answers():
    from app.evaluation.scenarios import raw_answer, save_results, load_results

    v = review_with({"checklist": [], "rationale": "r", "scene_a_fault": 0, "chart_no": "차41-1", "chart_a_is": "B", "chart_a_fault": 100,
                     "confidence": "high", "cited_rules": []})
    assert raw_answer(v) == "차41-1 | 도표 A=정황 B | 도표 A 과실 100 | 정황 A 과실 0"
    none = review_with({"checklist": [], "rationale": "r", "scene_a_fault": 30, "chart_no": "", "chart_a_is": "none", "chart_a_fault": 30,
                        "confidence": "high", "cited_rules": []})
    assert raw_answer(none) == "- | 도표 A=도표 미사용 | 도표 A 과실 30 | 정황 A 과실 30"


# ---------- 정황 기준 값과 도표 기준 답의 교차 검증 (실측 6건의 이중 변환 오류) ----------
def test_d10_double_conversion_is_detected_and_scene_value_is_used():
    # 실제 4.1 D10: 역할(도표 A=정황 B)은 맞게 답하고, 정황 A(추돌차) 값 100 을 도표 A 칸에도 적어 어긋남.
    # 근거 문장은 "정차 0, 추돌 100" 이라 정황 값 100 이 맞다.
    s = review_with({"checklist": [], "rationale": "정차 0, 추돌(정황 A) 100", "scene_a_fault": 100, "chart_no": "차42-2",
                     "chart_a_is": "B", "chart_a_fault": 100, "confidence": "high", "cited_rules": ["차42-2"]})
    assert s.fault_ratio_a == 100 and s.fault_ratio_b == 0          # 최종값은 정황 기준 값
    assert s.confidence == "low" and any("서로 다름" in w for w in s.warnings)
    assert (s.scene_a_fault, s.chart_a_fault) == (100, 100)


def test_consistent_answers_have_no_mismatch_warning():
    assert not [p for p in check_verdict(verdict()) if "서로 다름" in p]
    assert not [p for p in check_verdict(verdict(chart_no="", chart_a_is="none", chart_a_fault=55)) if "서로 다름" in p]
    bad = verdict(chart_a_is="A", chart_a_fault=100, scene_a_fault=0)    # A 인데 값이 반대
    assert any("서로 다름" in p for p in check_verdict(bad))


def test_pipeline_flags_mismatched_reviewers_even_when_numbers_agree(monkeypatch):
    mism = {"checklist": [], "rationale": "r", "scene_a_fault": 100, "chart_no": "차42-2", "chart_a_is": "B",
            "chart_a_fault": 100, "confidence": "high", "cited_rules": []}

    async def fake_complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        return model_cls.model_validate(mism)

    monkeypatch.setattr(LLMClient, "complete_json", fake_complete_json)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description="정차 중인 차량을 제가 뒤에서 추돌했습니다 충분히 길게")))
    assert r.fault_ratio_a == 100 and r.spread == 0 and r.needs_precise_review is True
