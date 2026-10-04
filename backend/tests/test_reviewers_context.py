import asyncio

import pytest

from app.agents.reviewer_precedent import ReviewerPrecedentAgent
from app.agents.reviewer_scene import ReviewerSceneAgent
from app.agents.reviewer_standard import ReviewerStandardAgent
from app.core.chart_lookup import format_chart, get_catalog
from app.core.llm_client import LLMClient
from app.core.pipeline import FaultAssessmentPipeline
from app.core.reviewer_context import (
    cited_chart_numbers,
    format_cited_charts,
    format_precedents,
    precedent_evidence,
)
from app.models.schemas import AccidentInput, AgentArgument

DESC = "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다"


# ---------- 택일 그룹 데이터 ----------
# 포털이 A측·B측의 '현저/중대한 과실'에 같은 그룹 번호를 함께 쓰는 도표. 포털 UI 는 한쪽을 체크하면 반대쪽이 해제된다.
# 인정기준 취지상 의도인지 포털 입력 관행인지는 확인하지 못해 포털 그대로 저장했다 (수집 보고서 한계 참고).
CROSS_SIDE_GROUP_CHARTS = {"차1-5", "차9-2", "차17-2", "차20-2"}


def test_every_modifier_has_group_and_exclusive_groups_are_single_sided_except_known():
    cross = set()
    for no, c in get_catalog().charts.items():
        for v in c["variants"]:
            groups: dict[int, set[str]] = {}
            for m in v["modifiers"]:
                assert isinstance(m["group"], int), (no, m)
                if m["group"]:
                    groups.setdefault(m["group"], set()).add(m["side"])
            if any(len(sides) > 1 for sides in groups.values()):
                cross.add(no)
    # 새로운 예외가 생기면(포털 개정 등) 이 테스트가 알려준다
    assert cross == CROSS_SIDE_GROUP_CHARTS


def test_format_chart_shows_exclusive_groups_and_formula():
    t = format_chart("차2-2", get_catalog().get("차2-2"))
    assert "[택일1] A 현저한 과실 +10" in t and "[택일1] A 중대한 과실 +20" in t
    assert "A 과실 = 기본 A + (A측 가감점 합) − (B측 가감점 합)" in t


# ---------- 인용 도표 / 판례 근거 등급 ----------
def test_cited_chart_numbers_dedup_and_drop_nonexistent():
    rounds = [
        AgentArgument(agent_name="vehicle_a_agent", round=1, content="차43-2 적용, 차99-9 도 본다", cited_rules=["차43-2", "도로교통법 제19조"]),
        AgentArgument(agent_name="vehicle_b_agent", round=1, content="차41-1 과 차43-2 비교"),
    ]
    assert cited_chart_numbers(rounds) == ["차43-2", "차41-1"]
    assert cited_chart_numbers(None) == [] and cited_chart_numbers([]) == []


def test_precedent_levels_direct_similar_none():
    assert precedent_evidence("차51-1").level == "direct"          # 자체 판례 있음
    sim = precedent_evidence("차3-2")                              # 판례 없음 → 같은 분류 도표의 사례
    assert sim.level == "similar" and sim.items and all(src != "차3-2" for src, _, _ in sim.items)
    assert precedent_evidence("차46-1").level == "none"             # 유사 도표에서도 못 찾음
    assert precedent_evidence("차99-9").level == "none"


def test_similar_precedents_are_labeled_in_prompt_text():
    t = format_precedents(["차3-2"])
    assert "유사 도표의 사례" in t and "직접 근거로 취급하지 말 것" in t and "[유사: 차3-1]" in t
    assert "근거 없음" in format_precedents(["차46-1"])
    assert "조회하지 못함" in format_precedents([])


def test_cited_charts_text_limits_count_and_notes():
    nos = ["차43-2", "차41-1", "차51-1", "차52-1", "차12-1"]
    t = format_cited_charts(nos)
    assert t.count("■ 차") == 4 and "차12-1" not in t
    assert "인용한 유효한 도표 없음" in format_cited_charts([])


# ---------- 위원 프롬프트 ----------
class Spy(LLMClient):
    def __init__(self):
        super().__init__()
        self.prompts = []

    async def complete_json(self, system_prompt, user_prompt, model_cls, *, mock, max_attempts=3):
        self.prompts.append((system_prompt, user_prompt))
        return model_cls.model_validate({**mock, "checklist": [{"item": "도표 선택", "finding": "차2-2"}]})


CTX = {"accident_input": "사고 상황 설명: x", "rounds": [], "fact_check": None,
       "cited_charts": "■ 차2-2 도표", "precedents": "■ 차2-2 근거 등급: 직접 근거"}


def test_each_reviewer_has_checklist_and_receives_cited_charts():
    for cls in (ReviewerStandardAgent, ReviewerSceneAgent, ReviewerPrecedentAgent):
        spy = Spy()
        s = asyncio.run(cls(spy).run(CTX))
        system, user = spy.prompts[0]
        assert "[체크리스트]" in system and "[출력 규칙]" in system and "checklist" in system
        assert "[인용 도표 상세]\n■ 차2-2 도표" in user
        assert s.checklist[0].item == "도표 선택"


def test_only_precedent_reviewer_gets_precedents_and_standard_gets_formula():
    spy = Spy()
    asyncio.run(ReviewerStandardAgent(spy).run(CTX))
    asyncio.run(ReviewerSceneAgent(spy).run(CTX))
    asyncio.run(ReviewerPrecedentAgent(spy).run(CTX))
    (s_std, u_std), (s_scene, u_scene), (s_prec, u_prec) = spy.prompts
    assert "[판례·조정사례]" not in u_std and "[판례·조정사례]" not in u_scene
    assert "[판례·조정사례]\n■ 차2-2 근거 등급: 직접 근거" in u_prec
    assert "(B측 가감점 합)" in s_std and "택일" in s_std
    assert "정황에 적힌 사실만" in s_scene
    assert "주어진 사례만" in s_prec and "근거 없음" in s_prec


# ---------- 파이프라인: 위원 실패 허용 ----------
def test_pipeline_continues_when_one_reviewer_fails_and_flags_precise_review(monkeypatch):
    async def boom(self, context):
        raise RuntimeError("테스트: 위원 장애")

    monkeypatch.setattr(ReviewerSceneAgent, "run", boom)
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert len(r.reviews) == 2 and r.needs_precise_review is True
    assert any("현장 정황" in w and "제외" in w for w in r.warnings)


def test_pipeline_aborts_when_too_few_reviewers(monkeypatch):
    async def boom(self, context):
        raise RuntimeError("테스트: 위원 장애")

    monkeypatch.setattr(ReviewerSceneAgent, "run", boom)
    monkeypatch.setattr(ReviewerPrecedentAgent, "run", boom)
    with pytest.raises(RuntimeError, match="유효한 검토위원 평가"):
        asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))


def test_pipeline_normal_run_has_three_reviews_and_no_forced_flag():
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description=DESC)))
    assert len(r.reviews) == 3 and r.needs_precise_review is False and r.warnings == []
