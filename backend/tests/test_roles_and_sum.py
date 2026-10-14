"""실제 API 첫 실행(2026-10-05)에서 드러난 오류를 고정하는 테스트.

- 도표 A/B 와 정황 A/B 의 역할 혼동 (도표 A 비율을 정황 A 비율처럼 씀 → A 100 오판)
- 가감요소를 한쪽에만 더해 과실비율 합이 100 이 아니게 됨 ("A 30 : B 80")
"""
import pytest

from app.agents.fact_checker import FactCheckerAgent
from app.agents.reviewer_precedent import ReviewerPrecedentAgent
from app.agents.reviewer_scene import ReviewerSceneAgent
from app.agents.reviewer_standard import ReviewerStandardAgent
from app.agents.vehicle_a import VehicleAAgent
from app.agents.vehicle_b import VehicleBAgent
from app.core.citation_check import verify_citations
from app.core.llm_client import LLMClient


def errors(text):
    return [f.message for f in verify_citations(text) if f.level == "error"]


# ---------- 합계 검증 ----------
def test_sum_not_100_is_error():
    e = errors("기본 A 30 : B 70 에 B 가산 10 을 더해 최종 과실비율은 A 30 : B 80 입니다")
    assert len(e) == 1 and "합이 100이 아님: A 30 : B 80 (합 110)" in e[0]


@pytest.mark.parametrize("text", [
    "최종: 정황 A 20 : 정황 B 80",       # 합이 100 인 최종 결론
    "A 30 : B 70",
    "A차량 30% : B차량 70%",
    "A 0 : B 100",
    "B 70 + 10% = B 80",                 # 계산 과정의 중간값은 쌍이 아니다
])
def test_valid_or_non_pair_text_not_flagged_for_sum(text):
    assert not [m for m in errors(text) if "합이 100" in m]


def test_sum_check_catches_the_exact_format_the_prompt_requires():
    # 실제 API 실행에서 에이전트 둘 다 이 형식으로 합 110 을 썼는데 정규식이 놓쳤다 (회귀 방지)
    for text in ("`최종: 정황 A 30 : 정황 B 80`", "**최종: 정황 A 30 : 정황 B 80**", "최종: 정황 A 30% : 정황 B 80%"):
        e = [m for m in errors(text) if "합이 100이 아님: A 30 : B 80" in m]
        assert len(e) == 1, text
    assert not [m for m in errors("최종: 정황 A 20 : 정황 B 80") if "합이 100" in m]
    assert not [m for m in errors("최종: 정황 A 30 : 정황 B 70") if "합이 100" in m]


def test_sum_error_reported_once_even_if_repeated():
    e = errors("A 30 : B 80 입니다. 다시 말해 A 30 : B 80 입니다.")
    assert len(e) == 1


def test_fact_checker_flags_sum_error_from_rule_check():
    import asyncio

    class Fake(LLMClient):
        async def run_agent(self, *a, **k):
            return "오류 없음"

    r = asyncio.run(FactCheckerAgent(Fake()).run({"argument_a": "최종 A 30 : B 80", "argument_b": "차43-2 A 30 : B 70"}))
    assert not r.is_valid and any(f.startswith("A측 과실비율 합이 100이 아님") for f in r.flagged_citations)


# ---------- 프롬프트에 역할 대응·계산 규칙이 들어 있는가 ----------
def test_advocate_prompts_require_role_mapping_and_sum_rule():
    for cls in (VehicleAAgent, VehicleBAgent):
        p = cls(LLMClient()).system_prompt
        assert "도표 A = 정황 A" in p and "도표 A = 정황 B" in p      # 역할 대응을 첫머리에 밝힘
        assert "정황의 A/B 기준" in p                                    # 이후 비율은 정황 기준으로 환산
        assert "합이 100" in p and "A 20 : B 80" in p                    # 계산 규칙과 예시
        assert "최종: 정황 A xx : 정황 B yy" in p
        assert "search_charts" in p and "get_chart" in p
        assert "갈아타" in p                                             # 2라운드 도표 갈아타기 금지


def test_reviewers_report_in_chart_terms_and_decide_number_last():
    for cls in (ReviewerStandardAgent, ReviewerSceneAgent, ReviewerPrecedentAgent):
        r = cls.OUTPUT_RULES
        assert "도표 A = 정황 A/B" in r                                   # 역할 대응 확인
        assert "숫자는 맨 마지막에" in r                                    # 근거 먼저, 숫자는 나중
        assert "**정황의 A(나의 차량)** 입장" in r and "scene_a_fault" in r and "**정황의 A/B** 기준" in r  # 정황 기준으로 직접 답
        assert "chart_a_fault" not in r                                    # 도표 기준 과실 칸은 없다
        assert 'chart_a_is=\"B\", scene_a_fault=0' in r                    # 역할 반전 예시(차41-1 앞차)
    assert "역할 대응" in ReviewerStandardAgent(LLMClient()).system_prompt
    assert "scene_a_fault" in ReviewerStandardAgent(LLMClient()).system_prompt


def test_fact_checker_checks_applicability_and_role_mapping():
    p = FactCheckerAgent(LLMClient()).system_prompt
    assert "(4)" in p and "(5)" in p and "대응" in p and "정황 A 기준" in p


def test_reviewers_must_quote_the_scene_for_every_modifier_they_apply():
    # 4.1 D07: 위원 3명이 정황에 없는 B측 가산(+20)을 "통상 그렇다"며 적용해 A 20 → 0 으로 내림
    for cls in (ReviewerStandardAgent, ReviewerSceneAgent, ReviewerPrecedentAgent):
        r = cls.OUTPUT_RULES
        assert "가감요소는 정황에 적힌 사실이 있을 때만" in r
        assert "정황 문장 그대로 인용" in r and "근거가 아닙니다" in r      # 추정 금지 + 인용 형식
        assert "해석이 갈리면 적용하지 말고 기본 비율을 유지" in r           # 애매하면 미적용
        assert "20%p 넘게" in r                                              # 큰 이동은 재확인
