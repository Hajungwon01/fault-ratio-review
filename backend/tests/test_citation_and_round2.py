import asyncio

from app.agents.fact_checker import FactCheckerAgent
from app.agents.vehicle_a import VehicleAAgent
from app.core.citation_check import verify_citations
from app.models.schemas import FactCheckResult


class FakeLLM:
    mock_mode = True

    def __init__(self, reply="오류 없음"):
        self.reply = reply
        self.prompts = []

    async def run_agent(self, system_prompt, user_prompt, tools, max_iterations=8):
        self.prompts.append(user_prompt)
        return self.reply

    async def complete(self, system_prompt, user_prompt):
        return self.reply


def test_valid_citation_has_no_findings():
    assert verify_citations("차43-2 기본 과실 30:70, 도로교통법 제19조 위반") == []


def test_reversed_ratio_is_accepted():
    # A:B 를 뒤집어 70:30 으로 쓴 경우는 오탐하지 않는다
    assert verify_citations("차43-2 는 70:30") == []


def test_wrong_ratio_nonexistent_chart_and_law_warning():
    msgs = [str(f) for f in verify_citations("차43-2 는 80:20. 차99-9 적용. 도로교통법 제18조")]
    assert any("error" in m and "80:20" in m and "30:70" in m for m in msgs)
    assert any("error" in m and "차99-9" in m for m in msgs)
    assert any("warning" in m and "제18조" in m for m in msgs)


def test_multi_variant_chart_accepts_any_variant_ratio():
    # 차12-1 은 (가)40:60 (나)30:70 (다)70:30
    assert verify_citations("차12-1 (나) 30:70") == []
    assert verify_citations("차12-1 은 50:50")[0].level == "error"


def test_fact_checker_combines_rule_and_llm_findings():
    llm = FakeLLM("검토 완료\n오류: B측 가감요소 '진로변경 신호불이행'은 정황에 근거 없음")
    r = asyncio.run(FactCheckerAgent(llm).run({
        "argument_a": "차43-2 는 80:20 이다", "argument_b": "차43-2 30:70, 도로교통법 제18조",
    }))
    assert not r.is_valid
    assert any(f.startswith("A측") and "80:20" in f for f in r.flagged_citations)
    assert any("정황에 근거 없음" in f for f in r.flagged_citations)
    assert any(f.startswith("(경고) B측") for f in r.flagged_citations)
    # 규칙 검증 결과가 LLM 프롬프트로도 전달된다
    assert "[error] 차43-2 의 기본 과실비율이 80:20" in llm.prompts[0]


def test_fact_checker_clean_is_valid_even_with_warning():
    r = asyncio.run(FactCheckerAgent(FakeLLM()).run({
        "argument_a": "차43-2 30:70", "argument_b": "차43-2 도로교통법 제18조",
    }))
    assert r.is_valid and r.flagged_citations == ["(경고) B측 도로교통법 제18조 는 인용된 도표(차43-2)의 관련 법규에 없음"]


def test_round2_prompt_splits_fact_check_by_side():
    llm = FakeLLM("차43-2 인용: 차43-2")
    fc = FactCheckResult(is_valid=False, notes="", flagged_citations=["A측 차99-9 는 존재하지 않는 도표번호", "B측 기본 과실 오류"])
    asyncio.run(VehicleAAgent(llm).run({"accident_input": "x", "round": 2, "fact_check": fc}))
    p = llm.prompts[0]
    assert "A측(당신) 지적: A측 차99-9" in p and "B측(상대) 지적: B측 기본 과실 오류" in p


def test_round1_has_no_fact_check_block():
    llm = FakeLLM()
    asyncio.run(VehicleAAgent(llm).run({"accident_input": "x", "round": 1}))
    assert "사실검증 결과" not in llm.prompts[0]
