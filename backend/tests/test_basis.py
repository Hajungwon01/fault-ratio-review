"""진행 에이전트의 근거(basis) 요약 — 모델에게 pydantic repr 대신 읽기 쉬운 텍스트를 주고, 숫자는 코드가 만든다 (#12)."""
import asyncio

from app.agents.orchestrator import ProceedingAgent, basis_header
from app.core.llm_client import LLMClient
from app.core.pipeline import FaultAssessmentPipeline
from app.models.schemas import AccidentInput, FactCheckResult, ReviewScore


def review(name, fault, chart="차2-6", conf="high", assumed=(), warnings=()):
    return ReviewScore(reviewer_name=f"{name}_agent", perspective=name, fault_ratio_a=fault, fault_ratio_b=100 - fault,
                       rationale=f"{name} 근거 문장", chart_no=chart, chart_a_is="B", scene_a_fault=fault, confidence=conf,
                       assumed_facts=list(assumed), warnings=list(warnings))


REVIEWS = [review("기준표 적용", 10, assumed=["상대 방향 신호가 녹색이었다"]), review("현장 정황", 20, conf="medium"),
           review("유사사례", 10, chart="차2-2", warnings=["도표 기본 비율과 어긋남"])]


class Spy(LLMClient):
    def __init__(self, reply="요약 문단입니다.", fail=False):
        super().__init__()
        self.reply, self.fail, self.calls = reply, fail, []

    @property
    def mock_mode(self):
        return False

    async def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


def run(llm, reviews=REVIEWS, fact_check=None):
    return asyncio.run(ProceedingAgent(llm).run({"reviews": reviews, "rounds": [], "fact_check": fact_check}))


# ---------- 모델에게 주는 프롬프트 ----------
def test_prompt_is_readable_text_not_pydantic_repr():
    llm = Spy()
    run(llm, fact_check=FactCheckResult(is_valid=False, notes="", flagged_citations=["B측 차2-2 의 비율이 틀림"]))
    system, user = llm.calls[0]
    for repr_leak in ("ReviewScore(", "reviewer_name=", "fault_ratio_a=", "checklist=[", "cited_rules="):
        assert repr_leak not in user
    assert "나(A) 10 : 상대(B) 90" in user                                    # 최종 결론(중앙값 10)
    assert "- 기준표 적용 (확신 high): 도표 차2-6, A 10" in user
    assert "가정한 사실: 상대 방향 신호가 녹색이었다" in user and "가정한 사실: 없음" in user
    assert "기준표 적용 근거 문장" in user
    assert "B측 차2-2 의 비율이 틀림" in user                                  # 사실검증 지적도 전달


def test_system_prompt_forbids_new_numbers_and_requires_assumptions():
    system = ProceedingAgent(Spy()).system_prompt
    for must in ("직접 판정하거나 비율을 바꾸지 않고", "4~6문장", "가정한 사실", "만들어 쓰지 마세요", "나의 차량=A"):
        assert must in system


# ---------- basis 구성 ----------
def test_basis_is_code_header_plus_model_narrative():
    r = run(Spy(reply="  세 위원이 대체로 일치합니다.  "))
    lines = r.basis.split("\n")
    assert lines[0] == "최종(위원 중앙값): 나(A) 10 : 상대(B) 90"
    assert lines[1] == "- 기준표 적용: A 10 (도표 차2-6, 확신 high) / 가정한 사실: 상대 방향 신호가 녹색이었다"
    assert lines[2] == "- 현장 정황: A 20 (도표 차2-6, 확신 medium)"
    assert lines[3] == "- 유사사례: A 10 (도표 차2-2, 확신 high) / 평가 의심 1건"
    assert r.basis.endswith("세 위원이 대체로 일치합니다.") and "\n\n세 위원" in r.basis      # 앞뒤 공백 정리


def test_summary_failure_keeps_the_assessment_with_header_only():
    r = run(Spy(fail=True))
    assert r.fault_ratio_a == 10 and r.basis == basis_header(REVIEWS, 10, 90)


def test_header_numbers_do_not_depend_on_the_model():
    r = run(Spy(reply="최종은 A 99 : B 1 입니다"))                              # 모델이 엉뚱한 숫자를 써도
    assert r.basis.startswith("최종(위원 중앙값): 나(A) 10 : 상대(B) 90") and r.fault_ratio_a == 10   # 리포트 숫자와 헤더는 코드 값


def test_two_reviewers_header_and_mock_pipeline():
    h = basis_header([review("기준표 적용", 30), review("현장 정황", 70)], 50, 50)
    assert h.splitlines()[0] == "최종(위원 중앙값): 나(A) 50 : 상대(B) 50"
    r = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(description="녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다")))
    assert r.basis.startswith("최종(위원 중앙값): 나(A) 50 : 상대(B) 50") and "[MOCK" in r.basis


# ---------- 사용자용 설명은 진행 에이전트가 만들지 않는다 (별도 호출, core/user_summary.py) ----------
def test_proceeding_agent_makes_only_the_internal_basis_in_one_call():
    llm = Spy(reply="내부 요약")
    r = run(llm)
    assert len(llm.calls) == 1 and r.user_summary == ""                    # 사용자용 설명은 여기서 만들지 않는다
    system = llm.calls[0][0]
    assert "user_summary" not in system and "JSON" not in system          # 두 문단을 한 호출로 받지 않는다
