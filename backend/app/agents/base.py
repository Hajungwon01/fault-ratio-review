"""모든 에이전트가 공유하는 베이스 클래스.

각 에이전트는 run(context)만 구현하면 된다. 라운드 순서/병렬 실행은
core/pipeline.py 의 FaultAssessmentPipeline이 담당하므로, 에이전트 자체는
"주어진 컨텍스트로 LLM을 호출해 구조화된 결과를 낸다"는 책임만 진다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.chart_tools import build_chart_tools
from app.core.llm_client import LLMClient
from app.models.schemas import AgentArgument


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """역할별 시스템 프롬프트. TODO(3~4주차부터 순차 작성)."""

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> Any:
        """context를 받아 이 에이전트의 산출물을 반환한다."""


def format_rounds(rounds: list | None) -> str:
    """A/B 논증 기록을 위원이 읽기 쉬운 텍스트로 만든다 (pydantic repr 노출 방지)."""
    if not rounds:
        return "(없음)"
    blocks = []
    for r in rounds:
        side = "A" if "vehicle_a" in r.agent_name else "B" if "vehicle_b" in r.agent_name else r.agent_name
        cited = f"\n인용: {', '.join(r.cited_rules)}" if r.cited_rules else ""
        blocks.append(f"[{side}측 {r.round}라운드]\n{r.content}{cited}")
    return "\n\n".join(blocks)


def format_fact_check(fc) -> str:
    if fc is None:
        return "(없음)"
    head = "인용 오류 없음" if fc.is_valid else "인용 오류 있음"
    flagged = "\n".join(f"- {f}" for f in fc.flagged_citations) or "- 지적 사항 없음"
    return f"{head}\n{flagged}"


class ReviewerAgent(BaseAgent):
    """검토위원 3종(기준표/현장정황/유사사례)의 공통 부모.

    perspective만 다르고 평가 로직 골격은 동일하므로 여기서 공유한다.
    응답은 ReviewerVerdict 스키마로 구조화 출력을 받고(파싱 실패 시 LLMClient 가 재시도),
    B 과실은 100 - A 로 계산한다. 관점별 체크리스트 프롬프트는 서브클래스의 system_prompt (#8~#10).
    """

    perspective: str = "unspecified"

    # 모든 위원에게 공통으로 붙는 출력 규칙. 관점별 프롬프트는 그대로 두고 뒤에 덧붙인다.
    OUTPUT_RULES = (
        "\n\n[출력 규칙]\n"
        "- 반드시 지정된 JSON 스키마로만 답하세요.\n"
        "- fault_ratio_a 는 A차량 과실비율(0~100 정수)이며 B 과실은 100 - A 로 자동 계산됩니다.\n"
        "- A/B 에이전트의 주장은 서로 유리하게 쓴 것이므로 그대로 따르지 말고, 당신의 관점에서 독립적으로 평가하세요.\n"
        "- 사실검증이 지적한 오류가 있는 주장은 근거로 삼지 마세요.\n"
        "- 도표·정황 정보가 부족해 판단이 불확실하면 confidence 를 low 로 두세요.\n"
        "- checklist 에는 위 [체크리스트] 항목을 빠짐없이, 항목 이름을 그대로 써서 항목마다 하나씩 판단을 적으세요. "
        "정황에 없는 사실은 추측하지 말고 '정황 정보 없음'이라고 쓰세요.\n"
        "- 주어진 [인용 도표 상세]·[판례·조정사례] 밖의 도표번호·판례·비율을 만들어 인용하지 마세요."
    )

    def _extra_sections(self, context: dict[str, Any]) -> str:
        """관점별로 사용자 프롬프트에 덧붙일 자료. 서브클래스가 재정의한다."""
        return ""

    async def run(self, context: dict[str, Any]):
        from app.models.schemas import ReviewerVerdict, ReviewScore

        verdict = await self.llm.complete_json(
            self.system_prompt + self.OUTPUT_RULES,
            self._build_user_prompt(context),
            ReviewerVerdict,
            mock={
                "fault_ratio_a": 50,
                "confidence": "low",
                "rationale": f"[MOCK {self.perspective}] 실제 LLM 이 연결되지 않아 자리표시 평가를 반환합니다.",
                "cited_rules": [],
            },
        )
        return ReviewScore(
            reviewer_name=self.name,
            perspective=self.perspective,
            fault_ratio_a=verdict.fault_ratio_a,
            fault_ratio_b=100 - verdict.fault_ratio_a,
            rationale=verdict.rationale,
            confidence=verdict.confidence,
            cited_rules=verdict.cited_rules,
            checklist=verdict.checklist,
        )

    def _build_user_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"[{self.perspective} 관점에서 평가]\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[인용 도표 상세]\n{context.get('cited_charts', '(없음)')}\n\n"
            f"{self._extra_sections(context)}"
            f"[A/B 논증]\n{format_rounds(context.get('rounds'))}\n\n"
            f"[사실검증 결과]\n{format_fact_check(context.get('fact_check'))}\n"
        )


class AdvocateAgent(BaseAgent):
    """A/B차량측 에이전트 공통 부모. side("A"/"B")만 다르고 논증 골격은 대칭이다."""

    side: str = "A"

    @property
    def other(self) -> str:
        return "B" if self.side == "A" else "A"

    @property
    def system_prompt(self) -> str:
        me, other = self.side, self.other
        return (
            f"당신은 {me}차량측을 대리해 과실비율 심사에서 {me}차량의 과실을 최소화하는 논증을 만드는 에이전트입니다.\n\n"
            "[규칙]\n"
            "1. 반드시 [참조 기준표]에 주어진 도표의 기본 과실비율·가감요소·법규만 근거로 사용하세요. "
            "목록에 없는 도표번호, 비율, 조문을 지어내지 마세요.\n"
            f"2. 입력된 사고 정황에 실제로 있는 사실만 가감요소 근거로 삼으세요. "
            f"{other}차량의 가산 사유({other} 쪽 가감요소)와 {me}차량의 감산 사유({me} 쪽 음수 가감요소)를 정황에서 찾아 제시하세요.\n"
            "3. 가장 적합한 도표 하나와 그 경우((가)(나)(다) 등)를 선택하고, 선택 이유를 밝히세요. "
            "정황 정보가 부족해 경우를 확정할 수 없으면 그 점을 명시하세요.\n"
            f"4. 2라운드에는 상대측 주장의 근거 오류(잘못된 도표 선택, 정황에 없는 가감요소)를 짚어 반박하고, 자기 주장을 보강하세요.\n"
            f"5. 마지막 줄에 `인용: 차XX-X, 도로교통법 제N조` 형식으로 인용한 도표·조문을 쉼표로 나열하세요.\n"
            f"6. 당신은 {me}차량측 변호인입니다. {me}에 유리한 해석을 하되, 사실을 왜곡하면 사실검증에서 감점됩니다."
        )

    def _fact_check_block(self, context: dict[str, Any]) -> str:
        """2라운드: 사실검증이 지적한 사항을 측별로 나눠 전달한다 (1라운드에는 없음)."""
        fc = context.get("fact_check")
        if fc is None:
            return ""
        mine = [f for f in fc.flagged_citations if f.removeprefix("(경고) ").startswith(f"{self.side}측")]
        theirs = [f for f in fc.flagged_citations if f.removeprefix("(경고) ").startswith(f"{self.other}측")]
        return (
            "[사실검증 결과]\n"
            f"- {self.side}측(당신) 지적: {'; '.join(mine) or '없음'}  → 틀린 인용은 정정하고, 정정 불가한 주장은 철회하세요.\n"
            f"- {self.other}측(상대) 지적: {'; '.join(theirs) or '없음'}  → 상대 주장의 약점으로 활용하세요.\n\n"
        )

    async def run(self, context: dict[str, Any]) -> AgentArgument:
        round_no = context.get("round", 1)
        user_prompt = (
            f"[참조 기준표]\n{context.get('reference_charts', '(없음)')}\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[상대측({self.other}차량) 이전 주장]\n{context.get('opponent_argument')}\n\n"
            f"{self._fact_check_block(context)}"
            f"현재 라운드: {round_no}"
        )
        trace: list[str] = []
        content = await self.llm.run_agent(self.system_prompt, user_prompt, build_chart_tools(trace))
        return AgentArgument(
            agent_name=self.name,
            round=round_no,
            content=content,
            cited_rules=extract_citations(content),
            tool_calls=trace,
        )


def extract_citations(text: str) -> list[str]:
    """논증 본문에서 도표번호(차12-1)와 도로교통법 조문을 뽑는다."""
    import re

    charts = re.findall(r"차\d+-\d+", text)
    laws = [f"도로교통법 제{n}조" for n in re.findall(r"도로교통법\s*제(\d+)조", text)]
    return list(dict.fromkeys(charts + laws))
