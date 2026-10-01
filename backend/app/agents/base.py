"""모든 에이전트가 공유하는 베이스 클래스.

각 에이전트는 run(context)만 구현하면 된다. 라운드 순서/병렬 실행은
core/pipeline.py 의 FaultAssessmentPipeline이 담당하므로, 에이전트 자체는
"주어진 컨텍스트로 LLM을 호출해 구조화된 결과를 낸다"는 책임만 진다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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


class ReviewerAgent(BaseAgent):
    """검토위원 3종(기준표/현장정황/유사사례)의 공통 부모.

    perspective만 다르고 평가 로직 골격은 동일하므로 여기서 공유한다.
    """

    perspective: str = "unspecified"

    async def run(self, context: dict[str, Any]):
        from app.models.schemas import ReviewScore

        raw = await self.llm.complete(self.system_prompt, self._build_user_prompt(context))
        # TODO(6~7주차): raw LLM 응답을 실제로 파싱해 fault_ratio_a/b, rationale을 채운다.
        # 지금은 파이프라인이 끝까지 도는지 확인하기 위한 자리표시 구조만 반환한다.
        return ReviewScore(
            reviewer_name=self.name,
            perspective=self.perspective,
            fault_ratio_a=50,
            fault_ratio_b=50,
            rationale=raw,
        )

    def _build_user_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"[{self.perspective} 관점에서 평가]\n"
            f"사고 정황: {context.get('accident_input')}\n"
            f"A/B 논증 라운드: {context.get('rounds')}\n"
            f"사실검증 결과: {context.get('fact_check')}\n"
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

    async def run(self, context: dict[str, Any]) -> AgentArgument:
        round_no = context.get("round", 1)
        user_prompt = (
            f"[참조 기준표]\n{context.get('reference_charts', '(없음)')}\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[상대측({self.other}차량) 이전 주장]\n{context.get('opponent_argument')}\n\n"
            f"현재 라운드: {round_no}"
        )
        content = await self.llm.complete(self.system_prompt, user_prompt)
        return AgentArgument(
            agent_name=self.name,
            round=round_no,
            content=content,
            cited_rules=extract_citations(content),
        )


def extract_citations(text: str) -> list[str]:
    """논증 본문에서 도표번호(차12-1)와 도로교통법 조문을 뽑는다."""
    import re

    charts = re.findall(r"차\d+-\d+", text)
    laws = [f"도로교통법 제{n}조" for n in re.findall(r"도로교통법\s*제(\d+)조", text)]
    return list(dict.fromkeys(charts + laws))
