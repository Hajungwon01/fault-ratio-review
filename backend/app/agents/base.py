"""모든 에이전트가 공유하는 베이스 클래스.

각 에이전트는 run(context)만 구현하면 된다. 라운드 순서/병렬 실행은
core/pipeline.py 의 FaultAssessmentPipeline이 담당하므로, 에이전트 자체는
"주어진 컨텍스트로 LLM을 호출해 구조화된 결과를 낸다"는 책임만 진다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.llm_client import LLMClient


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
