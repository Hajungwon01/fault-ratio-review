"""검토위원 C — 유사사례 관점. 유사 사고 유형의 통상적 처리 사례와의 형평성 중심 평가."""
from __future__ import annotations

from app.agents.base import ReviewerAgent


class ReviewerPrecedentAgent(ReviewerAgent):
    name = "reviewer_precedent_agent"
    perspective = "유사사례"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 유사 사고 유형의 통상적 처리 사례와의 형평성을 중심으로 사고를 평가하는 검토위원입니다.\n"
            "TODO(6~7주차): 유사사례 형평성 체크리스트 기반 스코어링 로직 구체화."
        )
