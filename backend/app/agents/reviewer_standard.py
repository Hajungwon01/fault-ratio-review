"""검토위원 A — 기준표 적용 관점. 표준 과실비율 인정기준과의 정합성 중심 평가."""
from __future__ import annotations

from app.agents.base import ReviewerAgent


class ReviewerStandardAgent(ReviewerAgent):
    name = "reviewer_standard_agent"
    perspective = "기준표 적용"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 표준 과실비율 인정기준표와의 정합성을 중심으로 사고를 평가하는 검토위원입니다.\n"
            "TODO(6~7주차): 기준표 적합도 체크리스트 기반 스코어링 로직 구체화."
        )
