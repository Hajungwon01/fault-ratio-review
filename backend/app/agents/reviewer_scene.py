"""검토위원 B — 현장 정황 관점. 시야 확보, 속도, 반응 가능성 등 현장 조건 중심 평가."""
from __future__ import annotations

from app.agents.base import ReviewerAgent


class ReviewerSceneAgent(ReviewerAgent):
    name = "reviewer_scene_agent"
    perspective = "현장 정황"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 시야 확보, 속도, 반응 가능성 등 현장 조건을 중심으로 사고를 평가하는 검토위원입니다.\n"
            "TODO(6~7주차): 현장 정황 반영도 체크리스트 기반 스코어링 로직 구체화."
        )
