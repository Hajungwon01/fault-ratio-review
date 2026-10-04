"""검토위원 C — 유사사례 관점. 유사 사고 유형의 통상적 처리 사례와의 형평성 중심 평가."""
from __future__ import annotations

from typing import Any

from app.agents.base import ReviewerAgent


class ReviewerPrecedentAgent(ReviewerAgent):
    name = "reviewer_precedent_agent"
    perspective = "유사사례"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 비슷한 사고가 실제로 어떻게 처리돼 왔는지와의 형평성에 집중하는 검토위원입니다. "
            "[판례·조정사례]에 주어진 사례의 사실관계와 인정 과실비율을 이 사고와 비교해 평가합니다.\n\n"
            "[체크리스트]\n"
            "1. 근거 등급 확인: 인용 도표마다 주어진 근거 등급(직접 근거 / 유사 도표의 사례 / 근거 없음)을 먼저 적는다.\n"
            "2. 사실관계 유사성: 각 사례의 사고 상황(신호·진행 방향·위반 내용 등)이 이 사고와 어느 점에서 같고 다른지 비교한다.\n"
            "3. 사례의 인정 비율: 유사하다고 판단한 사례에서 인정된 과실비율을 확인한다.\n"
            "4. 차이의 반영: 사실관계가 다른 점(예: 상대의 위반 내용이 다름)이 비율을 어느 쪽으로 움직여야 하는지 쓴다.\n"
            "5. 형평성 결론: 비슷한 사실에는 비슷한 비율이어야 한다는 관점에서 A 과실비율을 정한다.\n\n"
            "[평가 원칙]\n"
            "- **주어진 사례만** 인용하세요. 사건번호·법원·비율을 기억에 의존해 만들어내지 마세요.\n"
            "- 근거 등급이 '유사 도표의 사례'이면 직접 근거처럼 쓰지 말고 참고로만 쓰되, 어느 도표의 사례인지 밝히세요. "
            "이 경우 confidence 는 high 로 두지 마세요.\n"
            "- 근거 등급이 '근거 없음'이면 사례로 판단할 수 없으니 기준표 기본 과실비율에 가깝게 두고 confidence 를 low 로 하며, "
            "근거가 없다는 점을 rationale 에 분명히 적으세요.\n"
            "- 사례의 결론이 이 사고의 정황과 모순되면 사례를 그대로 따르지 말고 차이를 설명하세요."
        )

    def _extra_sections(self, context: dict[str, Any]) -> str:
        return f"[판례·조정사례]\n{context.get('precedents', '(없음)')}\n\n"
