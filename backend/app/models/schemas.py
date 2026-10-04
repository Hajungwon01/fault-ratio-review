"""과실비율 심사 파이프라인 전반에서 쓰는 공통 스키마.

입력은 자유 서술 중심(AccidentInput.description)이며, 사고 유형 코드 체계는 쓰지 않는다 —
도표 매칭은 core/chart_lookup.py + 에이전트의 도구 조회가 담당한다.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AccidentType(str, Enum):
    """사고 유형 **힌트** (선택 입력). 도표 후보 검색 점수에만 가볍게 반영되고, 정답을 정하지 않는다.

    실제 사고 유형 판단은 description(자유 서술)과 기준표 도구 조회로 한다.
    """

    STRAIGHT_VS_LEFT_TURN = "straight_vs_left_turn"       # 직진 vs 좌회전
    LANE_CHANGE = "lane_change"                            # 차선변경
    REAR_END_COLLISION = "rear_end_collision"               # 후행 추돌
    SIGNAL_VIOLATION = "signal_violation"                   # 신호위반 관련
    OTHER = "other"


class SignalState(str, Enum):
    A_GREEN_B_GREEN = "a_green_b_green"
    A_GREEN_B_RED = "a_green_b_red"
    A_RED_B_GREEN = "a_red_b_green"
    A_RED_B_RED = "a_red_b_red"
    UNKNOWN = "unknown"


_SIGNAL_TEXT = {
    SignalState.A_GREEN_B_GREEN: "A 녹색 / B 녹색",
    SignalState.A_GREEN_B_RED: "A 녹색 / B 적색",
    SignalState.A_RED_B_GREEN: "A 적색 / B 녹색",
    SignalState.A_RED_B_RED: "A 적색 / B 적색",
}


class AccidentInput(BaseModel):
    """사용자가 입력하는 사고 정황. **자유 서술(description)이 중심**이고 나머지는 선택 보조 정보다.

    description 작성 규칙: 나의 차량을 A, 상대 차량을 B로 쓴다
    (예: "제가(A) 1차선에서 직진 중 옆 차선 상대 차량(B)이 갑자기 앞으로 차선을 변경해 충돌했습니다").
    """

    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="사고 상황 상세 설명. 나의 차량=A, 상대 차량=B. 블랙박스 요약 포함 가능",
    )
    accident_type: Optional[AccidentType] = Field(None, description="사고 유형 힌트 (선택)")
    signal_state: SignalState = SignalState.UNKNOWN
    lane_info: Optional[str] = Field(None, max_length=200, description="차선 상태 (예: 2차로 중 1차로 주행 중 변경)")
    speed_a_kmh: Optional[float] = Field(None, ge=0, le=300, description="A차량 추정 속도(km/h)")
    speed_b_kmh: Optional[float] = Field(None, ge=0, le=300, description="B차량 추정 속도(km/h)")
    image_urls: list[str] = Field(default_factory=list, description="블랙박스 이미지 (Stretch, #22)")

    def to_prompt_text(self) -> str:
        """에이전트 프롬프트에 넣을 읽기 쉬운 사고 정황 텍스트 (enum/None 같은 파이썬 표현을 노출하지 않는다)."""
        lines = [f"사고 상황 설명: {self.description.strip()}"]
        if self.signal_state is not SignalState.UNKNOWN:
            lines.append(f"신호 상태: {_SIGNAL_TEXT[self.signal_state]}")
        if self.lane_info:
            lines.append(f"차선 상태: {self.lane_info.strip()}")
        if self.speed_a_kmh is not None:
            lines.append(f"A차량 추정 속도: {self.speed_a_kmh:g}km/h")
        if self.speed_b_kmh is not None:
            lines.append(f"B차량 추정 속도: {self.speed_b_kmh:g}km/h")
        return "\n".join(lines)


class AgentArgument(BaseModel):
    """A/B차량측 에이전트의 라운드별 논증."""

    agent_name: str
    round: int
    content: str
    cited_rules: list[str] = Field(default_factory=list, description="인용한 기준표/법조항")
    tool_calls: list[str] = Field(default_factory=list, description="이 논증을 만들며 실제로 호출한 도구 기록")


class FactCheckResult(BaseModel):
    """사실검증 에이전트의 검증 결과."""

    is_valid: bool
    notes: str
    flagged_citations: list[str] = Field(default_factory=list)


class ReviewScore(BaseModel):
    """검토위원 1인의 독립 평가."""

    reviewer_name: str
    perspective: str  # "기준표 적용" | "현장 정황" | "유사사례"
    fault_ratio_a: int = Field(..., ge=0, le=100)
    fault_ratio_b: int = Field(..., ge=0, le=100)
    rationale: str


class FaultReport(BaseModel):
    """진행 에이전트가 생성하는 최종 "몇 대 몇" 심사 리포트."""

    fault_ratio_a: int = Field(..., ge=0, le=100)
    fault_ratio_b: int = Field(..., ge=0, le=100)
    basis: str
    deviation: float = Field(..., description="검토위원 간 과실비율 편차(표준편차 등)")
    needs_precise_review: bool = Field(
        ..., description="편차가 임계값을 넘어 정밀 심사가 필요한 사고인지 여부"
    )
    rounds: list[AgentArgument] = Field(default_factory=list)
    fact_check: Optional[FactCheckResult] = None
    reviews: list[ReviewScore] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="심사 범위 관련 경고 등 사용자에게 알릴 유의 사항")
