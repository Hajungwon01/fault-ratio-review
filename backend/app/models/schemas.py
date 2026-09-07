"""과실비율 심사 파이프라인 전반에서 쓰는 공통 스키마.

TODO(3~4주차): AccidentType을 실제 과실비율정보포털 사고 유형 코드 체계로 정교화.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AccidentType(str, Enum):
    """사고 유형 (1차 버전 — 대표 유형만 우선 정의)."""

    STRAIGHT_VS_LEFT_TURN = "straight_vs_left_turn"       # 직진 vs 좌회전
    LANE_CHANGE = "lane_change"                            # 차선변경
    REAR_END_COLLISION = "rear_end_collision"               # 후행 추돌
    SIGNAL_VIOLATION = "signal_violation"                   # 신호위반 관련
    OTHER = "other"


class SignalState(str, Enum):
    A_GREEN_B_GREEN = "a_green_b_green"
    A_GREEN_B_RED = "a_green_b_red"
    A_RED_B_GREEN = "a_red_b_green"
    UNKNOWN = "unknown"


class AccidentInput(BaseModel):
    """사용자가 입력하는 사고 정황."""

    accident_type: AccidentType
    signal_state: SignalState = SignalState.UNKNOWN
    lane_info: Optional[str] = Field(None, description="차선 상태 (예: 2차로 중 1차로 주행 중 변경)")
    speed_a_kmh: Optional[float] = Field(None, description="A차량 추정 속도(km/h)")
    speed_b_kmh: Optional[float] = Field(None, description="B차량 추정 속도(km/h)")
    description: str = Field(..., description="사고 상황 상세 설명 (블랙박스 요약 포함 가능)")
    image_urls: list[str] = Field(default_factory=list, description="블랙박스 이미지 (9~10주차부터 사용)")


class AgentArgument(BaseModel):
    """A/B차량측 에이전트의 라운드별 논증."""

    agent_name: str
    round: int
    content: str
    cited_rules: list[str] = Field(default_factory=list, description="인용한 기준표/법조항")


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
