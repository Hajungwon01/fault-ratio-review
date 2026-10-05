"""과실비율 심사 파이프라인 전반에서 쓰는 공통 스키마.

입력은 자유 서술 중심(AccidentInput.description)이며, 사고 유형 코드 체계는 쓰지 않는다 —
도표 매칭은 core/chart_lookup.py + 에이전트의 도구 조회가 담당한다.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

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


class ChecklistItem(BaseModel):
    """관점별 체크리스트 한 항목의 판단 결과 (설명 가능성·위원 간 비교용)."""

    item: str = Field(..., description="체크리스트 항목 이름 (프롬프트에 주어진 항목 그대로)")
    finding: str = Field(..., description="이 항목에 대한 판단. 정보가 없으면 '정황 정보 없음'이라고 쓴다")


class ReviewerVerdict(BaseModel):
    """검토위원 LLM 이 구조화 출력으로 돌려주는 평가 (reviewer_name/perspective 는 코드가 채운다).

    설계 두 가지 (첫 실측에서 위원 3명이 근거 문장은 "A 과실 없음"인데 숫자는 A 100 으로 쓴 오류에서 나왔다):

    1. **필드 순서 = 모델이 생성하는 순서.** 근거(checklist, rationale)를 먼저 쓰고 숫자는 맨 뒤에 둔다.
       숫자를 앞에 두면 모델이 답부터 확정하고 근거를 끼워 맞춘다.
    2. **같은 결론을 두 이름으로 받아 교차 검증한다.** 정황 A 기준(scene_a_fault)과 도표 기준(chart_a_is, chart_a_fault)을
       모두 답하게 하고, 코드가 둘을 환산해 비교한다(verdict_check). 최종값은 scene_a_fault. B 과실은 100 - A 라 묻지 않는다.
       (처음에는 도표 기준만 받아 코드가 환산했으나, 모델이 정황 값을 도표 칸에 넣어 이중 변환되는 오류가 가장 많았다.)
    """

    checklist: list[ChecklistItem] = Field(default_factory=list, description="관점별 체크리스트 항목별 판단")
    rationale: str = Field(..., min_length=1, description="평가 근거. 적용한 도표·가감요소·정황을 구체적으로")
    scene_a_fault: int = Field(
        ..., ge=0, le=100,
        description="**정황의 A(나의 차량)** 입장의 최종 과실비율(0~100 정수). 최종 결론이며 위원 집계에 쓰인다",
    )
    chart_no: str = Field(..., description='적용한 도표번호. 예: "차43-2". 도표를 적용하지 않았으면 빈 문자열')
    chart_a_is: Literal["A", "B", "none"] = Field(
        ..., description='도표 안의 A차량이 정황의 A(나의 차량)이면 "A", 정황의 B(상대 차량)이면 "B". 도표를 적용하지 않았으면 "none"'
    )
    chart_a_fault: int = Field(
        ..., ge=0, le=100,
        description="**도표의 A차량 입장**에서의 최종 과실비율(0~100 정수): 도표 A의 기본 과실에 가감요소를 반영한 값. "
                    "정황의 A 로 환산하지 말 것(환산은 시스템이 한다). chart_a_is 가 none 이면 정황 A 의 과실비율",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="이 평가에 대한 확신. 근거 도표가 불명확하거나 정황 정보가 부족하면 low"
    )
    cited_rules: list[str] = Field(default_factory=list, description="인용한 도표번호·조문 (예: 차43-2, 도로교통법 제19조)")

    @property
    def fault_ratio_a(self) -> int:
        """정황 A(나의 차량) 기준 과실비율 = 모델이 직접 적은 scene_a_fault.

        실측에서 모델은 근거 문장에는 정황 기준 값을 맞게 쓰고도 도표 기준 칸에 같은 값을 넣어 코드 변환이 한 번 더 뒤집는
        오류가 가장 많았다(오답 위원 답 9개 중 6개). 그래서 최종값은 정황 기준 값을 쓰고, 도표 기준 답은 교차 검증에만 쓴다.
        """
        return self.scene_a_fault

    @property
    def converted_fault_a(self) -> int:
        """도표 기준 답(chart_a_is, chart_a_fault)을 정황 A 기준으로 환산한 값 — scene_a_fault 와 같아야 한다."""
        return 100 - self.chart_a_fault if self.chart_a_is == "B" else self.chart_a_fault


class ReviewScore(BaseModel):
    """검토위원 1인의 독립 평가."""

    reviewer_name: str
    perspective: str  # "기준표 적용" | "현장 정황" | "유사사례"
    fault_ratio_a: int = Field(..., ge=0, le=100)
    fault_ratio_b: int = Field(..., ge=0, le=100)
    rationale: str
    confidence: Literal["high", "medium", "low"] = "medium"
    cited_rules: list[str] = Field(default_factory=list)
    checklist: list[ChecklistItem] = Field(default_factory=list)
    chart_no: str = ""
    chart_a_is: Literal["A", "B", "none"] = "none"
    chart_a_fault: Optional[int] = None  # 도표 A 입장의 과실 (변환 전 값, 검증·분석용)
    scene_a_fault: Optional[int] = None  # 모델이 직접 적은 정황 A 기준 과실 (fault_ratio_a 와 같다)
    warnings: list[str] = Field(default_factory=list, description="도표 기본 비율과의 불일치 등 이 평가의 신뢰 문제")


class FaultReport(BaseModel):
    """진행 에이전트가 생성하는 최종 "몇 대 몇" 심사 리포트."""

    fault_ratio_a: int = Field(..., ge=0, le=100)
    fault_ratio_b: int = Field(..., ge=0, le=100)
    basis: str
    deviation: float = Field(..., description="검토위원 간 과실비율 편차(표준편차 등)")
    spread: int = Field(0, description="검토위원 A과실비율의 최대-최소 차이(%p). 표준편차가 이상치 한 명을 놓치는 것을 보완")
    needs_precise_review: bool = Field(
        ..., description="편차가 임계값을 넘어 정밀 심사가 필요한 사고인지 여부"
    )
    rounds: list[AgentArgument] = Field(default_factory=list)
    fact_check: Optional[FactCheckResult] = None
    reviews: list[ReviewScore] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="심사 범위 관련 경고 등 사용자에게 알릴 유의 사항")
