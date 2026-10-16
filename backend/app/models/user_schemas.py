"""사용자(사고 당사자)용 API 스키마. 내부 `FaultReport` 와 달리 보기별 도표·과실, 위원 의견, 플래그 사유를 담지 않는다.

설계: docs/USER_API_DESIGN.md. 이 스키마의 필드만 사용자에게 나가므로, 새 필드를 추가할 때는
"사용자가 유리한 답을 고르는 데 쓸 수 있는 정보인가"를 먼저 따진다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UserOption(BaseModel):
    id: int = Field(..., description="화면에 보이는 위치(0부터). -1 은 '모름'")
    label: str


class UserQuestion(BaseModel):
    seed: int = Field(..., description="보기 순서 섞기용. 답변할 때 그대로 돌려보낸다")
    text: str
    options: list[UserOption]


class ChartBasis(BaseModel):
    """도표 한 장의 근거. 모두 도표 데이터에서 코드가 만든 값이다 (모델이 쓰지 않는다)."""

    chart_no: str
    label: str
    situation: str = Field(..., description="도표가 다루는 사고 상황")
    my_base: int = Field(..., ge=0, le=100, description="나 기준 기본 과실비율 (경우가 여럿이면 최종값에 가장 가까운 경우)")
    other_base: int = Field(..., ge=0, le=100)
    laws: list[str] = Field(default_factory=list, description="관련 법규(최대 3개)")


class UserResult(BaseModel):
    my_fault_low: int = Field(..., ge=0, le=100, description="나의 과실 범위 하한 (사고 설명에서 나=A)")
    my_fault_high: int = Field(..., ge=0, le=100)
    other_fault_low: int = Field(..., ge=0, le=100)
    other_fault_high: int = Field(..., ge=0, le=100)
    assumption: str = Field("", description="질문에 답했을 때 '사용자 진술 기준' 표기. 사실로 확인된 것이 아님")
    summary: str = Field(..., description="사용자용 설명. 모델 문장은 검증을 통과해야 쓰고, 아니면 코드가 만든 문장")
    chart_basis: list[ChartBasis] = Field(default_factory=list, description="적용한 기준표의 근거. 모름이면 비운다(보기별 결과 비공개)")
    evidence_to_check: list[str] = Field(default_factory=list, description="판단을 바꿀 수 있는 증거 안내")
    other_possibilities: str = Field("", description="질문하지 않은 경우의 참고 문구")
    uncertain: bool = Field(..., description="위원 의견이 갈렸는지 — 참이면 전문가 확인 권장")
    disclaimer: str


class AssessResponse(BaseModel):
    status: Literal["needs_answer", "done"]
    session_id: Optional[str] = None
    expires_in_sec: Optional[int] = None
    question: Optional[UserQuestion] = None
    notice: Optional[str] = None
    result: Optional[UserResult] = None


class AnswerRequest(BaseModel):
    option_id: Optional[int] = Field(None, description="고른 보기의 id. 모름은 -1 또는 null, 해당하는 보기가 없으면 -2")
    seed: int
