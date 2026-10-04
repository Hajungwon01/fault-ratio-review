"""심사 지원 범위 판별.

현재 지원: 자동차 대 자동차 사고 (이륜차·긴급자동차 포함 — 기준표 차1~차61).
지원 안 함: 보행자 / 자전거 / 개인형 이동장치(PM, 킥보드) 상대 사고, 상대 차량이 없는 단독사고.
  → 기준표(자동차 대 보행자·자전거, 비정형 PM)를 수집하지 않았기 때문이다.

판정은 키워드 기반이라 단정하지 않는다.
- unsupported: 지원 밖 키워드가 있고, 상대 차량을 가리키는 표현은 없음 → 심사하지 않고 안내
- uncertain:   지원 밖 키워드가 있지만 상대 차량 표현도 있음 (예: "보행자를 피하다 옆 차와 충돌")
               → 차량 간 사고로 보고 진행하되 리포트에 경고를 남김
- supported:   해당 없음
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import AccidentInput

_PORTAL = "손해보험협회 과실비율정보포털(accident.knia.or.kr)"

# (분류, 정규식, 안내)
_OUT_OF_SCOPE: list[tuple[str, re.Pattern, str]] = [
    ("보행자", re.compile(r"보행자|행인|횡단\s*중인\s*(사람|노인|아이|어린이)|사람을\s*(치|쳤|받)"),
     f"보행자 사고는 아직 지원하지 않습니다. {_PORTAL}의 '자동차와 보행자의 사고'에서 확인하세요."),
    ("자전거", re.compile(r"자전거"),
     f"자전거 사고는 아직 지원하지 않습니다. {_PORTAL}의 '자동차와 자전거의 사고'에서 확인하세요."),
    ("개인형 이동장치", re.compile(r"킥보드|킥라니|개인형\s*이동\s*장치|전동\s*휠|세그웨이|\bPM\b"),
     f"개인형 이동장치(PM·킥보드) 사고는 아직 지원하지 않습니다. {_PORTAL}의 '비정형 과실비율'을 참고하세요."),
    ("단독사고", re.compile(r"단독\s*사고|1인\s*사고|상대(방)?\s*(차량|차)?\s*(은|는)?\s*없"),
     "상대 차량이 없는 단독사고는 과실비율 심사 대상이 아닙니다."),
]
_COUNTERPART_VEHICLE = re.compile(r"상대(방)?\s*(차량|차|자동차)|B\s*차량|차량\s*B|자동차\s*B|\(B\)")


@dataclass(frozen=True)
class ScopeResult:
    status: str  # "supported" | "unsupported" | "uncertain"
    category: str = ""
    message: str = ""


class OutOfScopeError(Exception):
    """지원 범위 밖 입력. API 계층에서 422 로 변환한다."""

    def __init__(self, result: ScopeResult) -> None:
        super().__init__(result.message)
        self.result = result


def check_scope(accident: AccidentInput) -> ScopeResult:
    text = " ".join([accident.description, accident.lane_info or ""])
    for category, pattern, message in _OUT_OF_SCOPE:
        if pattern.search(text):
            if category != "단독사고" and _COUNTERPART_VEHICLE.search(text):
                return ScopeResult(
                    "uncertain",
                    category,
                    f"설명에 '{category}' 관련 표현이 있으나 상대 차량이 있어 차량 간 사고로 보고 심사했습니다. "
                    f"{category}가 직접 충돌 상대라면 결과를 신뢰할 수 없습니다.",
                )
            return ScopeResult("unsupported", category, message)
    return ScopeResult("supported")
