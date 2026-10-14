"""질문 단계 — 사용자에게 보여 줄 형태(결과 숨김)와 답변 반영.

사고 당사자는 과실이 적게 나오는 답을 고를 동기가 있으므로, 질문 화면에는 과실비율·도표번호를 보여 주지 않는다.
완화책은 세 가지다: ① 보기에서 도표·과실을 제거 ② 보기 순서를 무작위로 섞음 ③ "모름"을 허용하고 모르면 가능한 결과 범위를 보여 줌.
답변은 사실로 확정하지 않고 "사용자 진술 기준"으로만 취급한다. 이는 왜곡을 줄일 뿐 막지는 못한다.
"""
from __future__ import annotations

import random
import secrets
from typing import Optional

from pydantic import BaseModel

from app.models.schemas import ClarifyingQuestion

UNKNOWN_ID = -1
UNKNOWN_LABEL = "모름 / 확실하지 않음"


def _order(n: int, seed: int) -> list[int]:
    """화면에 보이는 순서 → 원래 보기 번호. 같은 seed 면 항상 같은 순서라 서버가 상태를 저장하지 않아도 답을 되돌릴 수 있다."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def user_view(q: ClarifyingQuestion, seed: Optional[int] = None) -> dict:
    """사용자에게 보여 줄 질문: 보기는 label 만, 순서는 seed 로 섞고, 맨 끝에 '모름'. 도표번호·과실비율·증거 안내는 포함하지 않는다.

    보기 id 는 화면에 보이는 위치(0부터)이며 원래 번호와 무관하다. 답할 때 seed 와 id 를 함께 돌려받는다.
    """
    seed = secrets.randbelow(2**31) if seed is None else seed
    order = _order(len(q.options), seed)
    options = [{"id": pos, "label": q.options[orig].label} for pos, orig in enumerate(order)]
    options.append({"id": UNKNOWN_ID, "label": UNKNOWN_LABEL})
    return {"seed": seed, "question": q.question, "options": options}


class ClarifiedResult(BaseModel):
    """질문에 대한 답을 반영한 결과 (정황 A 기준)."""

    scene_a_low: int
    scene_a_high: int
    chart_no: Optional[str] = None  # 답이 정해졌을 때만 (모름이면 None)
    statement: str                  # "사용자 진술 기준 …" — 사실로 확정하지 않는다는 표기
    evidence: str                   # 판단을 바꿀 수 있는 증거 안내


def resolve(q: ClarifyingQuestion, answer_id: Optional[int], seed: int) -> ClarifiedResult:
    """사용자가 고른 보기(위치 id, 모름/미응답이면 None 또는 -1)를 반영한다. 모르면 가능한 모든 보기의 범위를 돌려준다."""
    if answer_id is None or answer_id == UNKNOWN_ID:
        faults = [o.scene_a_fault for o in q.options]
        return ClarifiedResult(
            scene_a_low=min(faults), scene_a_high=max(faults), chart_no=None,
            statement=f"'{q.question}'에 확실하지 않다고 답해 가능한 경우의 범위로 표시합니다",
            evidence=q.evidence,
        )
    order = _order(len(q.options), seed)
    if not 0 <= answer_id < len(order):
        raise ValueError(f"보기 번호가 범위를 벗어났습니다: {answer_id}")
    opt = q.options[order[answer_id]]
    return ClarifiedResult(
        scene_a_low=opt.scene_a_fault, scene_a_high=opt.scene_a_fault, chart_no=opt.chart_no,
        statement=f"사용자 진술 기준: '{q.question}' → {opt.label} (사실로 확인된 것이 아닙니다)",
        evidence=q.evidence,
    )
