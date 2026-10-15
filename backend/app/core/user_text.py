"""사용자 결과의 설명 생성 — 사실은 코드가 도표 데이터에서 만들고, 모델이 쓴 문장은 검증해서 실패하면 코드 문장으로 대체한다.

설계: docs/USER_API_DESIGN.md §9. 모델이 도표 내용을 지어내거나(없는 번호·숫자), 내부 용어를 노출하거나, 너무 길게 쓰면 그 문장을 버린다.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from app.core.chart_lookup import ChartCatalog, get_catalog
from app.models.user_schemas import ChartBasis

MAX_SENTENCES = 6
MAX_LAWS = 3
UNKNOWN_SUMMARY = "정황에 없는 사실에 따라 결과가 달라질 수 있어 가능한 범위로 표시합니다."

_CHART_NO = re.compile(r"차\s?\d+\s?-\s?\d+")
_INTERNAL_TERMS = re.compile(r"정황\s?[AB]|도표\s?[AB]|위원|확신|플래그|가정|chart|scene")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _norm_chart_no(s: str) -> str:
    return re.sub(r"\s", "", s)


def chart_basis_entry(
    chart_no: str, final_my_fault: int, chart_a_is: Optional[str] = None, catalog: ChartCatalog | None = None
) -> Optional[ChartBasis]:
    """도표 한 장의 근거 블록. 기본 과실비율은 '나' 기준이며, 경우((가)(나)(다))가 여럿이면 최종값에 가장 가까운 경우를 쓴다.

    chart_a_is 가 A/B 이면 그 대응으로, 모르면(질문 보기에는 역할이 없다) 두 방향 모두에서 가장 가까운 값을 고른다.
    """
    catalog = catalog or get_catalog()
    chart = catalog.get(chart_no)
    if chart is None:
        return None
    bases_a = [v["base_ratio"]["a"] for v in chart["variants"]]
    if chart_a_is == "A":
        candidates = bases_a
    elif chart_a_is == "B":
        candidates = [100 - b for b in bases_a]
    else:
        candidates = bases_a + [100 - b for b in bases_a]
    mine = min(candidates, key=lambda b: abs(b - final_my_fault))
    laws = [l["title"] for l in chart.get("laws", [])][:MAX_LAWS]
    return ChartBasis(
        chart_no=chart_no, label=chart["label"], situation=chart["situation"], my_base=mine, other_base=100 - mine, laws=laws
    )


def fallback_summary(basis: list[ChartBasis], low: int, high: int, *, answered: bool = False) -> str:
    """코드가 만드는 설명(모델 문장 대체용). 숫자와 도표는 모두 데이터에서 온 값이다."""
    mine = f"{low}%" if low == high else f"{low}~{high}%"
    if not basis:
        return f"적용할 기준표를 하나로 정하지 못해 검토 결과의 범위만 표시합니다. 나의 과실은 {mine}로 보았습니다."
    first = basis[0]
    lead = (
        f"고르신 답변을 기준으로 {first.chart_no}({first.label})를 적용했습니다."
        if answered
        else f"이 사고는 '{first.situation}'에 해당하는 {first.chart_no}를 적용했습니다."
    )
    return f"{lead} 기본 과실비율은 나 {first.my_base} : 상대 {first.other_base}이고, 검토 결과 나의 과실은 {mine}로 보았습니다."


def _numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", _CHART_NO.sub("", text))}


def validate_user_summary(text: str, basis: list[ChartBasis], low: int, high: int) -> list[str]:
    """모델이 쓴 사용자용 설명의 문제 목록 (비어 있으면 통과)."""
    text = (text or "").strip()
    if not text:
        return ["비어 있음"]
    problems: list[str] = []

    allowed_charts = {_norm_chart_no(b.chart_no) for b in basis}
    bad_charts = {_norm_chart_no(c) for c in _CHART_NO.findall(text)} - allowed_charts
    if bad_charts:
        problems.append(f"허용되지 않은 도표번호 {sorted(bad_charts)}")

    allowed_nums = {0, 100, low, high, 100 - low, 100 - high}
    for b in basis:
        allowed_nums |= {b.my_base, b.other_base}
        for law in b.laws:
            allowed_nums |= _numbers(law)
    bad_nums = _numbers(text) - allowed_nums
    if bad_nums:
        problems.append(f"주어지지 않은 숫자 {sorted(bad_nums)}")

    if (m := _INTERNAL_TERMS.search(text)):
        problems.append(f"내부 용어 '{m.group(0)}'")
    if len([s for s in _SENTENCE_SPLIT.split(text) if s.strip()]) > MAX_SENTENCES:
        problems.append(f"{MAX_SENTENCES}문장 초과")
    return problems


def choose_summary(model_text: str, basis: list[ChartBasis], low: int, high: int, *, answered: bool = False) -> str:
    """모델 문장이 검증을 통과하면 그것을, 아니면 코드 문장을 쓴다."""
    if model_text and not validate_user_summary(model_text, basis, low, high):
        return model_text.strip()
    return fallback_summary(basis, low, high, answered=answered)


def unique_basis(entries: Iterable[Optional[ChartBasis]]) -> list[ChartBasis]:
    seen, out = set(), []
    for e in entries:
        if e and e.chart_no not in seen:
            seen.add(e.chart_no)
            out.append(e)
    return out
