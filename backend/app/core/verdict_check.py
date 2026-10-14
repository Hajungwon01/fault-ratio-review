"""검토위원 평가를 도표 데이터와 대조해 앞뒤가 안 맞는 값을 잡는다 (LLM 호출 없음).

첫 실측에서 위원 3명이 근거 문장은 "정차한 A차량은 과실이 없다"인데 숫자는 100으로 쓴 오류가 났고,
3명이 똑같이 틀려 편차 0 이라 기존 플래그가 전혀 반응하지 않았다. 도표 기본 비율은 우리가 가진 데이터이므로
모델이 낸 숫자를 그것과 대조하면 "가감요소로 설명할 수 없는 값"은 코드로 걸러낼 수 있다.

한계: 도표의 A가 정황의 누구인지(chart_a_is)를 모델이 거꾸로 답하면서 숫자도 그에 맞춰 쓰면 이 검사로는 못 잡는다
(역할 판단 자체의 오류). 그것은 A/B 에이전트의 대응 명시, 사실검증(5), 위원 간 불일치로 잡는다.
"""
from __future__ import annotations

from app.core.chart_lookup import ChartCatalog, get_catalog
from app.models.schemas import ReviewerVerdict

# 가감요소는 보통 ±10~20, 겹치면 ±30~40 까지 가능하다. 이보다 크게 벗어나면 가감요소가 아니라 값/역할 오류로 본다.
MAX_DEVIATION_FROM_BASE = 40


def check_verdict(v: ReviewerVerdict, catalog: ChartCatalog | None = None) -> list[str]:
    """문제 목록 (비어 있으면 정상). 문제가 있으면 호출 측이 신뢰도를 낮추고 경고·정밀 심사 표시를 한다."""
    catalog = catalog or get_catalog()
    chart_no = v.chart_no.strip()
    problems: list[str] = []

    if not chart_no and v.chart_a_is != "none":
        problems.append(f"도표번호 없이 도표 역할 대응(chart_a_is={v.chart_a_is})을 답함")
    if chart_no and v.chart_a_is == "none":
        problems.append(f"도표 {chart_no} 를 적용했다면서 도표 A 가 정황의 누구인지 답하지 않음")
    if not chart_no:
        return problems

    if v.chart_a_is == "none":
        chart = catalog.get(chart_no)
        return problems + ([f"존재하지 않는 도표번호 {chart_no}"] if chart is None else [])
    return problems + check_scene_vs_chart(chart_no, v.chart_a_is, v.scene_a_fault, catalog)


def check_scene_vs_chart(chart_no: str, chart_a_is: str, scene_a_fault: int, catalog: ChartCatalog | None = None) -> list[str]:
    """정황 A 기준 과실이 해당 도표의 기본 비율(chart_a_is 로 정황 기준으로 바꾼 값)과 크게 어긋나는지.

    위원 평가(check_verdict)와 질문 단계의 선택지 검증이 같이 쓴다.
    """
    catalog = catalog or get_catalog()
    chart = catalog.get(chart_no)
    if chart is None:
        return [f"존재하지 않는 도표번호 {chart_no}"]

    # 도표의 A 가 정황의 누구인지(chart_a_is)에 따라 도표 기본 비율을 정황 A 기준으로 바꿔 모델이 낸 정황 A 과실과 비교한다
    flip = chart_a_is == "B"
    bases = sorted({(100 - var["base_ratio"]["a"]) if flip else var["base_ratio"]["a"] for var in chart["variants"]})
    nearest = min(bases, key=lambda b: abs(b - scene_a_fault))
    gap = abs(nearest - scene_a_fault)
    if gap > MAX_DEVIATION_FROM_BASE:
        return [
            f"도표 {chart_no} 의 기본 과실을 정황 A 기준으로 바꾸면 {'/'.join(map(str, bases))}"
            f"(도표 A=정황 {chart_a_is} 로 답함)인데 모델이 답한 정황 A 과실 {scene_a_fault} 가 "
            f"{gap}%p 차이 — 가감요소로 설명하기 어려운 폭 (값 또는 A/B 역할 오류 의심)"
        ]
    return []
