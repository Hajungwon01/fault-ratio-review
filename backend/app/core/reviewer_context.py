"""검토위원에게 줄 근거 자료를 만든다.

- 인용 도표 상세: A/B 논증이 실제로 인용한 (존재하는) 도표의 기본 과실·가감요소·수정요소 해설
- 판례·조정사례와 **근거 등급** (위원 C용)
    direct  인용 도표에 판례·조정사례가 있음
    similar 인용 도표에는 없지만 같은 분류의 유사 도표에 있음 (원문 31개 도표가 판례 없음)
    none    직접·유사 모두 없음
  등급을 명시해 위원이 근거 없는 판례를 지어내거나, 유사 도표 사례를 직접 근거처럼 쓰지 않게 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.chart_lookup import ChartCatalog, format_chart, get_catalog

_CHART_RE = re.compile(r"차\d+-\d+")
MAX_CITED_CHARTS = 4
MAX_PRECEDENTS = 3
_NOTE_LIMIT = 1200  # 도표당 수정요소 해설 글자 수 상한 (프롬프트 비대화 방지)
_PRECEDENT_TEXT_LIMIT = 500


@dataclass(frozen=True)
class PrecedentEvidence:
    chart_no: str
    level: str  # "direct" | "similar" | "none"
    items: list[tuple[str, str, str]]  # (출처 도표번호, 제목, 본문)


def cited_chart_numbers(rounds, catalog: ChartCatalog | None = None) -> list[str]:
    """논증의 인용 목록과 본문에서 실제로 존재하는 도표번호를 인용 순서대로 중복 없이 뽑는다."""
    catalog = catalog or get_catalog()
    found: list[str] = []
    for r in rounds or []:
        for no in [*[c for c in r.cited_rules if _CHART_RE.fullmatch(c)], *_CHART_RE.findall(r.content)]:
            if catalog.get(no) is not None and no not in found:
                found.append(no)
    return found


def precedent_evidence(chart_no: str, catalog: ChartCatalog | None = None) -> PrecedentEvidence:
    catalog = catalog or get_catalog()
    chart = catalog.get(chart_no)
    if chart is None:
        return PrecedentEvidence(chart_no, "none", [])
    if chart["precedents"]:
        items = [(chart_no, p["title"], p["text"]) for p in chart["precedents"][:MAX_PRECEDENTS]]
        return PrecedentEvidence(chart_no, "direct", items)

    # 같은 분류(정확히 같은 경로 → 한 단계 상위 경로) 도표 중 판례가 있는 것
    parts = chart["category"].split(" > ")
    for depth in (len(parts), len(parts) - 1):
        if depth < 2:  # 최상위 대분류만 같은 것은 너무 느슨해 유사로 인정하지 않는다
            break
        prefix = " > ".join(parts[:depth])
        siblings = [
            (no, c) for no, c in catalog.charts.items()
            if no != chart_no and c["precedents"] and (c["category"] + " ").startswith(prefix)
        ]
        if siblings:
            items = [(no, p["title"], p["text"]) for no, c in siblings for p in c["precedents"]][:MAX_PRECEDENTS]
            return PrecedentEvidence(chart_no, "similar", items)
    return PrecedentEvidence(chart_no, "none", [])


_LEVEL_TEXT = {
    "direct": "직접 근거 (인용 도표 자체의 판례·조정사례)",
    "similar": "유사 도표의 사례 (인용 도표에는 판례 없음 — 직접 근거로 취급하지 말 것)",
    "none": "근거 없음 (제공된 판례·조정사례 없음 — 사례를 지어내지 말고 confidence 를 low 로 둘 것)",
}


def format_cited_charts(chart_nos: list[str], catalog: ChartCatalog | None = None) -> str:
    catalog = catalog or get_catalog()
    if not chart_nos:
        return "(A/B 논증이 인용한 유효한 도표 없음)"
    blocks = []
    for no in chart_nos[:MAX_CITED_CHARTS]:
        text = format_chart(no, catalog.charts[no])
        notes = " ".join(n["text"] for n in catalog.charts[no]["modifier_notes"])
        if notes:
            text += "\n수정요소 해설: " + (notes[:_NOTE_LIMIT] + ("…" if len(notes) > _NOTE_LIMIT else ""))
        blocks.append(text)
    return "\n\n".join(blocks)


def format_precedents(chart_nos: list[str], catalog: ChartCatalog | None = None) -> str:
    catalog = catalog or get_catalog()
    if not chart_nos:
        return "(인용 도표가 없어 판례 근거를 조회하지 못함 — 근거 없음으로 취급)"
    blocks = []
    for no in chart_nos[:MAX_CITED_CHARTS]:
        ev = precedent_evidence(no, catalog)
        lines = [f"■ {no} 근거 등급: {_LEVEL_TEXT[ev.level]}"]
        for src, title, text in ev.items:
            tag = "" if src == no else f"[유사: {src}] "
            lines.append(f"- {tag}{title}: {text[:_PRECEDENT_TEXT_LIMIT]}{'…' if len(text) > _PRECEDENT_TEXT_LIMIT else ''}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
