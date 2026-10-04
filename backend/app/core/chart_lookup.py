"""과실비율 기준표(chart_catalog) 조회.

사고 입력 텍스트 → 후보 도표(chartNo) 매칭 → 에이전트 프롬프트에 넣을 도표 요약 생성.
검색 모드(설정 CHART_SEARCH_MODE):
- keyword   (기본) 한국어 문자 2-gram 겹침 + IDF. 의존성 없음
- embedding 로컬 KURE-v1 임베딩 코사인 (core/embedding_index.py)
- hybrid    keyword 와 embedding 의 순위를 RRF 로 결합
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.embedding_index import EmbeddingUnavailable
from app.models.schemas import AccidentInput, AccidentType

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fault_ratio_reference.json"
RRF_K = 60  # Reciprocal Rank Fusion 상수 (표준값)

# 입력의 AccidentType 은 대분류 힌트로만 쓴다. 매칭 점수에 가볍게 더해지는 키워드.
_TYPE_HINTS: dict[AccidentType, str] = {
    AccidentType.STRAIGHT_VS_LEFT_TURN: "직진 좌회전 교차로 신호",
    AccidentType.LANE_CHANGE: "진로변경 차로변경 후행 직진 선행",
    AccidentType.REAR_END_COLLISION: "후방 추돌 후행 선행 정지",
    AccidentType.SIGNAL_VIOLATION: "신호 신호위반 교차로 적색",
    AccidentType.OTHER: "",
}


def _bigrams(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[0-9A-Za-z가-힣]+", text):
        if len(token) == 1:
            out.add(token)
        out.update(token[i : i + 2] for i in range(len(token) - 1))
    return out


@dataclass(frozen=True)
class ChartMatch:
    chart_no: str
    score: float
    chart: dict[str, Any]


class ChartCatalog:
    def __init__(self, charts: dict[str, dict[str, Any]]) -> None:
        self.charts = charts
        self._docs = {no: self._index_text(c) for no, c in charts.items()}
        self._grams = {no: _bigrams(t) for no, t in self._docs.items()}
        n = len(charts)
        df: dict[str, int] = {}
        for grams in self._grams.values():
            for g in grams:
                df[g] = df.get(g, 0) + 1
        self._idf = {g: math.log((n + 1) / (c + 0.5)) for g, c in df.items()}

    @staticmethod
    def _index_text(c: dict[str, Any]) -> str:
        parts = [c["label"], c["category"], c["situation"]]
        for v in c["variants"]:
            parts += [v["vehicle_a"], v["vehicle_b"]]
        return " ".join(parts)

    def get(self, chart_no: str) -> dict[str, Any] | None:
        return self.charts.get(chart_no)

    @property
    def index_texts(self) -> dict[str, str]:
        """도표별 색인 텍스트 (키워드·임베딩 검색 공통)."""
        return self._docs

    def attach_embedding(self, index) -> None:
        """임베딩 색인을 주입한다 (테스트·사전 로드용). 없으면 처음 필요할 때 KURE-v1 으로 만든다."""
        self._embedding = index

    def _embedding_index(self):
        if getattr(self, "_embedding", None) is None:
            from app.core.embedding_index import EmbeddingIndex

            self._embedding = EmbeddingIndex(self._docs)
        return self._embedding

    def search(self, query: str, limit: int = 3, mode: str = "keyword") -> list[ChartMatch]:
        """mode: keyword(2-gram+IDF) | embedding(KURE-v1 코사인) | hybrid(두 순위의 RRF 결합).

        임베딩을 쓸 수 없으면(의존성 없음 등) 경고 후 keyword 로 폴백한다.
        """
        if mode == "keyword":
            return self._search_keyword(query, limit)
        if mode not in ("embedding", "hybrid"):
            raise ValueError(f"알 수 없는 검색 모드: {mode}")
        if not query.strip():
            return []
        try:
            emb = self._embedding_index()
        except EmbeddingUnavailable as e:
            logger.warning("임베딩 검색 불가, 키워드 검색으로 대체합니다: %s", e)
            return self._search_keyword(query, limit)

        if mode == "embedding":
            return [ChartMatch(no, s, self.charts[no]) for no, s in emb.search(query, limit)]

        # hybrid: 점수 스케일이 달라 점수 대신 순위로 결합한다 (Reciprocal Rank Fusion)
        n = len(self.charts)
        fused: dict[str, float] = {}
        for ranked in ([m.chart_no for m in self._search_keyword(query, n)], [no for no, _ in emb.search(query, n)]):
            for rank, no in enumerate(ranked):
                fused[no] = fused.get(no, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [ChartMatch(no, s, self.charts[no]) for no, s in top]

    def _search_keyword(self, query: str, limit: int) -> list[ChartMatch]:
        q = _bigrams(query)
        if not q:
            return []
        scored = []
        for no, grams in self._grams.items():
            hit = q & grams
            if not hit:
                continue
            # 질의 대비 문서 길이로 정규화해 긴 도표가 유리해지는 것을 줄인다.
            score = sum(self._idf.get(g, 0.0) for g in hit) / math.sqrt(len(grams))
            scored.append(ChartMatch(no, score, self.charts[no]))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:limit]


@lru_cache
def get_catalog() -> ChartCatalog:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    charts = {k: v for k, v in data["chart_catalog"].items() if not k.startswith("_")}
    return ChartCatalog(charts)


def find_candidate_charts(accident: AccidentInput, limit: int = 3) -> list[ChartMatch]:
    hint = _TYPE_HINTS.get(accident.accident_type, "") if accident.accident_type else ""
    parts = [accident.description, accident.lane_info or "", hint]
    return get_catalog().search(" ".join(parts), limit, mode=get_settings().chart_search_mode)


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def format_chart(no: str, chart: dict[str, Any], *, with_notes: bool = False) -> str:
    """에이전트 프롬프트에 넣을 도표 요약. 가감요소는 해당 차량 과실 가산(+)/감산(-) 포인트."""
    lines = [f"■ {no} {chart['label']}  ({chart['category']})", f"사고 상황: {chart['situation']}"]
    if chart["applicability"]:
        lines.append(f"적용(비적용): {chart['applicability']}")
    lines.append(f"기본 과실 해설: {chart['base_rationale']}")
    for v in chart["variants"]:
        r = v["base_ratio"]
        head = f"기본 과실 {v['label']} ".strip()
        lines.append(f"{head} A {r['a']} : B {r['b']}  (A={v['vehicle_a']} / B={v['vehicle_b']})")
        if v["modifiers"]:
            # 같은 [택일N] 표시는 포털에서 하나만 선택할 수 있는 항목들이다 (예: 현저한 과실 vs 중대한 과실)
            mods = ", ".join(
                f"{'[택일%d] ' % m['group'] if m.get('group') else ''}{m['side']} {m['condition']} {_signed(m['adjustment'])}"
                for m in v["modifiers"]
            )
            lines.append(f"  가감요소: {mods}")
    lines.append(
        "가감요소 적용 방식: A 과실 = 기본 A + (A측 가감점 합) − (B측 가감점 합), 0~100 범위. "
        "같은 [택일N] 표시끼리는 하나만 적용"
    )
    if chart["laws"]:
        lines.append("관련 법규: " + " / ".join(l["title"] for l in chart["laws"]))
    if with_notes:
        for n in chart["modifier_notes"]:
            lines.append("수정요소 해설: " + n["text"])
    return "\n".join(lines)


def format_candidates(matches: list[ChartMatch]) -> str:
    if not matches:
        return "(매칭되는 기준표 없음 — 일반 도로교통법 원칙에 따라 판단)"
    return "\n\n".join(format_chart(m.chart_no, m.chart) for m in matches)
