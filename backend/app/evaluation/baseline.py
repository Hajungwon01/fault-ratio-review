"""포털식 단순 매칭 baseline — 본 파이프라인이 "도표 찾기 + 기본 과실비율"보다 나은지 같은 시나리오로 비교한다 (NFR-09, #18).

손해보험협회 과실비율정보포털은 정황을 입력하면 도표를 찾아 주고 도표의 기본 과실비율과 가감요소로 비율을 계산하게 해 준다.
여기서는 LLM 없이 **검색 1위 도표의 기본 과실비율을 그대로 답으로 쓰는** 가장 단순한 형태를 baseline 으로 둔다. 두 가지 변형을 같이 낸다.

- naive:      도표의 A 기본 과실을 곧바로 "나의 과실"로 쓴다 (도표의 A/B 와 정황의 A/B 대응을 하지 않는다).
              첫 번째 경우((가))만 본다. 사용자가 도표 설명을 읽지 않고 숫자만 가져가는 경우에 해당한다.
- optimistic: 1위 도표가 맞고, 사용자가 역할을 올바로 대응하며 그 도표의 어느 경우((가)(나)(다))든 고를 수 있다고 가정한 상한이다.
              (도표 A 기본 과실, 또는 100 에서 뺀 값, 어느 경우든 정답 범위 안에 하나라도 있으면 적중)

가감요소 적용, 정황에 없는 사실의 처리, 근거 설명은 baseline 이 하지 않는 부분이다 — 정답 범위(시나리오 허용 폭 ±10)가 기본 비율만으로
충분히 맞는 쉬운 사례는 baseline 도 맞힐 수 있고, 차이는 가감요소가 필요한 사례와 역할 반전 사례에서 난다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from app.core.chart_lookup import ChartCatalog, get_catalog
from app.evaluation.scenarios import Scenario


@dataclass
class BaselineResult:
    id: str
    difficulty: str
    mode: str
    top_charts: list[str] = field(default_factory=list)
    top1: str = ""
    naive_a: Optional[int] = None
    naive_in_range: Optional[bool] = None
    optimistic_in_range: Optional[bool] = None
    chart_top1_hit: Optional[bool] = None
    chart_top3_hit: Optional[bool] = None
    chart_any_hit: Optional[bool] = None   # 상위 5개 안에 정답 도표가 있는가


def _candidate_values(chart: dict) -> set[int]:
    """도표의 모든 경우에 대해 도표 A 기본 과실과 그 반대(100 - A) — 역할 대응을 올바로 한 사람이 얻는 값의 집합."""
    values: set[int] = set()
    for v in chart["variants"]:
        a = v["base_ratio"]["a"]
        values |= {a, 100 - a}
    return values


def predict(scenario: Scenario, catalog: ChartCatalog | None = None, mode: str = "keyword", limit: int = 5) -> BaselineResult:
    catalog = catalog or get_catalog()
    matches = catalog.search(scenario.to_input().to_prompt_text(), limit, mode=mode)
    nos = [m.chart_no for m in matches]
    r = BaselineResult(id=scenario.id, difficulty=scenario.difficulty, mode=mode, top_charts=nos, top1=nos[0] if nos else "")

    exp = set(scenario.expected_chart)
    if exp:
        r.chart_top1_hit = bool(nos[:1] and nos[0] in exp)
        r.chart_top3_hit = bool(exp & set(nos[:3]))
        r.chart_any_hit = bool(exp & set(nos))

    if matches and scenario.expected_ratio_a is not None:
        lo, hi = scenario.expected_ratio_a
        first = matches[0].chart["variants"][0]["base_ratio"]["a"]
        r.naive_a = first
        r.naive_in_range = lo <= first <= hi
        r.optimistic_in_range = any(lo <= v <= hi for v in _candidate_values(matches[0].chart))
    elif scenario.expected_ratio_a is not None:
        r.naive_in_range = False
        r.optimistic_in_range = False
    return r


def run_baseline(scenarios: list[Scenario], mode: str = "keyword") -> list[BaselineResult]:
    catalog = get_catalog()
    return [predict(s, catalog, mode) for s in scenarios]


def _rate(flags: list[Optional[bool]]) -> Optional[float]:
    vals = [f for f in flags if f is not None]
    return (sum(vals) / len(vals)) if vals else None


def summarize_baseline(results: list[BaselineResult]) -> dict[str, Optional[float]]:
    return {
        "n": len(results),
        "n_scored": sum(1 for r in results if r.naive_in_range is not None),
        "chart_top1": _rate([r.chart_top1_hit for r in results]),
        "chart_top3": _rate([r.chart_top3_hit for r in results]),
        "chart_any": _rate([r.chart_any_hit for r in results]),
        "naive_in_range": _rate([r.naive_in_range for r in results]),
        "optimistic_in_range": _rate([r.optimistic_in_range for r in results]),
    }


def to_dicts(results: list[BaselineResult]) -> list[dict]:
    return [asdict(r) for r in results]
