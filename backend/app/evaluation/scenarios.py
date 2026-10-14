"""시나리오 일괄 평가: 시나리오 로드·검증 → 파이프라인 실행 → 정답 대비 판정 → 집계.

판정 기준
- chart_primary_hit: 검토위원들이 가장 많이 언급한 도표가 정답 도표 중 하나인가 (최종 판단이 어느 도표에 기댔는가)
- chart_any_hit:     정답 도표가 위원 텍스트 어디에든 언급됐는가
- in_range:          최종 A 과실이 정답 범위(expected_ratio_a) 안인가
- 명백 시나리오는 in_range, 애매 시나리오는 "정밀 심사 필요" 플래그 발동률이 핵심 지표다 (기획서 §평가).
가격은 코드에 넣지 않는다 (검증 불가) — 비용은 호출자가 단가를 넘길 때만 계산한다.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.chart_lookup import get_catalog
from app.models.schemas import AccidentInput, FaultReport, SignalState

CHART_RE = re.compile(r"차\d+-\d+")


# ---------------------------------------------------------------------------
# 시나리오
# ---------------------------------------------------------------------------
class Scenario(BaseModel):
    id: str
    difficulty: Literal["명백", "애매"] = "명백"
    description: str
    signal_state: SignalState = SignalState.UNKNOWN
    lane_info: Optional[str] = None
    speed_a_kmh: Optional[float] = None
    speed_b_kmh: Optional[float] = None
    expected_chart: list[str] = Field(default_factory=list)
    expected_ratio_a: Optional[tuple[int, int]] = None  # 정답 A 과실 범위 [하한, 상한]
    expected_modifiers: list[str] = Field(default_factory=list)
    # 정황에 없는 사실로 결과가 크게 갈려 사용자에게 질문해야 하는 사례면 True, 질문이 불필요한 사례면 False, 채점하지 않으면 None
    expects_question: Optional[bool] = None
    note: str = ""

    @field_validator("expected_chart", mode="before")
    @classmethod
    def _chart_str_to_list(cls, v):
        return [v] if isinstance(v, str) else (v or [])

    @model_validator(mode="after")
    def _check_range(self):
        if self.expected_ratio_a is not None:
            lo, hi = self.expected_ratio_a
            if not (0 <= lo <= hi <= 100):
                raise ValueError(f"{self.id}: expected_ratio_a 는 0 <= 하한 <= 상한 <= 100 이어야 함: {self.expected_ratio_a}")
        return self

    def to_input(self) -> AccidentInput:
        return AccidentInput(
            description=self.description,
            signal_state=self.signal_state,
            lane_info=self.lane_info,
            speed_a_kmh=self.speed_a_kmh,
            speed_b_kmh=self.speed_b_kmh,
        )


def load_scenarios(path: str | Path) -> list[Scenario]:
    """시나리오 파일을 읽고 검증한다. 형식: {"scenarios": [...]} 또는 [...] — id 중복·존재하지 않는 정답 도표는 오류."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["scenarios"] if isinstance(data, dict) else data
    scenarios = [Scenario.model_validate(r) for r in raw]

    dup = [i for i, n in Counter(s.id for s in scenarios).items() if n > 1]
    if dup:
        raise ValueError(f"시나리오 id 중복: {dup}")
    catalog = get_catalog()
    bad = {s.id: [c for c in s.expected_chart if catalog.get(c) is None] for s in scenarios}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ValueError(f"존재하지 않는 정답 도표(오타?): {bad}")
    return scenarios


# ---------------------------------------------------------------------------
# 결과 판정
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    id: str
    difficulty: str
    ok: bool = True
    error: str = ""
    final_a: Optional[int] = None
    expected_ratio_a: Optional[tuple[int, int]] = None
    in_range: Optional[bool] = None
    expected_chart: list[str] = field(default_factory=list)
    primary_chart: str = ""
    mentioned_charts: list[str] = field(default_factory=list)
    chart_primary_hit: Optional[bool] = None
    chart_any_hit: Optional[bool] = None
    reviewer_a: dict[str, int] = field(default_factory=dict)
    reviewer_notes: dict[str, str] = field(default_factory=dict)  # 위원별 판단 근거 앞부분 (오답 원인 분석용, 재실행 없이 확인)
    # 위원별 모델 원답: "차41-1 | 도표 A=정황 B | 도표 A 과실 100" (역할 오류·이중 변환 분석용)
    reviewer_raw: dict[str, str] = field(default_factory=dict)
    reviewer_confidence: dict[str, str] = field(default_factory=dict)  # 위원별 확신(high/medium/low) — 플래그 기준 조정용
    flag_reasons: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    expects_question: Optional[bool] = None
    clarification_hit: Optional[bool] = None   # expects_question 이 있을 때 needs_clarification 과 일치하는지
    clarification: str = ""        # 만든 질문과 보기별 (도표, 정황 A 과실). 질문이 없으면 참고 문구
    deviation: Optional[float] = None
    spread: Optional[int] = None
    needs_precise_review: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)
    fact_valid: Optional[bool] = None
    fact_flagged: list[str] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def mentioned_charts(texts: list[str]) -> list[str]:
    """텍스트들에서 언급된 도표번호를 (언급 횟수 내림차순, 먼저 나온 순)으로 정렬해 돌려준다."""
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    pos = 0
    for t in texts:
        for m in CHART_RE.finditer(t):
            counts[m.group()] += 1
            first.setdefault(m.group(), pos + m.start())
        pos += len(t) + 1
    return sorted(counts, key=lambda c: (-counts[c], first[c]))


def _reviewer_texts(report: FaultReport) -> list[str]:
    texts: list[str] = []
    for r in report.reviews:
        texts.append(r.rationale)
        texts += [f"{c.item} {c.finding}" for c in r.checklist]
        texts.append(" ".join(r.cited_rules))
    if not any(CHART_RE.search(t) for t in texts):  # 위원이 도표를 안 적었으면 2라운드 논증으로 대체
        texts = [a.content for a in report.rounds if a.round == max((x.round for x in report.rounds), default=1)]
    return texts


def format_clarification(report) -> str:
    """평가 결과에 남길 질문 요약 (내부용: 보기별 도표·과실 포함)."""
    if not report.clarifications:
        return report.clarification_note
    return " || ".join(
        f"{q.question} [" + " / ".join(f"{o.label} → {o.chart_no}, A {o.scene_a_fault}" for o in q.options) + "]"
        for q in report.clarifications
    )


def raw_answer(review) -> str:
    """위원이 모델에게서 받은 도표·역할·도표 A 과실 원답 (코드가 정황 A 기준으로 환산하기 전 값)."""
    role = {"A": "정황 A", "B": "정황 B"}.get(review.chart_a_is, "도표 미사용")
    return f"{review.chart_no or '-'} | 도표 A={role} | 정황 A 과실 {review.scene_a_fault}"


def evaluate_report(scenario: Scenario, report: FaultReport, *, seconds: float = 0.0, usage: dict | None = None) -> ScenarioResult:
    usage = usage or {}
    charts = mentioned_charts(_reviewer_texts(report))
    primary = charts[0] if charts else ""
    exp = set(scenario.expected_chart)
    tool_calls: Counter[str] = Counter()
    for a in report.rounds:
        tool_calls[a.agent_name] += len(a.tool_calls)

    in_range = None
    if scenario.expected_ratio_a is not None:
        lo, hi = scenario.expected_ratio_a
        in_range = lo <= report.fault_ratio_a <= hi

    return ScenarioResult(
        id=scenario.id,
        difficulty=scenario.difficulty,
        final_a=report.fault_ratio_a,
        expected_ratio_a=scenario.expected_ratio_a,
        in_range=in_range,
        expected_chart=list(scenario.expected_chart),
        primary_chart=primary,
        mentioned_charts=charts[:6],
        chart_primary_hit=(primary in exp) if exp else None,
        chart_any_hit=bool(exp & set(charts)) if exp else None,
        reviewer_a={r.perspective: r.fault_ratio_a for r in report.reviews},
        reviewer_notes={r.perspective: r.rationale[:240] for r in report.reviews},
        reviewer_raw={r.perspective: raw_answer(r) for r in report.reviews},
        reviewer_confidence={r.perspective: r.confidence for r in report.reviews},
        flag_reasons=list(report.flag_reasons),
        needs_clarification=report.needs_clarification,
        expects_question=scenario.expects_question,
        clarification_hit=None if scenario.expects_question is None else report.needs_clarification == scenario.expects_question,
        clarification=format_clarification(report),
        deviation=report.deviation,
        spread=report.spread,
        needs_precise_review=report.needs_precise_review,
        warnings=list(report.warnings),
        fact_valid=report.fact_check.is_valid if report.fact_check else None,
        fact_flagged=list(report.fact_check.flagged_citations)[:5] if report.fact_check else [],
        tool_calls=dict(tool_calls),
        seconds=round(seconds, 1),
        calls=usage.get("calls", 0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
PipelineFactory = Callable[[], Any]  # 시나리오마다 새 파이프라인(=새 LLMClient)을 만들어 사용량을 격리한다


async def run_one(scenario: Scenario, factory: PipelineFactory) -> ScenarioResult:
    pipeline = factory()
    t0 = time.time()
    try:
        report = await pipeline.run(scenario.to_input())
    except Exception as e:  # 한 시나리오의 실패가 전체 실행을 멈추지 않게 한다 (범위 밖 거절·구조화 실패 등)
        return ScenarioResult(
            id=scenario.id, difficulty=scenario.difficulty, ok=False,
            error=f"{type(e).__name__}: {str(e)[:300]}", seconds=round(time.time() - t0, 1),
            expected_ratio_a=scenario.expected_ratio_a, expected_chart=list(scenario.expected_chart),
            **_usage_fields(pipeline),
        )
    return evaluate_report(scenario, report, seconds=time.time() - t0, usage=_usage_dict(pipeline))


def _usage_dict(pipeline) -> dict:
    llm = getattr(pipeline, "llm", None)
    return llm.usage.as_dict() if llm is not None and hasattr(llm, "usage") else {}


def _usage_fields(pipeline) -> dict:
    u = _usage_dict(pipeline)
    return {"calls": u.get("calls", 0), "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}


async def run_all(
    scenarios: list[Scenario],
    factory: PipelineFactory,
    *,
    concurrency: int = 1,
    on_result: Callable[[ScenarioResult], Awaitable[None] | None] | None = None,
) -> list[ScenarioResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[ScenarioResult] = []

    async def worker(s: Scenario):
        async with sem:
            r = await run_one(s, factory)
        results.append(r)
        if on_result:
            maybe = on_result(r)
            if asyncio.iscoroutine(maybe):
                await maybe

    await asyncio.gather(*(worker(s) for s in scenarios))
    order = {s.id: i for i, s in enumerate(scenarios)}
    return sorted(results, key=lambda r: order[r.id])


# ---------------------------------------------------------------------------
# 집계·표시
# ---------------------------------------------------------------------------
def _rate(flags: list[Optional[bool]]) -> Optional[float]:
    vals = [f for f in flags if f is not None]
    return (sum(vals) / len(vals)) if vals else None


def summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    clear = [r for r in ok if r.difficulty == "명백"]
    ambiguous = [r for r in ok if r.difficulty == "애매"]
    return {
        "n": len(results),
        "errors": len(results) - len(ok),
        "chart_primary_hit": _rate([r.chart_primary_hit for r in ok]),
        "chart_any_hit": _rate([r.chart_any_hit for r in ok]),
        "in_range": _rate([r.in_range for r in ok]),
        "clear_in_range": _rate([r.in_range for r in clear]),
        "ambiguous_flag_rate": _rate([r.needs_precise_review for r in ambiguous]) if ambiguous else None,
        "clear_flag_rate": _rate([r.needs_precise_review for r in clear]) if clear else None,
        "fact_flagged_rate": _rate([bool(r.fact_flagged) for r in ok]),
        "clarify_hit": _rate([r.clarification_hit for r in ok]),
        "mean_seconds": round(sum(r.seconds for r in results) / len(results), 1) if results else 0.0,
        "calls": sum(r.calls for r in results),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }


def estimate_cost(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    """단가는 100만 토큰당 금액. 코드에 가격을 고정하지 않고 호출자가 넘긴 값으로만 계산한다."""
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def _pct(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v:>4.0%} "


def format_comparison(rows: list[tuple[str, dict[str, Any]]], price_in: float | None = None, price_out: float | None = None) -> str:
    head = f"{'모델/설정':<34}{'n':>3}{'오류':>4} {'도표(주)':>7}{'도표(any)':>8}{'범위':>6}{'애매플래그':>9}{'평균s':>7}{'호출':>6}{'토큰(in/out)':>16}"
    if price_in is not None and price_out is not None:
        head += f"{'비용':>9}"
    lines = [head, "-" * len(head)]
    for label, s in rows:
        line = (
            f"{label:<34}{s['n']:>3}{s['errors']:>4} {_pct(s['chart_primary_hit']):>7}{_pct(s['chart_any_hit']):>8}"
            f"{_pct(s['in_range']):>6}{_pct(s['ambiguous_flag_rate']):>9}{s['mean_seconds']:>7}{s['calls']:>6}"
            f"{s['input_tokens']:>9,}/{s['output_tokens']:<6,}"
        )
        if price_in is not None and price_out is not None:
            line += f"{estimate_cost(s['input_tokens'], s['output_tokens'], price_in, price_out):>9.3f}"
        lines.append(line)
    return "\n".join(lines)


def format_results(results: list[ScenarioResult]) -> str:
    lines = [f"{'id':<8}{'난이도':<5}{'도표':<5}{'A과실':>6}{'정답범위':>10}{'범위':>5}{'위원차':>6}{'플래그':>6}{'초':>6}{'호출':>5}  주 도표"]
    for r in results:
        if not r.ok:
            lines.append(f"{r.id:<8}{r.difficulty:<5}오류: {r.error[:90]}")
            continue
        chart = "  - " if r.chart_primary_hit is None else ("  ✓ " if r.chart_primary_hit else "  ✗ ")
        rng = "-" if r.expected_ratio_a is None else f"{r.expected_ratio_a[0]}~{r.expected_ratio_a[1]}"
        inr = "  - " if r.in_range is None else ("  ✓ " if r.in_range else "  ✗ ")
        flag = "Y" if r.needs_precise_review else "n"
        lines.append(
            f"{r.id:<8}{r.difficulty:<5}{chart}{r.final_a:>6}{rng:>10}{inr}{('-' if r.spread is None else r.spread):>6}{flag:>6}"
            f"{r.seconds:>6.0f}{r.calls:>5}  {r.primary_chart or '-'} (정답 {','.join(r.expected_chart) or '-'})"
        )
    return "\n".join(lines)


def format_misses(results: list[ScenarioResult]) -> str:
    out = []
    for r in results:
        if not r.ok or r.chart_primary_hit is False or r.in_range is False:
            why = r.error if not r.ok else ", ".join(
                x for x in (
                    f"도표 {r.primary_chart or '-'} (정답 {','.join(r.expected_chart)})" if r.chart_primary_hit is False else "",
                    f"A {r.final_a} (정답 {r.expected_ratio_a[0]}~{r.expected_ratio_a[1]})" if r.in_range is False else "",
                ) if x
            )
            notes = "".join(
                f"\n      · {p}({r.reviewer_a.get(p)}): {n}  [원답 {r.reviewer_raw.get(p, '-')}]" for p, n in r.reviewer_notes.items()
            )
            out.append(
                f"- {r.id}: {why}\n    위원 A과실 {r.reviewer_a} (표준편차 {r.deviation}, 최대-최소 {r.spread}) "
                f"/ 사실검증 지적 {r.fact_flagged[:2]} / 경고 {r.warnings}{notes}"
            )
    return "\n".join(out) or "(오답 없음)"


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
def save_results(path: str | Path, meta: dict[str, Any], results: list[ScenarioResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"meta": meta, "summary": summarize(results), "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def load_results(path: str | Path) -> tuple[dict[str, Any], list[ScenarioResult]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for r in data["results"]:
        if r.get("expected_ratio_a") is not None:
            r["expected_ratio_a"] = tuple(r["expected_ratio_a"])
        results.append(ScenarioResult(**r))
    return data["meta"], results
