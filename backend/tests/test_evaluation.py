"""시나리오 일괄 평가 모듈 테스트 — 전부 오프라인(mock·가짜 데이터)."""
import asyncio
import json

import pytest

from app.core.pipeline import FaultAssessmentPipeline
from app.core.scope_check import OutOfScopeError
from app.evaluation.scenarios import (
    Scenario,
    ScenarioResult,
    estimate_cost,
    evaluate_report,
    format_comparison,
    format_misses,
    format_results,
    load_results,
    load_scenarios,
    mentioned_charts,
    run_all,
    run_one,
    save_results,
    summarize,
)
from app.models.schemas import AgentArgument, FactCheckResult, FaultReport, ReviewScore

DEV = "eval/scenarios_dev.json"


def write(tmp_path, data):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def sc(**kw):
    base = dict(id="T1", description="제가(A) 직진 중 상대 차량(B)이 차선 변경해 충돌했습니다", expected_chart=["차43-2"], expected_ratio_a=[20, 40])
    return Scenario.model_validate({**base, **kw})


def review(perspective, a, text="", cited=()):
    return ReviewScore(reviewer_name=perspective, perspective=perspective, fault_ratio_a=a, fault_ratio_b=100 - a,
                       rationale=text, cited_rules=list(cited))


def report(a=30, reviews=None, flagged=False, dev=0.0, rounds=None, fact=None):
    return FaultReport(
        fault_ratio_a=a, fault_ratio_b=100 - a, basis="b", deviation=dev, needs_precise_review=flagged,
        rounds=rounds or [], fact_check=fact, reviews=reviews or [],
    )


# ---------- 시나리오 로드·검증 ----------
def test_dev_scenarios_load_and_cover_role_reversal_pairs():
    s = load_scenarios(DEV)
    assert len(s) == 13 and all(x.difficulty == "명백" for x in s)
    by = {x.id: x for x in s}
    # 도표 A/B 역할이 반전된 쌍: 같은 도표인데 기대 A 과실이 반대편이 된다
    assert by["D01"].expected_ratio_a == (20, 40) and by["D02"].expected_ratio_a == (60, 80)
    assert by["D03"].expected_ratio_a == (0, 10) and by["D04"].expected_ratio_a == (90, 100)
    assert by["D10"].expected_ratio_a == (90, 100) and by["D11"].expected_ratio_a == (0, 10)
    assert sum("역할 반전" in x.note for x in s) == 6
    assert all(x.expected_ratio_a[1] - x.expected_ratio_a[0] <= 20 for x in s)


def test_load_accepts_list_and_single_chart_string(tmp_path):
    p = write(tmp_path, [{"id": "X", "description": "제가(A) 직진 중 상대 차량(B)과 충돌했습니다 충분히 길게", "expected_chart": "차43-2"}])
    s = load_scenarios(p)
    assert s[0].expected_chart == ["차43-2"] and s[0].expected_ratio_a is None


def test_load_rejects_duplicate_ids_unknown_chart_and_bad_range(tmp_path):
    d = "제가(A) 직진 중 상대 차량(B)과 충돌했습니다 충분히 길게"
    with pytest.raises(ValueError, match="중복"):
        load_scenarios(write(tmp_path, [{"id": "X", "description": d}, {"id": "X", "description": d}]))
    with pytest.raises(ValueError, match="존재하지 않는 정답 도표"):
        load_scenarios(write(tmp_path, [{"id": "X", "description": d, "expected_chart": ["차99-9"]}]))
    with pytest.raises(ValueError):
        load_scenarios(write(tmp_path, [{"id": "X", "description": d, "expected_ratio_a": [60, 20]}]))
    with pytest.raises(ValueError):
        load_scenarios(write(tmp_path, [{"id": "X", "description": d, "expected_ratio_a": [0, 120]}]))


def test_scenario_converts_to_valid_accident_input():
    a = sc(signal_state="a_green_b_red", speed_a_kmh=50).to_input()
    assert a.description.startswith("제가(A)") and a.signal_state.value == "a_green_b_red" and a.speed_a_kmh == 50


# ---------- 판정 ----------
def test_mentioned_charts_sorted_by_count_then_first_seen():
    # 횟수 우선: 차43-2(3회) > 차43-4(2회) > 차41-1(1회)
    assert mentioned_charts(["차43-4 차43-2 차43-2 차43-2", "차41-1 차43-4"]) == ["차43-2", "차43-4", "차41-1"]
    # 동률이면 먼저 나온 순: 차43-4 와 차43-2 가 2회씩이면 차43-4 가 앞
    assert mentioned_charts(["차43-4 차43-2 차43-2", "차43-4"]) == ["차43-4", "차43-2"]
    assert mentioned_charts(["도표 언급 없음"]) == []


def test_evaluate_hit_and_in_range():
    rep = report(30, reviews=[review("기준표 적용", 30, "차43-2 적용", ["차43-2"]), review("현장 정황", 30, "차43-2 기준"),
                              review("유사사례", 30, "차43-2, 차43-4 참고")])
    r = evaluate_report(sc(), rep, seconds=12.34, usage={"calls": 9, "input_tokens": 1000, "output_tokens": 300})
    assert r.primary_chart == "차43-2" and r.chart_primary_hit and r.chart_any_hit and r.in_range
    assert (r.calls, r.input_tokens, r.output_tokens, r.seconds) == (9, 1000, 300, 12.3)
    assert r.reviewer_a == {"기준표 적용": 30, "현장 정황": 30, "유사사례": 30}


def test_evaluate_detects_wrong_chart_and_out_of_range_like_the_real_failure():
    # 실제 첫 실행: 위원들이 차43-4 를 고르고 A 100 을 냄
    rep = report(67, reviews=[review("기준표 적용", 100, "차43-4 적용"), review("현장 정황", 100, "차43-4"), review("유사사례", 0, "차43-4, 차43-2")],
                 flagged=True, dev=47.1)
    r = evaluate_report(sc(), rep)
    assert r.primary_chart == "차43-4" and r.chart_primary_hit is False and r.chart_any_hit is True
    assert r.in_range is False and r.needs_precise_review is True and r.deviation == 47.1


def test_evaluate_falls_back_to_round_texts_when_reviewers_name_no_chart():
    rounds = [AgentArgument(agent_name="vehicle_a_agent", round=1, content="차41-1"),
              AgentArgument(agent_name="vehicle_a_agent", round=2, content="차43-2 적용", tool_calls=["a", "b"]),
              AgentArgument(agent_name="vehicle_b_agent", round=2, content="차43-2", tool_calls=["c"])]
    r = evaluate_report(sc(), report(30, reviews=[review("기준표 적용", 30, "도표 언급 없음")], rounds=rounds))
    assert r.primary_chart == "차43-2" and r.chart_primary_hit
    assert r.tool_calls == {"vehicle_a_agent": 2, "vehicle_b_agent": 1}


def test_unset_expectations_give_none_not_false():
    r = evaluate_report(sc(expected_chart=[], expected_ratio_a=None), report(30, reviews=[review("a", 30, "차43-2")]))
    assert r.chart_primary_hit is None and r.chart_any_hit is None and r.in_range is None


def test_evaluate_records_fact_check_findings():
    fact = FactCheckResult(is_valid=False, notes="", flagged_citations=[f"A측 오류{i}" for i in range(8)])
    r = evaluate_report(sc(), report(30, reviews=[review("a", 30, "차43-2")], fact=fact))
    assert r.fact_valid is False and len(r.fact_flagged) == 5


# ---------- 실행 ----------
class FakeLLM:
    def __init__(self):
        self.usage = type("U", (), {"as_dict": lambda s: {"calls": 7, "input_tokens": 700, "output_tokens": 70}})()


class FakePipeline:
    """시나리오마다 새로 만들어져 사용량이 격리되는지, 실패가 격리되는지 확인하는 가짜 파이프라인."""
    made = 0

    def __init__(self):
        FakePipeline.made += 1
        self.llm = FakeLLM()

    async def run(self, accident):
        await asyncio.sleep(0)
        if "보행자" in accident.description:
            from app.core.scope_check import ScopeResult

            raise OutOfScopeError(ScopeResult("unsupported", "보행자", "지원 안 함"))
        return report(30, reviews=[review("기준표 적용", 30, "차43-2")])


def test_run_one_isolates_failure_into_error_result_and_keeps_usage():
    ok = asyncio.run(run_one(sc(), FakePipeline))
    bad = asyncio.run(run_one(sc(id="T2", description="보행자를 제 차가 쳤습니다 상대 차량 없음 충분히 길게"), FakePipeline))
    assert ok.ok and ok.calls == 7 and ok.input_tokens == 700
    assert not bad.ok and "OutOfScopeError" in bad.error and bad.calls == 7 and bad.expected_chart == ["차43-2"]


def test_run_all_new_pipeline_per_scenario_keeps_order_and_calls_callback():
    FakePipeline.made = 0
    scenarios = [sc(id=f"T{i}") for i in range(5)]
    seen = []
    results = asyncio.run(run_all(scenarios, FakePipeline, concurrency=3, on_result=lambda r: seen.append(r.id)))
    assert [r.id for r in results] == [f"T{i}" for i in range(5)]
    assert FakePipeline.made == 5 and sorted(seen) == sorted(r.id for r in results)


def test_real_pipeline_mock_mode_runs_all_dev_scenarios_offline():
    scenarios = load_scenarios(DEV)
    results = asyncio.run(run_all(scenarios, FaultAssessmentPipeline))
    assert len(results) == 13 and all(r.ok for r in results)
    assert all(r.calls == 0 and r.input_tokens == 0 for r in results)  # mock 모드는 실제 호출이 없다
    assert all(r.final_a == 50 for r in results)                        # mock 위원은 전부 50


# ---------- 집계·표시·저장 ----------
def res(id, diff="명백", ok=True, in_range=None, prim=None, any_=None, flag=False, secs=10.0, calls=5, i=100, o=50, fact=()):
    return ScenarioResult(id=id, difficulty=diff, ok=ok, in_range=in_range, chart_primary_hit=prim, chart_any_hit=any_,
                          needs_precise_review=flag, seconds=secs, calls=calls, input_tokens=i, output_tokens=o,
                          fact_flagged=list(fact), final_a=30, expected_ratio_a=(20, 40), expected_chart=["차43-2"],
                          primary_chart="차43-2", deviation=0.0)


def test_summarize_rates_and_totals():
    s = summarize([
        res("A", in_range=True, prim=True, any_=True),
        res("B", in_range=False, prim=False, any_=True, fact=["x"]),
        res("C", diff="애매", flag=True, in_range=None),
        res("D", diff="애매", flag=False),
        res("E", ok=False),
    ])
    assert s["n"] == 5 and s["errors"] == 1
    assert s["chart_primary_hit"] == 0.5 and s["chart_any_hit"] == 1.0 and s["in_range"] == 0.5
    assert s["clear_in_range"] == 0.5 and s["ambiguous_flag_rate"] == 0.5
    assert s["fact_flagged_rate"] == pytest.approx(0.25)
    assert s["calls"] == 25 and s["input_tokens"] == 500 and s["mean_seconds"] == 10.0


def test_summarize_handles_empty_and_no_expectations():
    s = summarize([])
    assert s["n"] == 0 and s["mean_seconds"] == 0.0 and s["in_range"] is None
    assert summarize([res("A")])["ambiguous_flag_rate"] is None


def test_estimate_cost_uses_given_prices_only():
    assert estimate_cost(2_000_000, 500_000, price_in=1.0, price_out=4.0) == pytest.approx(4.0)
    assert estimate_cost(0, 0, 1.0, 4.0) == 0


def test_comparison_table_shows_cost_column_only_with_prices():
    row = ("openai:m/keyword", summarize([res("A", in_range=True, prim=True, any_=True, i=1_000_000, o=0)]))
    assert "비용" not in format_comparison([row])
    t = format_comparison([row], price_in=2.0, price_out=8.0)
    assert "비용" in t and "2.000" in t and "openai:m/keyword" in t


def test_result_tables_and_miss_report_render():
    good, bad = res("A", in_range=True, prim=True), res("B", in_range=False, prim=False)
    err = res("C", ok=False)
    err.error = "OutOfScopeError: 지원 안 함"
    t = format_results([good, bad, err])
    assert "A " in t and "오류: OutOfScopeError" in t
    m = format_misses([good, bad, err])
    assert "- B:" in m and "- C: OutOfScopeError" in m and "- A:" not in m
    assert format_misses([good]) == "(오답 없음)"


def test_save_and_load_roundtrip_restores_tuples(tmp_path):
    p = tmp_path / "out" / "r.json"
    save_results(p, {"label": "x"}, [res("A", in_range=True, prim=True), res("B", ok=False)])
    meta, results = load_results(p)
    assert meta == {"label": "x"} and [r.id for r in results] == ["A", "B"]
    assert results[0].expected_ratio_a == (20, 40) and isinstance(results[0].expected_ratio_a, tuple)
    assert json.loads(p.read_text(encoding="utf-8"))["summary"]["n"] == 2


# ---------- 스크립트 (실제로 실행해 안전장치 확인, 오프라인) ----------
def _run_script(*args):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "scripts/run_scenarios.py", *args], cwd=root, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_script_refuses_mock_mode_without_flag_and_writes_nothing():
    from pathlib import Path

    before = set((Path(__file__).resolve().parents[1] / "eval" / "results").glob("*")) if (Path(__file__).resolve().parents[1] / "eval" / "results").exists() else set()
    p = _run_script("--limit", "1")
    assert p.returncode == 0 and "mock 모드입니다" in p.stdout and "--allow-mock" in p.stdout
    after = set((Path(__file__).resolve().parents[1] / "eval" / "results").glob("*")) if (Path(__file__).resolve().parents[1] / "eval" / "results").exists() else set()
    assert after == before


def test_script_mock_results_use_separate_label_and_file_name():
    from pathlib import Path

    p = _run_script("--allow-mock", "--limit", "1", "--ids", "D01")
    assert p.returncode == 0, p.stderr
    assert "/MOCK" in p.stdout and "_MOCK.json" in p.stdout  # 실제 실행 결과와 섞여 --resume 되지 않도록 분리
    out = Path(__file__).resolve().parents[1] / "eval" / "results" / "scenarios_dev__openai_(미설정)_keyword_MOCK.json"
    try:
        meta, results = load_results(out)
        assert meta["label"].endswith("/MOCK") and [r.id for r in results] == ["D01"]
    finally:
        out.unlink(missing_ok=True)


def test_script_rejects_unknown_scenario_id():
    p = _run_script("--allow-mock", "--ids", "NOPE")
    assert p.returncode == 2 and "없는 시나리오 id" in p.stdout


def test_script_dry_run_never_calls_anything():
    p = _run_script("--dry-run", "--limit", "2")
    assert p.returncode == 0 and "[dry-run]" in p.stdout and "D01, D02" in p.stdout


# ---------- 위원 간 편차: 이상치 한 명을 놓치지 않는가 (실제 첫 측정: 30/0/0 인데 플래그 안 켜짐) ----------
def _proceed(ratios):
    from app.agents.orchestrator import ProceedingAgent
    from app.core.llm_client import LLMClient

    reviews = [review(f"위원{i}", a, "근거") for i, a in enumerate(ratios)]
    return asyncio.run(ProceedingAgent(LLMClient()).run({"reviews": reviews, "rounds": [], "fact_check": None}))


def test_one_outlier_of_30_points_is_flagged_even_though_stdev_is_below_old_threshold():
    r = _proceed([30, 0, 0])
    assert r.deviation == 14.1 and r.deviation < 15      # 표준편차만으로는 임계값(15)에 못 미친다
    assert r.spread == 30 and r.needs_precise_review     # 최대-최소로는 잡힌다
    assert r.fault_ratio_a == 0 and r.fault_ratio_b == 100   # 최종값은 중앙값 (평균 10 이 아님)


def test_spread_threshold_boundary():
    assert _proceed([30, 25, 30]).needs_precise_review is False   # 최대-최소 5
    assert _proceed([30, 20, 11]).needs_precise_review is False   # 19
    assert _proceed([30, 20, 10]).needs_precise_review is True    # 20 (경계값 포함)


def test_stdev_rule_still_applies_and_two_reviewers_work():
    assert _proceed([0, 100]).needs_precise_review is True
    assert _proceed([40, 40]).needs_precise_review is False and _proceed([40, 40]).spread == 0


def test_results_keep_spread_and_reviewer_notes_for_root_cause_analysis():
    rep = report(10, reviews=[review("기준표 적용", 30, "차43-2 적용 " + "가" * 400), review("현장 정황", 0, "근거2"), review("유사사례", 0, "근거3")],
                 flagged=True, dev=14.1)
    rep.spread = 30
    r = evaluate_report(sc(), rep)
    assert r.spread == 30 and len(r.reviewer_notes["기준표 적용"]) == 240 and r.reviewer_notes["현장 정황"] == "근거2"
    out = format_misses([r])
    assert "최대-최소 30" in out and "· 현장 정황(0): 근거2" in out


def test_old_result_without_spread_renders_dash_not_zero():
    import re

    old = res("OLD", in_range=True, prim=True)   # 범위·도표 열이 ✓ 로 찍히게 명시
    old.spread = None
    row = format_results([old]).splitlines()[1]
    assert re.search(r"✓\s+-\s+n\s", row), row          # 위원차 열이 '-' 이고 바로 뒤가 플래그(n)
    new = res("NEW", in_range=True, prim=True)
    new.spread = 0
    assert re.search(r"✓\s+0\s+n\s", format_results([new]).splitlines()[1])  # 진짜 0 은 0 으로 표시
