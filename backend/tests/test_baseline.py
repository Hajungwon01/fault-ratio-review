"""포털식 단순 매칭 baseline (#18, NFR-09) — LLM 없이 검색 1위 도표의 기본 과실비율만 쓴다."""
from app.core.chart_lookup import ChartMatch, get_catalog
from app.evaluation.baseline import _candidate_values, predict, run_baseline, summarize_baseline
from app.evaluation.scenarios import Scenario, load_scenarios


def chart(*bases_a):
    return {"variants": [{"base_ratio": {"a": a, "b": 100 - a}} for a in bases_a]}


class FakeCatalog:
    def __init__(self, ranking):
        self.ranking = ranking                     # [(chart_no, chart dict)]

    def search(self, query, limit=5, mode="keyword"):
        return [ChartMatch(no, 1.0, c) for no, c in self.ranking[:limit]]


def scenario(**kw):
    base = dict(id="T", description="제가(A) 직진 중 상대 차량(B)이 차선을 변경해 충돌했어요", expected_chart=["차43-2"], expected_ratio_a=[60, 80])
    return Scenario.model_validate({**base, **kw})


def test_candidate_values_contain_every_variant_and_its_reverse():
    assert _candidate_values(chart(30, 80)) == {30, 70, 80, 20}


def test_naive_uses_first_variant_as_is_and_optimistic_allows_role_and_variant_choice():
    # 정답: 사용자가 도표 B(기본 30 의 반대 = 70). naive 는 도표 A 값 30 을 그대로 써서 틀리고, 역할을 맞게 대응하면(70) 적중
    cat = FakeCatalog([("차43-2", chart(30)), ("차1-1", chart(0))])
    r = predict(scenario(), cat)
    assert (r.top1, r.naive_a, r.naive_in_range, r.optimistic_in_range) == ("차43-2", 30, False, True)
    assert (r.chart_top1_hit, r.chart_top3_hit, r.chart_any_hit) == (True, True, True)


def test_naive_hits_when_the_user_happens_to_be_chart_a():
    r = predict(scenario(expected_ratio_a=[20, 40]), FakeCatalog([("차43-2", chart(30))]))
    assert r.naive_in_range is True and r.optimistic_in_range is True


def test_wrong_top_chart_and_chart_rank_metrics():
    cat = FakeCatalog([("차1-1", chart(0)), ("차2-2", chart(0)), ("차43-2", chart(30))])
    r = predict(scenario(), cat)
    assert (r.chart_top1_hit, r.chart_top3_hit, r.chart_any_hit) == (False, True, True)
    far = predict(scenario(expected_chart=["차99-9"]), cat)
    assert (far.chart_top1_hit, far.chart_top3_hit, far.chart_any_hit) == (False, False, False)


def test_scenarios_without_a_range_are_not_scored_and_empty_search_counts_as_a_miss():
    r = predict(scenario(expected_ratio_a=None), FakeCatalog([("차43-2", chart(30))]))
    assert r.naive_in_range is None and r.optimistic_in_range is None and r.chart_top1_hit is True
    none = predict(scenario(), FakeCatalog([]))
    assert none.naive_in_range is False and none.optimistic_in_range is False and none.top1 == ""


def test_dev_scenarios_keyword_baseline_invariants():
    scenarios = load_scenarios("eval/scenarios_dev.json")
    res = run_baseline(scenarios, "keyword")
    s = summarize_baseline(res)
    assert s["n"] == len(scenarios) and s["n_scored"] == sum(1 for x in scenarios if x.expected_ratio_a)
    assert all(not r.naive_in_range or r.optimistic_in_range for r in res)            # naive 가 맞으면 optimistic 도 맞는다
    assert s["optimistic_in_range"] >= s["naive_in_range"]
    assert s["chart_top1"] <= s["chart_top3"] <= s["chart_any"]
    assert all(len(r.top_charts) <= 5 and set(r.top_charts) <= set(get_catalog().charts) for r in res)
