"""포털식 단순 매칭 baseline 과 본 파이프라인을 같은 시나리오로 비교한다 (이슈 #18, NFR-09).

    cd backend
    python scripts/run_baseline.py                                           # baseline 만 (키워드 검색)
    python scripts/run_baseline.py --modes keyword,embedding,hybrid          # 검색 방식별 baseline
    python scripts/run_baseline.py --pipeline-results eval/results/a.json,eval/results/b.json   # 저장된 파이프라인 결과와 비교

LLM 을 호출하지 않으므로 **API 비용이 없다** (임베딩 검색은 로컬 KURE-v1 사용). 파이프라인 쪽 수치는 run_scenarios.py 가 저장한 결과 파일에서 읽는다.
개발용 시나리오(개발자 작성)의 수치는 보고용이 아니다 — 직접 수집한 세트로 다시 측정한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.evaluation.baseline import run_baseline, summarize_baseline, to_dicts  # noqa: E402
from app.evaluation.scenarios import load_results, load_scenarios, summarize  # noqa: E402


def pct(v):
    return "   - " if v is None else f"{v:>4.0%} "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(ROOT / "eval" / "scenarios_dev.json"))
    ap.add_argument("--modes", default="keyword", help="쉼표로 구분한 검색 방식 (keyword, embedding, hybrid)")
    ap.add_argument("--pipeline-results", default="", help="run_scenarios.py 결과 파일(쉼표로 여러 개) — 같은 id 는 뒤의 파일이 우선")
    ap.add_argument("--save", action="store_true", help="결과를 eval/results/ 에 저장")
    ap.add_argument("-v", "--verbose", action="store_true", help="시나리오별 표 출력")
    args = ap.parse_args()

    scenarios = load_scenarios(args.set)
    merged = {}
    for p in [x.strip() for x in args.pipeline_results.split(",") if x.strip()]:
        for r in load_results(p)[1]:
            merged[r.id] = r
    if merged:
        # 파이프라인 결과가 있는 시나리오만 비교한다 (분모를 맞춘다)
        scenarios = [s for s in scenarios if s.id in merged]
    rows = []
    per_mode = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        res = run_baseline(scenarios, mode)
        per_mode[mode] = res
        rows.append((f"baseline(naive)/{mode}", summarize_baseline(res), "naive_in_range"))
        rows.append((f"baseline(optimistic)/{mode}", summarize_baseline(res), "optimistic_in_range"))

    pipeline_rows = []
    if merged:
        ids = {s.id for s in scenarios}
        pipeline_rows.append(("본 파이프라인(저장된 결과)", summarize([merged[i] for i in sorted(merged) if i in ids])))

    print(f"\n시나리오 {len(scenarios)}건 (정답 범위가 있는 건 {sum(1 for s in scenarios if s.expected_ratio_a)}건)\n")
    print(f"{'방식':<36}{'도표 1위':>9}{'도표 3위내':>10}{'도표 5위내':>10}{'정답 범위':>10}")
    print("-" * 76)
    for label, summ, key in rows:
        print(f"{label:<36}{pct(summ['chart_top1']):>9}{pct(summ['chart_top3']):>10}{pct(summ['chart_any']):>10}{pct(summ[key]):>10}")
    for label, s in pipeline_rows:
        print(f"{label:<36}{pct(s['chart_primary_hit']):>9}{'   - ':>10}{pct(s['chart_any_hit']):>10}{pct(s['in_range']):>10}")
    print("\n도표 1위/3위내/5위내 = 정답 도표가 검색 상위 N개에 있는 비율 (파이프라인은 위원 판단의 중심 도표 / 인용 도표 중 하나)")
    print("naive = 1위 도표의 첫 경우 도표 A 기본 과실을 그대로 / optimistic = 1위 도표가 맞고 역할·경우를 올바로 고른다는 상한")

    if args.verbose:
        for mode, res in per_mode.items():
            print(f"\n[{mode}] 시나리오별")
            print(f"{'id':<5}{'난이도':<5}{'1위 도표':<9}{'naive A':>8}{'naive':>7}{'optim.':>8}  상위5")
            for r in res:
                mark = lambda v: " - " if v is None else ("O" if v else "X")
                print(f"{r.id:<5}{r.difficulty:<5}{r.top1:<9}{str(r.naive_a):>8}{mark(r.naive_in_range):>7}{mark(r.optimistic_in_range):>8}  {', '.join(r.top_charts)}")

    if args.save:
        out = ROOT / "eval" / "results" / f"baseline__{Path(args.set).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({m: to_dicts(r) for m, r in per_mode.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {out.relative_to(ROOT)}")
    print("\n※ 개발용 시나리오(개발자 작성)의 수치는 보고용이 아닙니다. 직접 수집한 세트로 다시 측정하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
