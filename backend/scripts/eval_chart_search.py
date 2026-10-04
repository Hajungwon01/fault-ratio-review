"""도표 검색 방식 비교 평가: keyword vs embedding(KURE-v1) vs hybrid.

실행:  python scripts/eval_chart_search.py [--set eval/chart_search_eval.json] [--modes keyword,embedding,hybrid] [-v]
지표:  Top-1 / Top-3 적중률(정답 도표가 상위 N위 안에 있는 질의 비율), MRR(첫 정답 순위의 역수 평균)

주의: embedding/hybrid 는 임베딩을 실제로 쓸 수 있을 때만 평가한다 (조용히 keyword 로 폴백되면 비교가 왜곡되므로 중단).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.chart_lookup import get_catalog  # noqa: E402


def rank_of(expected: list[str], ranked: list[str]) -> int | None:
    """정답 중 가장 앞선 순위(1부터). 없으면 None."""
    ranks = [ranked.index(e) + 1 for e in expected if e in ranked]
    return min(ranks) if ranks else None


def evaluate(catalog, items: list[dict], mode: str, depth: int = 10) -> dict:
    rows = []
    for it in items:
        ranked = [m.chart_no for m in catalog.search(it["query"], depth, mode=mode)]
        rows.append({"id": it["id"], "query": it["query"], "expected": it["expected"], "ranked": ranked,
                     "rank": rank_of(it["expected"], ranked)})
    n = len(rows)
    top = lambda k: sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n  # noqa: E731
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return {"mode": mode, "n": n, "top1": top(1), "top3": top(3), "mrr": mrr, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(ROOT / "eval" / "chart_search_eval.json"))
    ap.add_argument("--modes", default="keyword,embedding,hybrid")
    ap.add_argument("-v", "--verbose", action="store_true", help="모드별 오답 질의 출력")
    args = ap.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    items = data["items"]
    catalog = get_catalog()
    modes = [m.strip() for m in args.modes.split(",")]
    if any(m in ("embedding", "hybrid") for m in modes):
        catalog._embedding_index()  # 실패하면 EmbeddingUnavailable 로 중단 (폴백 방지)

    print(f"평가셋: {args.set} ({len(items)}건)")
    if data.get("_note"):
        print(f"주의: {data['_note']}\n")
    print(f"{'mode':<10}{'Top-1':>8}{'Top-3':>8}{'MRR':>8}")
    results = [evaluate(catalog, items, m) for m in modes]
    for r in results:
        print(f"{r['mode']:<10}{r['top1']:>8.0%}{r['top3']:>8.0%}{r['mrr']:>8.2f}")

    if args.verbose:
        for r in results:
            misses = [x for x in r["rows"] if x["rank"] != 1]
            print(f"\n[{r['mode']}] Top-1 이 아닌 질의 {len(misses)}건")
            for x in misses:
                got = ", ".join(x["ranked"][:3])
                print(f"  {x['id']} 정답 {x['expected']} / 순위 {x['rank']} / 상위3: {got}\n    {x['query']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
