"""시나리오 일괄 실행·평가 (이슈 #2 튜닝, #18 최종 평가 공용).

    cd backend
    python scripts/run_scenarios.py --dry-run                      # 실제 호출 없이 시나리오·설정만 확인
    python scripts/run_scenarios.py --limit 3 -v                    # 3건만 실제 실행
    python scripts/run_scenarios.py --models gpt-4o-mini,<다른모델> --search-mode hybrid --price-in X --price-out Y

주의: 실제 LLM API 를 호출하므로 **비용이 든다** (시나리오 1건 = 에이전트 7개의 LLM 호출 수십 회 가능).
 - 처음에는 --limit 으로 몇 건만 돌려 호출 수·토큰(출력에 표시)을 보고 전체 비용을 가늠한다.
 - 가격은 코드에 고정하지 않는다. --price-in/--price-out (100만 토큰당 금액)을 주면 그 단가로 비용을 계산한다.
 - 중간에 끊겨도 --resume 으로 끝난 시나리오는 건너뛴다 (결과는 시나리오마다 즉시 저장).
키는 출력하지 않는다. 결과 파일: eval/results/ (git 추적 제외).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # .env 를 실행 위치와 무관하게 읽는다

from app.config import get_settings  # noqa: E402
from app.core.pipeline import FaultAssessmentPipeline  # noqa: E402
from app.evaluation.scenarios import (  # noqa: E402
    format_comparison,
    format_misses,
    format_results,
    load_results,
    load_scenarios,
    run_all,
    save_results,
    summarize,
)


def apply_overrides(args, model: str | None) -> None:
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    provider = os.environ.get("LLM_PROVIDER") or get_settings().llm_provider
    if model:
        os.environ["OPENAI_MODEL" if provider == "openai" else "LLM_MODEL"] = model
    if args.search_mode:
        os.environ["CHART_SEARCH_MODE"] = args.search_mode
    if args.reasoning_effort is not None:
        os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    get_settings.cache_clear()


def result_path(args, set_name: str, label: str) -> Path:
    safe = label.replace("/", "_").replace(":", "_").replace(" ", "")
    p = ROOT / "eval" / "results" / f"{set_name}__{safe}.json"
    if p.exists() and not args.resume:  # 이전 결과를 덮어쓰지 않는다
        p = p.with_name(p.stem + dt.datetime.now().strftime("_%m%d-%H%M%S") + ".json")
    return p


async def run_for_model(args, scenarios, set_name: str, model: str | None):
    apply_overrides(args, model)
    s = get_settings()
    probe = FaultAssessmentPipeline().llm
    provider = probe.provider
    label = f"{provider}:{probe.model or '(미설정)'}/{s.chart_search_mode}" + (f"/effort={s.openai_reasoning_effort}" if s.openai_reasoning_effort and provider == "openai" else "")
    if probe.mock_mode:
        # mock 결과는 실제 실행과 라벨·파일명을 분리한다 (--resume 이 mock 결과를 실제 결과처럼 건너뛰는 사고 방지)
        label += "/MOCK"
    print(f"\n=== {label} | 시나리오 {len(scenarios)}건 | mock 모드: {probe.mock_mode} ===", flush=True)

    if probe.mock_mode and not args.allow_mock and not args.dry_run:
        print("API 키가 없어 mock 모드입니다 (결과가 무의미). 키를 넣거나, 구조만 확인하려면 --allow-mock 을 쓰세요.")
        return label, None
    if args.dry_run:
        print("[dry-run] 실제 호출 없이 종료합니다. 대상:", ", ".join(x.id for x in scenarios))
        return label, None

    out = result_path(args, set_name, label)
    done: list = []
    if args.resume and out.exists():
        _, done = load_results(out)
        skip = {r.id for r in done}
        scenarios = [x for x in scenarios if x.id not in skip]
        print(f"--resume: 이미 끝난 {len(skip)}건 건너뜀, 남은 {len(scenarios)}건 실행")
    meta = {"label": label, "provider": provider, "model": probe.model, "search_mode": s.chart_search_mode,
            "reasoning_effort": s.openai_reasoning_effort, "scenario_set": set_name,
            "started": dt.datetime.now().isoformat(timespec="seconds")}

    collected = list(done)

    def on_result(r):
        collected.append(r)
        save_results(out, meta, collected)  # 끊겨도 --resume 할 수 있게 즉시 저장
        flag = "오류" if not r.ok else f"A={r.final_a} 도표={r.primary_chart or '-'} {r.seconds:.0f}s 호출 {r.calls}회"
        print(f"  [{len(collected):>2}] {r.id}: {flag}", flush=True)

    await run_all(scenarios, FaultAssessmentPipeline, concurrency=args.concurrency, on_result=on_result)
    results = sorted(collected, key=lambda r: [x.id for x in load_scenarios(args.set)].index(r.id))
    save_results(out, meta, results)
    print(f"\n{format_results(results)}")
    if args.verbose:
        print(f"\n[오답 상세]\n{format_misses(results)}")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")
    return label, results


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(ROOT / "eval" / "scenarios_dev.json"), help="시나리오 JSON")
    ap.add_argument("--models", default="", help="쉼표로 구분한 모델 ID 목록 (비우면 .env 의 모델 하나)")
    ap.add_argument("--provider", choices=["openai", "anthropic"], help="LLM_PROVIDER 덮어쓰기")
    ap.add_argument("--search-mode", choices=["keyword", "embedding", "hybrid"], help="도표 검색 방식 덮어쓰기")
    ap.add_argument("--reasoning-effort", default=None, help="OPENAI_REASONING_EFFORT 덮어쓰기 (빈 문자열 = 모델 기본값)")
    ap.add_argument("--ids", default="", help="이 id 들만 실행 (쉼표 구분)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 실행")
    ap.add_argument("--concurrency", type=int, default=1, help="동시 실행 시나리오 수 (기본 1, 속도 제한에 주의)")
    ap.add_argument("--resume", action="store_true", help="기존 결과 파일에서 끝난 시나리오는 건너뜀")
    ap.add_argument("--dry-run", action="store_true", help="시나리오·설정 검증만, API 호출 없음")
    ap.add_argument("--allow-mock", action="store_true", help="키 없이 mock 으로 구조만 실행")
    ap.add_argument("--price-in", type=float, help="입력 100만 토큰당 단가 (비용 계산용)")
    ap.add_argument("--price-out", type=float, help="출력 100만 토큰당 단가 (비용 계산용)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    scenarios = load_scenarios(args.set)  # 형식·id 중복·정답 도표 오타를 먼저 검증
    if args.ids:
        want = {i.strip() for i in args.ids.split(",")}
        missing = want - {s.id for s in scenarios}
        if missing:
            print(f"없는 시나리오 id: {sorted(missing)}")
            return 2
        scenarios = [s for s in scenarios if s.id in want]
    if args.limit:
        scenarios = scenarios[: args.limit]
    set_name = Path(args.set).stem

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [None]
    rows = []
    for m in models:
        label, results = await run_for_model(args, scenarios, set_name, m)
        if results:
            rows.append((label, summarize(results)))

    if rows:
        print("\n=== 비교 요약 ===")
        print(format_comparison(rows, args.price_in, args.price_out))
        print("\n도표(주)=위원 판단의 중심 도표가 정답 / 범위=최종 A 과실이 정답 범위 / 애매플래그=애매 시나리오의 정밀 심사 플래그율")
        print("※ 개발용 시나리오(개발자 작성)의 수치는 보고용이 아닙니다. 직접 수집한 세트로 다시 측정하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
