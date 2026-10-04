"""실제 LLM API 연동 확인 (이슈 #1). API 키를 .env 에 넣은 뒤 실행한다.

    cd backend && python scripts/verify_llm.py

확인 항목 (각각 호출 1~수 회, 비용은 소액):
  1. 설정: 프로바이더·모델 ID (OpenAI 는 모델 목록 API 로 OPENAI_MODEL 이 실제 존재하는지)
  2. 단순 호출 complete()
  3. 구조화 출력 complete_json() — 실제 검토위원 스키마(ReviewerVerdict)를 API 가 받아주는지, 값 범위가 지켜지는지
  4. 도구 호출 루프 run_agent() — get_chart 를 실제로 호출하는지
키는 출력하지 않는다. 하나라도 실패하면 종료 코드 1.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.chart_tools import build_chart_tools  # noqa: E402
from app.core.llm_client import LLMClient  # noqa: E402
from app.models.schemas import ReviewerVerdict  # noqa: E402

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


async def check_model_exists(llm: LLMClient) -> None:
    if llm.provider != "openai":
        return
    ids = {m.id async for m in llm._client.models.list()}
    if llm.model in ids:
        record("모델 ID 존재", True, llm.model)
    else:
        sample = ", ".join(sorted(i for i in ids if i.startswith(("gpt", "o")))[:15])
        record("모델 ID 존재", False, f"'{llm.model}' 가 목록에 없음. 사용 가능 예: {sample}")


async def main() -> int:
    llm = LLMClient()
    print(f"프로바이더: {llm.provider} / 모델: {llm.model or '(미설정)'} / mock 모드: {llm.mock_mode}\n")
    if llm.mock_mode:
        key = "OPENAI_API_KEY" if llm.provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"{key} 가 비어 있어 mock 모드입니다. backend/.env 에 키를 넣고 다시 실행하세요.")
        return 2

    async def step(name, coro):
        t = time.time()
        try:
            detail = await coro
            record(name, True, f"{detail} ({time.time() - t:.1f}s)" if detail else f"{time.time() - t:.1f}s")
        except Exception as e:  # 어떤 실패든 원인을 보여주고 다음 항목으로
            record(name, False, f"{type(e).__name__}: {str(e)[:300]}")

    await step("모델 목록 조회", check_model_exists(llm))

    async def simple():
        out = await llm.complete("한 문장으로만 답하세요.", "교통사고 과실비율이 뭔지 설명해 주세요.")
        assert out.strip(), "빈 응답"
        return f"응답 {len(out)}자"

    async def structured():
        v = await llm.complete_json(
            "당신은 교통사고 과실비율 검토위원입니다. 지정된 JSON 스키마로만 답하세요.",
            "[사고 정황]\n사고 상황 설명: 제가(A) 1차선에서 직진 중, 옆 차선의 상대 차량(B)이 방향지시등 없이 갑자기 "
            "제 앞으로 차선을 변경해 충돌했습니다.\n[체크리스트] 1. 도표 선택 2. 기본 과실비율 3. 가감요소\n"
            "checklist 에 체크리스트 항목마다 판단을 적으세요.",
            ReviewerVerdict,
            mock={},
            max_attempts=1,  # 재시도 없이 첫 응답이 스키마를 지키는지 본다
        )
        assert 0 <= v.fault_ratio_a <= 100 and v.checklist, "범위/체크리스트 이상"
        return f"A 과실 {v.fault_ratio_a}, confidence={v.confidence}, 체크리스트 {len(v.checklist)}개"

    async def agent():
        trace: list[str] = []
        out = await llm.run_agent(
            "당신은 과실비율 기준표 전문가입니다. 답하기 전에 반드시 get_chart 도구로 도표를 조회하세요.",
            "도표 차43-2 의 기본 과실비율을 알려 주세요.",
            build_chart_tools(trace),
            max_iterations=4,
        )
        assert trace, "도구를 한 번도 호출하지 않음"
        assert "30" in out and "70" in out, f"기본 과실 30:70 이 응답에 없음: {out[:120]}"
        return f"도구 호출 {len(trace)}회 {trace}"

    await step("단순 호출 complete()", simple())
    await step("구조화 출력 complete_json() (첫 응답이 스키마 준수)", structured())
    await step("도구 호출 루프 run_agent()", agent())

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
