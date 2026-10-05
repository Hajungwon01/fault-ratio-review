"""429(속도 제한) 재시도 — 실제 첫 비교 실행에서 위원 3명이 동시에 호출되며 TPM 한도에 걸려 탈락한 문제를 고정한다."""
import asyncio
import json

import httpx2 as httpx
import openai
import pytest

from app.core import llm_client
from app.core.llm_client import LLMClient, call_with_rate_limit_retry
from app.models.schemas import ReviewerVerdict

GOOD = {"checklist": [], "rationale": "근거", "chart_no": "", "chart_a_is": "none", "chart_a_fault": 30,
        "confidence": "high", "cited_rules": []}
MSG429 = ("Rate limit reached for gpt-4o on tokens per min (TPM): Limit 30000, Used 27016, Requested 5981. "
          "Please try again in 5.994s.")


class Err(Exception):
    def __init__(self, status, msg="x", headers=None):
        super().__init__(msg)
        self.status_code = status
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake(sec):
        waited.append(sec)

    monkeypatch.setattr(llm_client, "_sleep", fake)
    return waited


def run_with(errors, result="ok", **kw):
    seq = list(errors)
    calls = {"n": 0}

    async def req():
        calls["n"] += 1
        if seq:
            raise seq.pop(0)
        return result

    return asyncio.run(call_with_rate_limit_retry(req, **kw)), calls["n"]


def test_retries_429_then_succeeds(sleeps):
    out, n = run_with([Err(429, MSG429), Err(429, MSG429)])
    assert out == "ok" and n == 3 and len(sleeps) == 2


def test_waits_the_time_the_server_asked_for(sleeps):
    run_with([Err(429, MSG429)])
    assert 5.994 + 0.5 <= sleeps[0] <= 5.994 + 1.5 + 1e-6      # 문구의 "try again in 5.994s" + 여유 + 지터


def test_retry_after_header_takes_precedence(sleeps):
    run_with([Err(429, MSG429, headers={"retry-after-ms": "2000"})])
    assert 2.5 <= sleeps[0] <= 3.5
    run_with([Err(429, "x", headers={"retry-after": "4"})])
    assert 4.5 <= sleeps[1] <= 5.5


def test_ms_unit_in_message(sleeps):
    run_with([Err(429, "Please try again in 250ms.")])
    assert sleeps[0] < 2


def test_exponential_backoff_without_hint_and_cap(sleeps):
    run_with([Err(429), Err(429), Err(429)])
    for s, base in zip(sleeps, (1, 2, 4)):       # 2**attempt + 0.5 + 지터(0~1)
        assert base + 0.5 <= s <= base + 1.5
    run_with([Err(429, "try again in 9999s")])
    assert sleeps[-1] <= llm_client.RATE_LIMIT_MAX_WAIT


def test_gives_up_after_max_retries(sleeps):
    with pytest.raises(Err):
        run_with([Err(429, MSG429)] * 10, max_retries=2)
    assert len(sleeps) == 2


def test_quota_exhaustion_and_other_errors_are_not_retried(sleeps):
    for e in (Err(429, "You exceeded your current quota (insufficient_quota)"), Err(500), Err(400), ValueError("x")):
        with pytest.raises(Exception):
            run_with([e])
    assert sleeps == []


# ---------- OpenAI 백엔드 통합: 429 응답 뒤 성공 ----------
def test_openai_backend_recovers_from_429(monkeypatch, sleeps):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    from app.config import get_settings

    get_settings.cache_clear()
    n = {"i": 0}
    ok = {"id": "r", "object": "response", "created_at": 0, "status": "completed", "model": "m", "parallel_tool_calls": True,
          "tool_choice": "auto", "tools": [],
          "output": [{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                      "content": [{"type": "output_text", "text": json.dumps(GOOD), "annotations": []}]}],
          "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}

    def handler(request):
        n["i"] += 1
        if n["i"] <= 2:
            return httpx.Response(429, json={"error": {"message": MSG429, "type": "tokens", "code": "rate_limit_exceeded"}})
        return httpx.Response(200, json=ok)

    llm = LLMClient()
    llm._client = openai.AsyncOpenAI(api_key="d", max_retries=0,
                                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    v = asyncio.run(llm.complete_json("s", "u", ReviewerVerdict, mock=GOOD))
    assert v.chart_a_fault == 30 and n["i"] == 3 and len(sleeps) == 2
    assert llm.usage.calls == 1   # 실패한 429 호출은 토큰을 쓰지 않으므로 성공한 호출만 센다
