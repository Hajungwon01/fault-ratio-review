"""LLMClient.usage — 호출 수·토큰 집계 (재시도와 도구 루프의 매 호출도 비용이므로 모두 센다)."""
import asyncio
import json

import anthropic
import httpx2 as httpx
import openai

from app.core.chart_tools import build_chart_tools
from app.core.llm_client import LLMClient
from app.models.schemas import ReviewerVerdict

GOOD = {"checklist": [], "rationale": "근거", "chart_no": "", "chart_a_is": "none", "chart_a_fault": 30,
        "confidence": "high", "cited_rules": []}


# ---------- OpenAI ----------
def oa(output, i=100, o=20):
    return {"id": "r", "object": "response", "created_at": 0, "status": "completed", "model": "m", "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def oa_text(t, i=100, o=20):
    return oa([{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": t, "annotations": []}]}], i, o)


def oa_call(name, args, i=100, o=20):
    return oa([{"type": "function_call", "id": "f", "call_id": "c1", "name": name,
                "arguments": json.dumps(args, ensure_ascii=False), "status": "completed"}], i, o)


def openai_llm(monkeypatch, replies):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    from app.config import get_settings

    get_settings.cache_clear()
    n = {"i": 0}

    def handler(request):
        r = replies[min(n["i"], len(replies) - 1)]
        n["i"] += 1
        return httpx.Response(200, json=r)

    llm = LLMClient()
    llm._client = openai.AsyncOpenAI(api_key="d", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return llm


def test_openai_counts_each_call_and_tokens(monkeypatch):
    llm = openai_llm(monkeypatch, [oa_text("a", 100, 20), oa_text("b", 40, 10)])
    asyncio.run(llm.complete("s", "u"))
    asyncio.run(llm.complete("s", "u"))
    assert llm.usage.as_dict() == {"calls": 2, "input_tokens": 140, "output_tokens": 30}


def test_openai_tool_loop_counts_every_round_trip(monkeypatch):
    llm = openai_llm(monkeypatch, [oa_call("get_chart", {"chart_no": "차43-2"}, 300, 30), oa_text("끝", 500, 60)])
    asyncio.run(llm.run_agent("s", "u", build_chart_tools([])))
    assert llm.usage.as_dict() == {"calls": 2, "input_tokens": 800, "output_tokens": 90}


def test_openai_structured_output_retries_are_counted(monkeypatch):
    llm = openai_llm(monkeypatch, [oa_text("깨짐", 100, 20), oa_text(json.dumps(GOOD), 110, 25)])
    asyncio.run(llm.complete_json("s", "u", ReviewerVerdict, mock={}))
    assert llm.usage.calls == 2 and llm.usage.input_tokens == 210 and llm.usage.output_tokens == 45


def test_usage_missing_in_response_counts_call_but_not_tokens(monkeypatch):
    bare = oa_text("a")
    del bare["usage"]
    llm = openai_llm(monkeypatch, [bare])
    asyncio.run(llm.complete("s", "u"))
    assert llm.usage.as_dict() == {"calls": 1, "input_tokens": 0, "output_tokens": 0}


def test_mock_mode_usage_stays_zero():
    llm = LLMClient()
    assert llm.mock_mode
    asyncio.run(llm.complete("s", "u"))
    asyncio.run(llm.complete_json("s", "u", ReviewerVerdict, mock=GOOD))
    assert llm.usage.as_dict() == {"calls": 0, "input_tokens": 0, "output_tokens": 0}


# ---------- Anthropic ----------
def an(content, stop, i=50, o=10):
    return {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": content, "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": i, "output_tokens": o}}


def anthropic_llm(monkeypatch, replies):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    from app.config import get_settings

    get_settings.cache_clear()
    n = {"i": 0}

    def handler(request):
        r = replies[min(n["i"], len(replies) - 1)]
        n["i"] += 1
        return httpx.Response(200, json=r)

    llm = LLMClient()
    llm._client = anthropic.AsyncAnthropic(api_key="d", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return llm


def test_anthropic_counts_complete_and_structured(monkeypatch):
    llm = anthropic_llm(monkeypatch, [an([{"type": "text", "text": "a"}], "end_turn", 50, 10),
                                      an([{"type": "text", "text": json.dumps(GOOD)}], "end_turn", 70, 15)])
    asyncio.run(llm.complete("s", "u"))
    asyncio.run(llm.complete_json("s", "u", ReviewerVerdict, mock={}))
    assert llm.usage.as_dict() == {"calls": 2, "input_tokens": 120, "output_tokens": 25}


def test_anthropic_tool_runner_counts_every_message(monkeypatch):
    llm = anthropic_llm(monkeypatch, [
        an([{"type": "tool_use", "id": "t1", "name": "get_chart", "input": {"chart_no": "차43-2"}}], "tool_use", 200, 20),
        an([{"type": "text", "text": "끝"}], "end_turn", 300, 40),
    ])
    asyncio.run(llm.run_agent("s", "u", build_chart_tools([])))
    assert llm.usage.as_dict() == {"calls": 2, "input_tokens": 500, "output_tokens": 60}


def test_each_pipeline_gets_its_own_usage_counter():
    from app.core.pipeline import FaultAssessmentPipeline

    a, b = FaultAssessmentPipeline(), FaultAssessmentPipeline()
    assert a.llm is not b.llm and a.llm.usage is not b.llm.usage
