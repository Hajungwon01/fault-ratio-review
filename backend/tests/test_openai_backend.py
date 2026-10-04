"""OpenAI 백엔드(Responses API) 테스트 — 실제 API 없이 httpx2.MockTransport 로 /v1/responses 를 흉내낸다.

검증 범위: 요청 형태(instructions/text.format/tools), 함수 호출 루프, 구조화 출력 재시도, 거절·잘림 처리.
실제 모델이 스키마·도구 호출을 얼마나 잘 따르는지는 API 키를 넣은 뒤 확인해야 한다.
"""
import asyncio
import json

import httpx2 as httpx
import openai
import pytest

from app.agents.reviewer_standard import ReviewerStandardAgent
from app.core.chart_tools import build_chart_tools
from app.core.llm_client import LLMClient, StructuredOutputError, strict_json_schema
from app.models.schemas import ReviewerVerdict

GOOD = {"fault_ratio_a": 30, "confidence": "high", "rationale": "차43-2 기본 과실 30:70",
        "checklist": [{"item": "도표 선택", "finding": "차43-2"}], "cited_rules": ["차43-2"]}


def _resp(output, status="completed", incomplete=None):
    r = {"id": "resp_x", "object": "response", "created_at": 0, "status": status, "model": "test-model",
         "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
    if incomplete:
        r["incomplete_details"] = {"reason": incomplete}
    return r


def text(t):
    return _resp([{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": t, "annotations": []}]}])


def call(name, args, call_id="call_1"):
    return _resp([{"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
                   "arguments": json.dumps(args, ensure_ascii=False), "status": "completed"}])


REFUSAL = _resp([{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                  "content": [{"type": "refusal", "refusal": "도와드릴 수 없습니다"}]}])
TRUNCATED = _resp([], status="incomplete", incomplete="max_output_tokens")


def make_llm(monkeypatch, replies, model="test-model"):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", model)
    from app.config import get_settings

    get_settings.cache_clear()
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies[min(len(seen) - 1, len(replies) - 1)])

    llm = LLMClient()
    llm._client = openai.AsyncOpenAI(api_key="dummy", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    get_settings.cache_clear()
    return llm, seen


def json_call(llm):
    return asyncio.run(llm.complete_json("시스템", "사용자 프롬프트", ReviewerVerdict, mock=GOOD))


# ---------- 설정 ----------
def test_openai_without_key_is_mock_mode(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert LLMClient().mock_mode
    finally:
        get_settings.cache_clear()


def test_openai_key_without_model_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            LLMClient()
    finally:
        get_settings.cache_clear()


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            LLMClient()
    finally:
        get_settings.cache_clear()


# ---------- strict 스키마 ----------
def test_strict_schema_requires_all_fields_and_closes_objects():
    s = strict_json_schema(ReviewerVerdict)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"]) == {"fault_ratio_a", "confidence", "checklist", "rationale", "cited_rules"}
    item = s["$defs"]["ChecklistItem"]
    assert item["additionalProperties"] is False and set(item["required"]) == {"item", "finding"}
    dumped = json.dumps(s)
    for dropped in ('"default"', '"minimum"', '"maximum"', '"minLength"', '"title"'):
        assert dropped not in dumped
    assert s["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


# ---------- 구조화 출력 ----------
def test_complete_json_sends_strict_schema_and_parses(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text(json.dumps(GOOD))])
    v = json_call(llm)
    assert v.fault_ratio_a == 30 and v.checklist[0].item == "도표 선택" and v.cited_rules == ["차43-2"]
    body = seen[0]
    assert body["model"] == "test-model" and body["instructions"] == "시스템" and body["input"] == "사용자 프롬프트"
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "ReviewerVerdict"
    assert "reasoning" not in body  # OPENAI_REASONING_EFFORT 미설정이면 보내지 않는다


def test_reasoning_effort_sent_only_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    llm, seen = make_llm(monkeypatch, [text(json.dumps(GOOD))])
    json_call(llm)
    assert seen[0]["reasoning"] == {"effort": "low"}


def test_retry_on_invalid_json_then_success(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text("JSON 아님"), text(json.dumps(GOOD))])
    assert json_call(llm).fault_ratio_a == 30 and len(seen) == 2
    assert "[직전 응답 오류]" in seen[1]["input"] and "사용자 프롬프트" in seen[1]["input"]


def test_retry_on_out_of_range_ratio(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text(json.dumps({**GOOD, "fault_ratio_a": 150})), text(json.dumps(GOOD))])
    assert json_call(llm).fault_ratio_a == 30 and len(seen) == 2


def test_truncated_output_is_retried(monkeypatch):
    llm, seen = make_llm(monkeypatch, [TRUNCATED, text(json.dumps(GOOD))])
    assert json_call(llm).fault_ratio_a == 30 and len(seen) == 2


def test_gives_up_after_three_attempts(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text("깨짐")])
    with pytest.raises(StructuredOutputError):
        json_call(llm)
    assert len(seen) == 3


def test_refusal_not_retried(monkeypatch):
    llm, seen = make_llm(monkeypatch, [REFUSAL])
    with pytest.raises(RuntimeError, match="거절"):
        json_call(llm)
    assert len(seen) == 1


# ---------- 함수 호출 루프 ----------
def run(llm, trace, **kw):
    return asyncio.run(llm.run_agent("시스템", "사용자", build_chart_tools(trace), **kw))


def test_agent_loop_executes_tool_and_feeds_result_back(monkeypatch):
    llm, seen = make_llm(monkeypatch, [call("get_chart", {"chart_no": "차43-2"}), text("최종 논증. 인용: 차43-2")])
    trace = []
    assert run(llm, trace) == "최종 논증. 인용: 차43-2"
    assert trace == ["get_chart('차43-2')"] and len(seen) == 2

    first = seen[0]
    assert first["instructions"] == "시스템" and first["input"] == [{"role": "user", "content": "사용자"}]
    assert {t["name"] for t in first["tools"]} == {"search_charts", "get_chart"}
    assert all(t["type"] == "function" and t["parameters"]["type"] == "object" for t in first["tools"])

    # 두 번째 요청: 앞선 function_call 이 그대로 이어지고, 그 call_id 로 function_call_output 이 붙는다
    inp = seen[1]["input"]
    assert [i.get("type") for i in inp] == [None, "function_call", "function_call_output"]
    assert inp[1]["call_id"] == inp[2]["call_id"] == "call_1"
    assert "기본 과실" in inp[2]["output"]


def test_agent_loop_runs_parallel_calls_and_returns_all_outputs(monkeypatch):
    two = _resp([
        {"type": "function_call", "id": "f1", "call_id": "c1", "name": "get_chart", "arguments": '{"chart_no":"차43-2"}', "status": "completed"},
        {"type": "function_call", "id": "f2", "call_id": "c2", "name": "get_chart", "arguments": '{"chart_no":"차41-1"}', "status": "completed"},
    ])
    llm, seen = make_llm(monkeypatch, [two, text("끝")])
    trace = []
    run(llm, trace)
    outs = [i for i in seen[1]["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outs] == ["c1", "c2"] and len(trace) == 2


def test_tool_errors_are_reported_to_model_not_raised(monkeypatch):
    bad_args = call("get_chart", {"wrong_arg": 1}, "c_bad")
    unknown = call("no_such_tool", {}, "c_unk")
    llm, seen = make_llm(monkeypatch, [bad_args, unknown, text("복구 후 답")])
    assert run(llm, []) == "복구 후 답"
    out1 = [i for i in seen[1]["input"] if i.get("type") == "function_call_output"][0]["output"]
    out2 = [i for i in seen[2]["input"] if i.get("type") == "function_call_output"][-1]["output"]
    assert "도구 실행 오류" in out1 and "알 수 없는 도구" in out2


def test_iteration_limit_forces_final_answer_without_tools(monkeypatch):
    # 항상 도구만 부르는 서버: 한도(3회) 후 마지막 요청은 tool_choice=none 이어야 한다
    llm, seen = make_llm(monkeypatch, [call("get_chart", {"chart_no": "차43-2"})] * 3 + [text("한도 후 최종")])
    assert run(llm, [], max_iterations=3) == "한도 후 최종"
    assert len(seen) == 4 and seen[3]["tool_choice"] == "none" and "tool_choice" not in seen[0]


def test_agent_refusal_raises(monkeypatch):
    llm, _ = make_llm(monkeypatch, [REFUSAL])
    with pytest.raises(RuntimeError, match="거절"):
        run(llm, [])


def test_plain_complete(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text("요약")])
    assert asyncio.run(llm.complete("시스템", "사용자")) == "요약" and seen[0]["input"] == "사용자"


# ---------- 에이전트 통합 ----------
def test_reviewer_runs_through_openai_backend(monkeypatch):
    llm, seen = make_llm(monkeypatch, [text(json.dumps({**GOOD, "fault_ratio_a": 40}))])
    s = asyncio.run(ReviewerStandardAgent(llm).run(
        {"accident_input": "x", "rounds": [], "fact_check": None, "cited_charts": "■ 차43-2"}))
    assert (s.fault_ratio_a, s.fault_ratio_b) == (40, 60) and s.checklist[0].finding == "차43-2"
    assert "[출력 규칙]" in seen[0]["instructions"] and "[체크리스트]" in seen[0]["instructions"]
