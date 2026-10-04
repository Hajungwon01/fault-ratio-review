"""실제 API 없이 Tool Runner 루프 배선을 검증한다 (httpx MockTransport 로 /v1/messages 를 흉내)."""
import asyncio
import json

import anthropic
import httpx2 as httpx

from app.core.chart_tools import build_chart_tools
from app.core.llm_client import LLMClient


def _msg(content, stop_reason):
    return {
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_run_agent_executes_tools_then_returns_final_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    from app.config import get_settings

    get_settings.cache_clear()
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            return httpx.Response(200, json=_msg(
                [{"type": "tool_use", "id": "tu_1", "name": "get_chart", "input": {"chart_no": "차43-2"}}], "tool_use"))
        return httpx.Response(200, json=_msg(
            [{"type": "text", "text": "최종 논증. 인용: 차43-2"}], "end_turn"))

    try:
        llm = LLMClient()
        assert not llm.mock_mode
        llm._client = anthropic.AsyncAnthropic(
            api_key="dummy", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        trace: list[str] = []
        out = asyncio.run(llm.run_agent("시스템", "사용자", build_chart_tools(trace)))
    finally:
        get_settings.cache_clear()

    assert out == "최종 논증. 인용: 차43-2"
    assert trace == ["get_chart('차43-2')"]
    assert len(requests_seen) == 2
    # 두 번째 요청에는 도구 실행 결과(tool_result)가 담겨 있어야 한다
    last_user = requests_seen[1]["messages"][-1]
    assert last_user["content"][0]["type"] == "tool_result"
    assert "기본 과실" in json.dumps(last_user, ensure_ascii=False)
    assert {t["name"] for t in requests_seen[0]["tools"]} == {"search_charts", "get_chart"}
