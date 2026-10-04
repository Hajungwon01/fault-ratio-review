"""검토위원 구조화 출력(complete_json) + 재시도. 실제 API 없이 httpx2.MockTransport 로 /v1/messages 를 흉내낸다."""
import asyncio
import json

import anthropic
import httpx2 as httpx
import pytest

from app.agents.reviewer_standard import ReviewerStandardAgent
from app.core.llm_client import LLMClient, StructuredOutputError
from app.core.pipeline import FaultAssessmentPipeline
from app.models.schemas import AgentArgument, FactCheckResult, ReviewerVerdict

GOOD = {"fault_ratio_a": 30, "confidence": "high", "rationale": "차43-2 기본 과실 30:70 적용", "cited_rules": ["차43-2"]}


def _text_msg(text, stop_reason="end_turn"):
    return {
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def make_llm(monkeypatch, replies):
    """replies: 요청 순서대로 돌려줄 응답 text 목록(마지막 값 반복). 보낸 요청 본문은 seen 에 쌓인다."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    from app.config import get_settings

    get_settings.cache_clear()
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        i = min(len(seen) - 1, len(replies) - 1)
        r = replies[i]
        stop = "refusal" if r == "REFUSAL" else "end_turn"
        return httpx.Response(200, json=_text_msg("" if r == "REFUSAL" else r, stop))

    llm = LLMClient()
    llm._client = anthropic.AsyncAnthropic(api_key="dummy", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    get_settings.cache_clear()
    return llm, seen


def call(llm):
    return asyncio.run(llm.complete_json("시스템", "사용자 프롬프트", ReviewerVerdict, mock=GOOD))


def test_valid_response_parsed_and_schema_sent(monkeypatch):
    llm, seen = make_llm(monkeypatch, [json.dumps(GOOD)])
    v = call(llm)
    assert v.fault_ratio_a == 30 and v.confidence == "high" and v.cited_rules == ["차43-2"]
    assert len(seen) == 1
    fmt = seen[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "fault_ratio_a" in fmt["schema"]["properties"]


def test_retries_after_invalid_json_and_tells_model_why(monkeypatch):
    llm, seen = make_llm(monkeypatch, ["이건 JSON 이 아닙니다", json.dumps(GOOD)])
    assert call(llm).fault_ratio_a == 30
    assert len(seen) == 2
    retry_prompt = seen[1]["messages"][0]["content"]
    assert "사용자 프롬프트" in retry_prompt and "[직전 응답 오류]" in retry_prompt


def test_out_of_range_ratio_triggers_retry(monkeypatch):
    bad = {**GOOD, "fault_ratio_a": 150}
    llm, seen = make_llm(monkeypatch, [json.dumps(bad), json.dumps(GOOD)])
    assert call(llm).fault_ratio_a == 30 and len(seen) == 2


def test_gives_up_after_max_attempts(monkeypatch):
    llm, seen = make_llm(monkeypatch, ["깨진 응답"])
    with pytest.raises(StructuredOutputError):
        call(llm)
    assert len(seen) == 3


def test_refusal_is_not_retried(monkeypatch):
    llm, seen = make_llm(monkeypatch, ["REFUSAL"])
    with pytest.raises(RuntimeError, match="거절"):
        call(llm)
    assert len(seen) == 1


def test_mock_mode_returns_validated_mock_without_calling_api():
    llm = LLMClient()
    assert llm.mock_mode
    assert call(llm).fault_ratio_a == 30


def test_mock_payload_is_validated_too():
    llm = LLMClient()
    with pytest.raises(ValueError):
        asyncio.run(llm.complete_json("s", "u", ReviewerVerdict, mock={**GOOD, "fault_ratio_a": 999}))


# ---------- ReviewerAgent ----------
CTX = {
    "accident_input": "사고 상황 설명: 제가(A) 직진 중 상대 차량(B)이 차선 변경",
    "rounds": [
        AgentArgument(agent_name="vehicle_a_agent", round=1, content="A 논증 본문", cited_rules=["차43-2"]),
        AgentArgument(agent_name="vehicle_b_agent", round=1, content="B 논증 본문"),
    ],
    "fact_check": FactCheckResult(is_valid=False, notes="", flagged_citations=["B측 차99-9 는 존재하지 않는 도표번호"]),
}


def test_reviewer_builds_score_with_b_as_complement(monkeypatch):
    llm, seen = make_llm(monkeypatch, [json.dumps({**GOOD, "fault_ratio_a": 40})])
    s = asyncio.run(ReviewerStandardAgent(llm).run(CTX))
    assert (s.fault_ratio_a, s.fault_ratio_b) == (40, 60)
    assert s.reviewer_name == "reviewer_standard_agent" and s.perspective == "기준표 적용"
    assert s.confidence == "high" and s.cited_rules == ["차43-2"]
    assert "[출력 규칙]" in seen[0]["system"][0]["text"] if isinstance(seen[0]["system"], list) else "[출력 규칙]" in seen[0]["system"]


def test_reviewer_prompt_is_readable_not_repr():
    p = ReviewerStandardAgent(LLMClient())._build_user_prompt(CTX)
    assert "[A측 1라운드]\nA 논증 본문\n인용: 차43-2" in p and "[B측 1라운드]" in p
    assert "인용 오류 있음" in p and "- B측 차99-9" in p
    assert "AgentArgument" not in p and "FactCheckResult" not in p and "agent_name" not in p


def test_pipeline_reviews_in_mock_mode_are_valid_scores():
    r = asyncio.run(FaultAssessmentPipeline().run(
        __import__("app.models.schemas", fromlist=["AccidentInput"]).AccidentInput(
            description="녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다")))
    assert len(r.reviews) == 3 and all(x.fault_ratio_a + x.fault_ratio_b == 100 for x in r.reviews)
    assert all(x.confidence == "low" and "MOCK" in x.rationale for x in r.reviews)
