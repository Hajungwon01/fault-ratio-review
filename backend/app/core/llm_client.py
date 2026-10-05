"""LLM 호출을 감싸는 얇은 래퍼 — 프로바이더(OpenAI / Anthropic) 선택형.

API 키가 설정되지 않은 동안(mock 모드)에는 실제 호출 없이 고정된 형태의 응답을 돌려줘서,
키 없이도 오케스트레이션 파이프라인 전체를 실행/테스트할 수 있게 한다.

- complete():      도구 없는 단발 텍스트 호출
- complete_json(): 구조화 출력(pydantic) + 파싱·검증 실패 시 재시도 (검토위원 등)
- run_agent():     도구 호출 루프. 모델이 도구(기준표 조회 등)를 스스로 호출하며 결론까지 진행한다.

프로바이더는 설정 LLM_PROVIDER(openai | anthropic)로 고른다. 각 백엔드는 "실제 호출"만 맡고,
mock 처리와 구조화 출력의 검증·재시도는 이 파일의 LLMClient 가 공통으로 처리한다.

TODO: 실제 API 호출 경로는 API 키 설정 후 검증 필요 (아직 가짜 서버(MockTransport) 테스트와 mock 경로만 검증됨).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    """LLM 호출 집계 (호출 수, 입력/출력 토큰). 평가 도구가 시나리오별 비용을 보려고 쓴다. 가격은 여기서 다루지 않는다."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response) -> None:
        self.calls += 1
        u = getattr(response, "usage", None)
        if u is not None:
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


class StructuredOutputError(RuntimeError):
    """구조화 출력을 재시도 한도 안에 얻지 못했을 때."""


class ModelRefusal(RuntimeError):
    """모델이 응답을 거절함 (재시도하지 않는다)."""


class IncompleteOutput(ValueError):
    """응답이 길이 제한 등으로 잘림 (구조화 출력에서는 재시도 대상)."""


# ---------------------------------------------------------------------------
# 속도 제한(429) 재시도
# ---------------------------------------------------------------------------
# 위원 3명이 동시에 호출되면 분당 토큰(TPM) 한도를 넘기 쉽다. SDK 기본 재시도(2회, 최대 8초 대기)로는 부족해서
# 서버가 알려 준 대기 시간만큼 기다린 뒤 다시 보낸다. 한도 소진(insufficient_quota)은 기다려도 풀리지 않으므로 재시도하지 않는다.
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_MAX_WAIT = 60.0  # 한 번에 기다리는 최대 초
_sleep = asyncio.sleep  # 테스트에서 교체한다


def _is_retryable_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 and "insufficient_quota" not in str(exc)


def _rate_limit_delay(exc: Exception, attempt: int) -> float:
    """서버가 알려 준 대기 시간(헤더 → 오류 문구 순)을 우선하고, 없으면 지수 백오프. 동시 재시도가 겹치지 않게 지터를 더한다."""
    import random
    import re

    headers = getattr(getattr(exc, "response", None), "headers", None)
    wait: float | None = None
    try:
        if headers is not None and headers.get("retry-after-ms"):
            wait = float(headers["retry-after-ms"]) / 1000
        elif headers is not None and headers.get("retry-after"):
            wait = float(headers["retry-after"])
    except (TypeError, ValueError):
        wait = None
    if wait is None:
        m = re.search(r"try again in ([\d.]+)\s*(ms|s)\b", str(exc))
        if m:
            wait = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
    if wait is None:
        wait = min(2.0 ** attempt, 30.0)
    return min(wait + 0.5 + random.uniform(0, 1.0), RATE_LIMIT_MAX_WAIT)


async def call_with_rate_limit_retry(make_request, *, max_retries: int = RATE_LIMIT_MAX_RETRIES):
    """make_request() 를 호출하고, 429 이면 대기 후 최대 max_retries 회 다시 호출한다. 그 밖의 오류는 그대로 던진다."""
    for attempt in range(max_retries + 1):
        try:
            return await make_request()
        except Exception as e:
            if attempt >= max_retries or not _is_retryable_rate_limit(e):
                raise
            await _sleep(_rate_limit_delay(e, attempt))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
_STRICT_DROP = {
    "default", "title", "format", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "pattern",
}


def strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """pydantic 모델을 OpenAI Structured Outputs strict 모드 스키마로 바꾼다.

    strict 모드는 모든 객체에 additionalProperties=false 와 '모든 필드 required' 를 요구하고
    값 범위 제약(min/max 등)을 지원하지 않는다. 제약 위반은 응답을 받은 뒤 pydantic 이 검증한다.
    """

    def fix(node: Any) -> Any:
        if isinstance(node, list):
            return [fix(x) for x in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRICT_DROP:
                continue
            if key in ("properties", "$defs"):  # 이름→스키마 매핑: 키 자체는 필터하지 않는다
                out[key] = {name: fix(sub) for name, sub in value.items()}
            else:
                out[key] = fix(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return fix(model_cls.model_json_schema())


class _OpenAIBackend:
    def __init__(self, settings) -> None:
        self.model = settings.openai_model
        self.reasoning_effort = settings.openai_reasoning_effort
        self.usage = Usage()
        self.mock_mode = not settings.openai_api_key
        self._client = None
        if not self.mock_mode:
            if not self.model:
                raise ValueError("OPENAI_MODEL 이 설정되지 않았습니다. 사용할 OpenAI 모델 ID 를 .env 에 지정하세요.")
            import openai

            # 파이프라인이 asyncio.gather 로 에이전트를 병렬 실행하므로 async 클라이언트를 쓴다.
            # SDK 기본 재시도(연결 오류·5xx·짧은 429)는 그대로 두고, 그래도 남는 429 는 _create 가 이어서 처리한다.
            self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def _create(self, **params):
        return await call_with_rate_limit_retry(lambda: self._client.responses.create(**params))

    def _params(self, system_prompt: str, **extra) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "instructions": system_prompt, **extra}
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}
        return params

    def _check(self, response, *, allow_incomplete: bool = False) -> None:
        self.usage.add(response)  # 모든 호출이 이 검사를 거치므로 여기서 집계한다
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise ModelRefusal(f"모델이 응답을 거절했습니다: {part.refusal}")
        if response.status == "incomplete" and not allow_incomplete:
            reason = getattr(response.incomplete_details, "reason", None)
            raise RuntimeError(f"응답이 완료되지 않았습니다: {reason}")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._create(**self._params(system_prompt, input=user_prompt))
        self._check(response)
        return response.output_text

    async def generate_json(self, system_prompt: str, user_prompt: str, model_cls: type[BaseModel]) -> str:
        response = await self._create(
            **self._params(
                system_prompt,
                input=user_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": model_cls.__name__,
                        "schema": strict_json_schema(model_cls),
                        "strict": True,
                    }
                },
            )
        )
        self._check(response, allow_incomplete=True)
        if response.status == "incomplete":
            raise IncompleteOutput(f"응답이 잘렸습니다: {getattr(response.incomplete_details, 'reason', None)}")
        return response.output_text

    async def run_agent(self, system_prompt: str, user_prompt: str, tools: list, max_iterations: int) -> str:
        """Responses API 함수 호출 루프: function_call 이 없을 때까지 도구를 실행하고 결과를 돌려준다."""
        by_name = {t.name: t for t in tools}
        spec = [t.to_openai() for t in tools]
        input_list: list[Any] = [{"role": "user", "content": user_prompt}]

        for _ in range(max_iterations):
            response = await self._create(
                **self._params(system_prompt, input=input_list, tools=spec)
            )
            self._check(response)
            # 추론(reasoning) 항목과 function_call 항목을 그대로 다음 입력에 이어 붙여야 한다
            input_list += response.output
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text
            input_list += await asyncio.gather(*(self._execute(by_name, c) for c in calls))

        # 반복 한도: 도구 없이 지금까지 얻은 정보로 답하게 한다
        response = await self._create(
            **self._params(system_prompt, input=input_list, tools=spec, tool_choice="none")
        )
        self._check(response)
        return response.output_text

    @staticmethod
    async def _execute(by_name: dict, call) -> dict[str, Any]:
        tool = by_name.get(call.name)
        if tool is None:
            output = f"알 수 없는 도구입니다: {call.name}"
        else:
            try:
                output = await tool.call(json.loads(call.arguments or "{}"))
            except Exception as e:  # 도구 오류는 모델에게 알려 스스로 고치게 한다 (루프를 죽이지 않는다)
                output = f"도구 실행 오류: {type(e).__name__}: {e}"
        return {"type": "function_call_output", "call_id": call.call_id, "output": str(output)}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class _AnthropicBackend:
    def __init__(self, settings) -> None:
        self.model = settings.llm_model
        self.usage = Usage()
        self.mock_mode = not settings.anthropic_api_key
        self._client = None
        if not self.mock_mode:
            import anthropic

            # tool_runner 는 내부에서 요청하므로 SDK 재시도 횟수를 늘려 둔다. complete/generate_json 은 _create 가 추가로 처리한다.
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=RATE_LIMIT_MAX_RETRIES)

    async def _create(self, **params):
        return await call_with_rate_limit_retry(lambda: self._client.messages.create(**params))

    @staticmethod
    def _text(message) -> str:
        # thinking 등 비텍스트 블록이 앞에 올 수 있으므로 text 블록만 합친다.
        if message.stop_reason == "refusal":
            raise ModelRefusal(f"모델이 응답을 거절했습니다: {getattr(message, 'stop_details', None)}")
        return "".join(b.text for b in message.content if b.type == "text")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        self.usage.add(response)
        return self._text(response)

    async def generate_json(self, system_prompt: str, user_prompt: str, model_cls: type[BaseModel]) -> str:
        # messages.parse 는 거절처럼 본문이 비면 stop_reason 확인 전에 예외를 던져 거절을 재시도해 버리므로,
        # create 로 응답을 먼저 받고 직접 검증한다.
        from anthropic import transform_schema

        response = await self._create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": transform_schema(model_cls)}},
        )
        self.usage.add(response)
        return self._text(response)

    async def run_agent(self, system_prompt: str, user_prompt: str, tools: list, max_iterations: int) -> str:
        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            tools=[t.to_anthropic() for t in tools],
            messages=[{"role": "user", "content": user_prompt}],
            max_iterations=max_iterations,
        )
        last = None
        async for message in runner:
            self.usage.add(message)
            last = message
        if last is None:
            raise RuntimeError("에이전트가 응답을 반환하지 않았습니다")
        return self._text(last)


# ---------------------------------------------------------------------------
# 파사드
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        if self.provider == "openai":
            self._backend = _OpenAIBackend(settings)
        elif self.provider == "anthropic":
            self._backend = _AnthropicBackend(settings)
        else:
            raise ValueError(f"알 수 없는 LLM_PROVIDER: {self.provider!r} (openai | anthropic)")

    @property
    def mock_mode(self) -> bool:
        return self._backend.mock_mode

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def usage(self) -> Usage:
        """이 클라이언트가 지금까지 실제로 보낸 호출 수·토큰 (mock 모드에서는 0)."""
        return self._backend.usage

    @property
    def _client(self):
        return self._backend._client

    @_client.setter
    def _client(self, value) -> None:  # 테스트에서 가짜 서버(MockTransport)용 클라이언트를 주입한다
        self._backend._client = value

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt)
        return await self._backend.complete(system_prompt, user_prompt)

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
        *,
        mock: dict,
        max_attempts: int = 3,
    ) -> T:
        """구조화 출력: 응답을 pydantic 모델로 파싱·검증해 돌려준다.

        스키마를 강제해 요청하고, 응답은 우리가 직접 검증한다. 파싱/검증에 실패하면(잘림, 범위 밖 값 등)
        오류 내용을 알려주며 최대 max_attempts 회까지 다시 요청한다. 거절(refusal)은 재시도하지 않는다.
        mock 모드에서는 호출하지 않고 mock 값을 같은 모델로 검증해 돌려준다.
        """
        if self.mock_mode:
            return model_cls.model_validate(mock)

        prompt = user_prompt
        last_error: Exception | None = None
        for _ in range(max_attempts):
            try:
                text = await self._backend.generate_json(system_prompt, prompt, model_cls)
                return model_cls.model_validate_json(text)
            except ValueError as e:  # JSON 오류, pydantic ValidationError(범위/필수), IncompleteOutput 모두 ValueError 계열
                last_error = e
            # 다음 시도: 같은 요청에 직전 실패 사유를 덧붙인다
            prompt = (
                f"{user_prompt}\n\n[직전 응답 오류] {str(last_error)[:500]}\n"
                "지정된 JSON 스키마를 정확히 지켜 다시 답하세요 (값의 범위와 필수 필드 포함)."
            )
        raise StructuredOutputError(f"{max_attempts}회 시도에도 구조화 출력을 얻지 못함: {last_error}")

    async def run_agent(
        self, system_prompt: str, user_prompt: str, tools: list, max_iterations: int = 8
    ) -> str:
        """도구를 가진 에이전트 루프. 모델이 도구 호출을 멈출 때까지 반복한다."""
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt)
        return await self._backend.run_agent(system_prompt, user_prompt, tools, max_iterations)

    @staticmethod
    def _mock_response(system_prompt: str, user_prompt: str) -> str:
        role = system_prompt.splitlines()[0] if system_prompt else "agent"
        return (
            f"[MOCK 응답 / {role}] 아직 실제 프롬프트가 연결되지 않았습니다. "
            "API 키(OPENAI_API_KEY 또는 ANTHROPIC_API_KEY)를 설정하면 실제 LLM 응답으로 대체됩니다."
        )
