"""프로바이더 중립 에이전트 도구.

도구를 한 번만 정의하고, LLM 프로바이더별 형식(OpenAI Responses 함수 도구 / Anthropic Tool Runner 도구)으로 변환한다.
도구 함수는 키워드 인자를 받는 async 함수이고 결과는 문자열이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema (type=object)
    func: Callable[..., Awaitable[str]]

    async def call(self, args: dict[str, Any]) -> str:
        return await self.func(**args)

    def to_openai(self) -> dict[str, Any]:
        """OpenAI Responses API 의 function 도구 정의."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_anthropic(self):
        """Anthropic Tool Runner 가 받는 도구 객체 (anthropic 은 이 변환이 필요할 때만 import)."""
        from anthropic.lib.tools import BetaAsyncFunctionTool

        return BetaAsyncFunctionTool(
            self.func, name=self.name, description=self.description, input_schema=self.input_schema
        )
