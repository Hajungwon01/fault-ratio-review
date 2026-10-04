"""LLM 호출을 감싸는 얇은 래퍼.

API 키가 설정되지 않은 동안(mock 모드)에는 실제 호출 없이 고정된 형태의 응답을 돌려줘서,
프롬프트를 작성하기 전(3~4주차 이전)에도 오케스트레이션 파이프라인 전체를 실행/테스트할 수 있게 한다.

- complete(): 도구 없는 단발 호출 (검토위원 등)
- run_agent(): Tool Runner 루프. 모델이 도구(기준표 조회 등)를 스스로 호출하며 결론까지 진행한다.

TODO: 실제 API 호출 경로(mock_mode=False)는 ANTHROPIC_API_KEY 설정 후 검증 필요 (아직 mock 경로만 검증됨).
"""
from __future__ import annotations

from app.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.mock_mode = settings.llm_mock_mode
        self.model = settings.llm_model
        self._client = None
        if not self.mock_mode:
            # anthropic 패키지는 requirements.txt에 포함. 키가 있을 때만 지연 임포트.
            import anthropic

            # 파이프라인이 asyncio.gather 로 에이전트를 병렬 실행하므로 반드시 async 클라이언트를 쓴다.
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt)

        # 실제 호출 경로 (키가 설정된 경우)
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return self._text(response)

    async def run_agent(
        self, system_prompt: str, user_prompt: str, tools: list, max_iterations: int = 8
    ) -> str:
        """도구를 가진 에이전트 루프. 모델이 tool_use 를 멈출 때까지 SDK Tool Runner 가 반복한다."""
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt)

        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": user_prompt}],
            max_iterations=max_iterations,
        )
        last = None
        async for message in runner:
            last = message
        if last is None:
            raise RuntimeError("에이전트가 응답을 반환하지 않았습니다")
        return self._text(last)

    @staticmethod
    def _text(message) -> str:
        # thinking 등 비텍스트 블록이 앞에 올 수 있으므로 text 블록만 합친다.
        text = "".join(b.text for b in message.content if b.type == "text")
        if message.stop_reason == "refusal":
            raise RuntimeError(f"모델이 응답을 거절했습니다: {getattr(message, 'stop_details', None)}")
        return text

    @staticmethod
    def _mock_response(system_prompt: str, user_prompt: str) -> str:
        role = system_prompt.splitlines()[0] if system_prompt else "agent"
        return (
            f"[MOCK 응답 / {role}] 아직 실제 프롬프트가 연결되지 않았습니다. "
            "ANTHROPIC_API_KEY를 설정하면 실제 LLM 응답으로 대체됩니다."
        )
