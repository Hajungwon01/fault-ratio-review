"""LLM 호출을 감싸는 얇은 래퍼.

API 키가 설정되지 않은 동안(mock 모드)에는 실제 호출 없이 고정된 형태의 응답을 돌려줘서,
프롬프트를 작성하기 전(3~4주차 이전)에도 오케스트레이션 파이프라인 전체를 실행/테스트할 수 있게 한다.

TODO(3~4주차): vehicle_a/vehicle_b 에이전트의 실제 시스템 프롬프트를 작성하면서
실제 API 호출 경로(mock_mode=False)를 함께 검증한다.
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

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt)

        # 실제 호출 경로 (키가 설정된 경우)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    @staticmethod
    def _mock_response(system_prompt: str, user_prompt: str) -> str:
        role = system_prompt.splitlines()[0] if system_prompt else "agent"
        return (
            f"[MOCK 응답 / {role}] 아직 실제 프롬프트가 연결되지 않았습니다. "
            "ANTHROPIC_API_KEY를 설정하면 실제 LLM 응답으로 대체됩니다."
        )
