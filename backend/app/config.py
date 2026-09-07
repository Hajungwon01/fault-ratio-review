from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 설정. .env 파일 또는 환경변수에서 읽는다.

    ANTHROPIC_API_KEY가 비어 있으면 LLMClient는 자동으로 mock 모드로 동작하므로,
    키가 없는 상태에서도 오케스트레이션 파이프라인 전체를 실행/테스트할 수 있다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "과실비율 산정 심사 자동화"
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5"
    port: int = 8000

    @property
    def llm_mock_mode(self) -> bool:
        return not self.anthropic_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
