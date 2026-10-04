from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 설정. .env 파일 또는 환경변수에서 읽는다.

    선택한 프로바이더(LLM_PROVIDER)의 API 키가 비어 있으면 LLMClient는 자동으로 mock 모드로 동작하므로,
    키가 없는 상태에서도 오케스트레이션 파이프라인 전체를 실행/테스트할 수 있다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "과실비율 산정 심사 자동화"
    port: int = 8000

    # LLM 프로바이더: openai | anthropic
    llm_provider: str = "openai"

    # OpenAI — OPENAI_MODEL 은 모델 ID 를 직접 지정한다 (코드에 고정하지 않음. 키가 있으면 scripts/check_openai.py 로 사용 가능 여부 확인)
    openai_api_key: str = ""
    openai_model: str = ""
    openai_reasoning_effort: str = ""  # 비우면 모델 기본값. 추론 모델의 비용·지연 조절용 (예: low | medium | high)

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"  # LLM_PROVIDER=anthropic 일 때의 모델. 비용을 낮추려면 claude-sonnet-5-5

    # 도표 검색 방식: keyword | embedding | hybrid (embedding/hybrid 는 requirements-embedding.txt 필요)
    chart_search_mode: str = "keyword"

    @property
    def llm_mock_mode(self) -> bool:
        key = self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key
        return not key


@lru_cache
def get_settings() -> Settings:
    return Settings()
