"""테스트 격리: 실제 API 키가 든 backend/.env 를 테스트가 절대 읽지 않게 한다.

이 파일은 테스트 모듈들이 app 을 import 하기 전에 먼저 실행되어야 하므로 모듈 최상단에서 설정한다.
(.env 에 실제 키가 있으면 mock 을 가정한 파이프라인 테스트가 실제 API 를 호출해 비용이 나가고 느려진다.)
실제 API 를 쓰는 확인은 scripts/verify_llm.py 로만 한다.
"""
import os

import pytest

# 환경변수가 .env 파일보다 우선하므로 빈 값/기본값으로 고정한다
for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT"):
    os.environ[_key] = ""
os.environ["LLM_PROVIDER"] = "openai"
os.environ["CHART_SEARCH_MODE"] = "keyword"

from app.config import Settings, get_settings  # noqa: E402

# 테스트 중에는 .env 파일 자체를 읽지 않는다 (monkeypatch.delenv 로 환경변수를 지워도 .env 값이 되살아나지 않게)
Settings.model_config["env_file"] = None
get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
