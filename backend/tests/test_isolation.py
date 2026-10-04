from app.config import get_settings
from app.core.llm_client import LLMClient


def test_tests_never_see_real_api_keys_or_dotenv():
    """tests/conftest.py 격리가 깨지면 mock 가정 테스트가 실제 API 를 호출해 비용이 나간다 — 그 전에 여기서 실패시킨다."""
    s = get_settings()
    assert s.openai_api_key == "" and s.anthropic_api_key == ""
    assert s.llm_mock_mode is True
    assert LLMClient().mock_mode is True
    assert s.chart_search_mode == "keyword"
