"""임베딩 검색 로직 테스트 — 실제 KURE-v1 모델을 내려받지 않고, 결정적인 가짜 인코더로 색인·캐시·결합 로직을 검증한다."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from app.core import chart_lookup
from app.core.chart_lookup import ChartCatalog, get_catalog
from app.core.embedding_index import EmbeddingIndex, EmbeddingUnavailable


def fake_encoder(texts):
    """문자 2-gram 을 64차원으로 해싱한 L2 정규화 벡터 (같은 어휘를 많이 공유할수록 유사)."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.replace(" ", "")
        for j in range(len(t) - 1):
            out[i, hash(t[j : j + 2]) % 64] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class CountingEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(len(texts))
        return fake_encoder(texts)


DOCS = {"차A": "후행 직진 차량과 선행 진로변경 차량 충돌", "차B": "주차장 통로 주행과 주차구획 출차 충돌", "차C": "회전교차로 진입 차량과 회전 차량 충돌"}


def test_index_ranks_most_similar_doc_first(tmp_path):
    idx = EmbeddingIndex(DOCS, encoder=fake_encoder, cache_dir=tmp_path)
    assert idx.search("주차장에서 출차하는 차와 통로 주행 차 충돌", 3)[0][0] == "차B"
    assert idx.search("   ") == []


def test_vectors_are_cached_and_invalidated_by_content(tmp_path):
    enc = CountingEncoder()
    EmbeddingIndex(DOCS, encoder=enc, cache_dir=tmp_path)
    EmbeddingIndex(DOCS, encoder=enc, cache_dir=tmp_path)
    assert enc.calls == [3]  # 두 번째 생성은 캐시를 읽어 인코딩하지 않는다
    EmbeddingIndex({**DOCS, "차C": "내용이 바뀐 도표"}, encoder=enc, cache_dir=tmp_path)
    assert enc.calls == [3, 3]  # 내용이 바뀌면 다시 계산
    assert len(list(tmp_path.glob("*.npy"))) == 2


def small_catalog(tmp_path):
    def chart(label, situation):
        return {"label": label, "category": "분류", "situation": situation,
                "variants": [{"vehicle_a": "", "vehicle_b": "", "base_ratio": {"a": 50, "b": 50}, "modifiers": []}]}
    cat = ChartCatalog({"차1": chart("진로변경 사고", "후행 직진 대 선행 진로변경"),
                        "차2": chart("주차장 출차", "통로 주행 대 주차구획 출차"),
                        "차3": chart("회전교차로", "진입 대 회전")})
    cat.attach_embedding(EmbeddingIndex(cat.index_texts, encoder=fake_encoder, cache_dir=tmp_path))
    return cat


def test_modes_return_chart_matches(tmp_path):
    cat = small_catalog(tmp_path)
    q = "주차구획에서 출차하는 차량과 통로 주행"
    assert cat.search(q, 1, mode="embedding")[0].chart_no == "차2"
    assert cat.search(q, 1, mode="hybrid")[0].chart_no == "차2"
    assert cat.search(q, 1)[0].chart_no == "차2"  # 기본은 keyword
    with pytest.raises(ValueError):
        cat.search(q, 1, mode="nonsense")


def test_hybrid_fuses_ranks_from_both_lists(tmp_path):
    cat = small_catalog(tmp_path)
    # 키워드 순위에는 있으나 임베딩에는 약한 문서도 결합 결과에 들어올 수 있어야 한다 (RRF)
    res = cat.search("진로변경 직진 선행", 3, mode="hybrid")
    assert {m.chart_no for m in res} <= {"차1", "차2", "차3"} and res[0].chart_no == "차1"
    assert all(a.score >= b.score for a, b in zip(res, res[1:]))


def test_falls_back_to_keyword_when_embedding_unavailable(tmp_path, monkeypatch, caplog):
    cat = small_catalog(tmp_path)
    cat.attach_embedding(None)

    def boom():
        raise EmbeddingUnavailable("테스트: 의존성 없음")

    monkeypatch.setattr(cat, "_embedding_index", boom)
    q = "주차구획에서 출차하는 차량과 통로 주행"
    with caplog.at_level("WARNING"):
        res = cat.search(q, 2, mode="embedding")
    assert [m.chart_no for m in res] == [m.chart_no for m in cat.search(q, 2)]
    assert "키워드 검색으로 대체" in caplog.text


def test_default_mode_is_keyword_without_embedding_dependency():
    from app.config import get_settings

    assert get_settings().chart_search_mode == "keyword"


def test_eval_script_rank_and_metrics():
    spec = importlib.util.spec_from_file_location("eval_chart_search", Path(__file__).parents[1] / "scripts" / "eval_chart_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.rank_of(["차A", "차B"], ["차X", "차B", "차A"]) == 2
    assert mod.rank_of(["차A"], ["차X"]) is None
    r = mod.evaluate(get_catalog(), [{"id": "t", "query": "정차한 앞차를 뒤차가 추돌", "expected": ["차41-1"]}], "keyword")
    assert r["n"] == 1 and r["top3"] == 1.0 and 0 < r["mrr"] <= 1


@pytest.mark.skipif(not os.getenv("RUN_KURE"), reason="실제 KURE-v1 모델 필요 (RUN_KURE=1 로 실행)")
def test_real_kure_hybrid_not_worse_than_keyword_on_top3():
    spec = importlib.util.spec_from_file_location("eval_chart_search", Path(__file__).parents[1] / "scripts" / "eval_chart_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import json

    items = json.loads((Path(__file__).parents[1] / "eval" / "chart_search_eval.json").read_text(encoding="utf-8"))["items"]
    cat = chart_lookup.get_catalog()
    kw = mod.evaluate(cat, items, "keyword")
    hy = mod.evaluate(cat, items, "hybrid")
    assert hy["top3"] >= kw["top3"] and hy["mrr"] >= kw["mrr"]
