"""도표 임베딩 검색 (로컬 KURE-v1).

도표 108개의 색인 텍스트(명칭·분류·사고 상황·A/B 설명)를 한 번 벡터화해 디스크에 캐시하고,
질의는 코사인 유사도(정규화 벡터의 내적)로 순위를 매긴다. 벡터 수가 적어 numpy 로 충분하다.

- 모델은 처음 쓸 때 지연 로드한다 (최초 다운로드 약 2GB대, 이후 로드 수 초). 서버 시작 시에는 로드하지 않는다.
- 캐시 키 = 모델 이름 + 색인 텍스트 해시. 기준표가 갱신되면 자동으로 다시 계산한다.
- sentence-transformers 가 설치돼 있지 않으면 EmbeddingUnavailable 을 던지고, 호출 측(chart_lookup)이 키워드 검색으로 폴백한다.
  설치: pip install -r requirements-embedding.txt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_MODEL = "nlpai-lab/KURE-v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"

Encoder = Callable[[list[str]], np.ndarray]  # 문자열 목록 -> (n, d) L2 정규화 벡터


class EmbeddingUnavailable(RuntimeError):
    """임베딩 의존성이 없거나 모델을 불러오지 못함."""


def _sentence_transformer_encoder(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise EmbeddingUnavailable(
            "sentence-transformers 가 설치돼 있지 않습니다 (pip install -r requirements-embedding.txt)"
        ) from e
    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False))

    return encode


class EmbeddingIndex:
    def __init__(
        self,
        docs: dict[str, str],
        *,
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        cache_dir: Path | None = CACHE_DIR,
    ) -> None:
        """docs: {chartNo: 색인 텍스트}. encoder 를 주입하면 모델을 로드하지 않는다 (테스트용)."""
        self.model_name = model_name
        self.keys = list(docs)
        self._encode = encoder or _sentence_transformer_encoder(model_name)
        self.vectors = self._load_or_build([docs[k] for k in self.keys], cache_dir)

    def _cache_path(self, texts: list[str], cache_dir: Path) -> Path:
        digest = hashlib.sha256(json.dumps([self.model_name, self.keys, texts], ensure_ascii=False).encode()).hexdigest()[:16]
        return cache_dir / f"chart_vectors_{self.model_name.replace('/', '_')}_{digest}.npy"

    def _load_or_build(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        if cache_dir is not None:
            path = self._cache_path(texts, cache_dir)
            if path.exists():
                return np.load(path)
        vectors = self._encode(texts).astype(np.float32)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors

    def search(self, query: str, limit: int = 3) -> list[tuple[str, float]]:
        """[(chartNo, 코사인 유사도)] 내림차순."""
        if not query.strip():
            return []
        q = self._encode([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[:limit]
        return [(self.keys[i], float(scores[i])) for i in order]
