from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.chart_lookup import get_catalog
from app.core.embedding_index import EmbeddingUnavailable
from app.core.pipeline import FaultAssessmentPipeline
from app.core.scope_check import OutOfScopeError
from app.models.schemas import AccidentInput, FaultReport

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 임베딩 검색을 쓰는 설정이면 모델을 서버 시작 시 미리 올려, 첫 /assess 요청이 수 초 멈추지 않게 한다.
    # 불러오지 못하면 chart_lookup 이 요청마다 키워드 검색으로 폴백한다.
    if settings.chart_search_mode != "keyword":
        try:
            await asyncio.to_thread(get_catalog()._embedding_index)
            logger.info("도표 임베딩 색인 준비 완료 (mode=%s)", settings.chart_search_mode)
        except EmbeddingUnavailable as e:
            logger.warning("임베딩 색인을 준비하지 못했습니다. 키워드 검색으로 동작합니다: %s", e)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# 프론트엔드(index.html)를 file:// 또는 다른 포트에서 열어도 붙을 수 있도록 전부 허용.
# TODO: 배포 시 실제 프론트엔드 도메인으로 제한.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = FaultAssessmentPipeline()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_mock_mode": settings.llm_mock_mode,
    }


@app.post("/assess", response_model=FaultReport)
async def assess(accident_input: AccidentInput) -> FaultReport:
    try:
        return await pipeline.run(accident_input)
    except OutOfScopeError as e:
        # 지원하지 않는 사고 유형: 심사 대신 사유와 안내를 돌려준다
        raise HTTPException(
            status_code=422,
            detail={"code": "out_of_scope", "category": e.result.category, "message": e.result.message},
        )
