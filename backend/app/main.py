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
from app.core.sessions import AlreadyAnswered, SessionNotFound, SessionStore, TooManySessions
from app.core.user_flow import AssessmentService, InvalidOption
from app.models.schemas import AccidentInput, FaultReport
from app.models.user_schemas import AnswerRequest, AssessResponse

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
service = AssessmentService(pipeline, SessionStore())


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "llm_mock_mode": settings.llm_mock_mode,
    }


@app.post("/assess", response_model=FaultReport)
async def assess(accident_input: AccidentInput) -> FaultReport:
    """내부·개발용: FaultReport 전체를 돌려준다 (사용자용은 /v1/assess). INTERNAL_API_ENABLED=false 면 404."""
    if not settings.internal_api_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        return await pipeline.run(accident_input)
    except OutOfScopeError as e:
        # 지원하지 않는 사고 유형: 심사 대신 사유와 안내를 돌려준다
        raise HTTPException(
            status_code=422,
            detail={"code": "out_of_scope", "category": e.result.category, "message": e.result.message},
        )


# ---------------------------------------------------------------------------
# 사용자(사고 당사자)용 API — docs/USER_API_DESIGN.md
# ---------------------------------------------------------------------------
@app.post("/v1/assess", response_model=AssessResponse, response_model_exclude_none=True)
async def assess_for_user(accident_input: AccidentInput) -> AssessResponse:
    """사고 정황 → 결과, 또는 결과가 크게 갈리면 (결과를 숨긴) 질문."""
    try:
        return await service.start(accident_input)
    except OutOfScopeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "out_of_scope", "category": e.result.category, "message": e.result.message},
        )
    except TooManySessions:
        raise HTTPException(status_code=503, detail={"code": "busy", "message": "요청이 많습니다. 잠시 후 다시 시도해 주세요."})


@app.post("/v1/assess/{session_id}/answer", response_model=AssessResponse, response_model_exclude_none=True)
async def answer_for_user(session_id: str, body: AnswerRequest) -> AssessResponse:
    """질문에 대한 답변(보기 id, 모름은 -1/null)을 반영한 결과. 세션당 1회."""
    try:
        return await service.answer(session_id, body.option_id, body.seed)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail={"code": "session_not_found"})
    except AlreadyAnswered:
        raise HTTPException(status_code=409, detail={"code": "already_answered"})
    except InvalidOption:
        raise HTTPException(status_code=422, detail={"code": "invalid_option"})
