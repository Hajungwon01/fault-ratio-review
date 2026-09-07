from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.pipeline import FaultAssessmentPipeline
from app.models.schemas import AccidentInput, FaultReport

settings = get_settings()
app = FastAPI(title=settings.app_name)

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
    return await pipeline.run(accident_input)
