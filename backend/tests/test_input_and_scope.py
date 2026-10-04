import asyncio

import pytest
from pydantic import ValidationError

from app.core.pipeline import FaultAssessmentPipeline
from app.core.scope_check import OutOfScopeError, check_scope
from app.models.schemas import AccidentInput, SignalState


def acc(desc, **kw):
    return AccidentInput(description=desc, **kw)


# ---------- 입력 스키마 ----------
def test_only_description_is_required():
    a = acc("제가(A) 1차선 직진 중 상대 차량(B)이 갑자기 차선을 변경해 충돌했습니다")
    assert a.accident_type is None and a.signal_state is SignalState.UNKNOWN


def test_description_too_short_rejected():
    with pytest.raises(ValidationError):
        acc("추돌")


def test_speed_range_validated():
    with pytest.raises(ValidationError):
        acc("제가(A) 직진 중 상대 차량(B)과 충돌했습니다", speed_a_kmh=-5)
    with pytest.raises(ValidationError):
        acc("제가(A) 직진 중 상대 차량(B)과 충돌했습니다", speed_b_kmh=900)


def test_prompt_text_is_readable_and_omits_empty_fields():
    t = acc("제가(A) 직진 중 상대 차량(B)과 충돌했습니다").to_prompt_text()
    assert t == "사고 상황 설명: 제가(A) 직진 중 상대 차량(B)과 충돌했습니다"
    t = acc("제가(A) 직진 중 상대 차량(B)과 충돌했습니다", signal_state=SignalState.A_GREEN_B_RED,
            lane_info="2차로", speed_a_kmh=50, speed_b_kmh=32.5).to_prompt_text()
    assert "신호 상태: A 녹색 / B 적색" in t and "A차량 추정 속도: 50km/h" in t and "32.5km/h" in t
    # 파이썬 repr 이 새지 않는다
    assert "SignalState" not in t and "None" not in t and "a_green" not in t


# ---------- 범위 판별 ----------
@pytest.mark.parametrize("desc", [
    "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다",
    "정차 중 문을 열어 뒤에서 오던 오토바이가 부딪혔습니다",          # 이륜차는 지원
    "구급차가 적색 신호에 직진하다 제 차와 충돌했습니다",              # 긴급자동차는 지원
    "주차장 통로를 지나는데 주차칸에서 후진으로 나오던 차와 부딪혔습니다",
])
def test_supported(desc):
    assert check_scope(acc(desc)).status == "supported"


@pytest.mark.parametrize("desc,category", [
    ("횡단보도를 건너던 보행자를 제 차가 쳤습니다", "보행자"),
    ("자전거가 갑자기 도로로 나와서 부딪혔어요", "자전거"),
    ("전동 킥보드가 신호를 무시하고 달려와 충돌했습니다", "개인형 이동장치"),
    ("PM 이 인도에서 갑자기 튀어나왔습니다 피하지 못했어요", "개인형 이동장치"),
    ("가드레일을 혼자 들이받은 단독 사고입니다 상대 차량은 없어요", "단독사고"),
])
def test_unsupported(desc, category):
    r = check_scope(acc(desc))
    assert r.status == "unsupported" and r.category == category and r.message


def test_mention_with_counterpart_vehicle_is_uncertain_not_rejected():
    r = check_scope(acc("보행자를 피하려다 상대 차량(B)과 측면 충돌했습니다"))
    assert r.status == "uncertain" and "보행자" in r.message


def test_single_accident_not_softened_by_counterpart_phrase():
    # '상대 차량은 없다'라는 표현이 상대 차량 언급으로 오인되면 안 된다
    assert check_scope(acc("혼자 전봇대를 박은 단독 사고입니다 상대 차량은 없습니다")).status == "unsupported"


# ---------- 파이프라인/API ----------
class SpyLLM:
    mock_mode = True
    calls = 0

    async def complete(self, *a, **k):
        SpyLLM.calls += 1
        return "x"

    async def run_agent(self, *a, **k):
        SpyLLM.calls += 1
        return "x"


def test_pipeline_rejects_out_of_scope_before_any_llm_call(monkeypatch):
    SpyLLM.calls = 0
    monkeypatch.setattr("app.core.pipeline.LLMClient", SpyLLM)
    p = FaultAssessmentPipeline()
    with pytest.raises(OutOfScopeError) as e:
        asyncio.run(p.run(acc("횡단보도를 건너던 보행자를 제 차가 쳤습니다")))
    assert e.value.result.category == "보행자" and SpyLLM.calls == 0


def test_pipeline_uncertain_adds_warning_and_runs():
    r = asyncio.run(FaultAssessmentPipeline().run(acc("보행자를 피하려다 상대 차량(B)과 측면 충돌했습니다")))
    assert r.warnings and "보행자" in r.warnings[0] and len(r.rounds) == 4


def test_pipeline_supported_has_no_warnings():
    r = asyncio.run(FaultAssessmentPipeline().run(acc("녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다")))
    assert r.warnings == []


def test_assess_endpoint_returns_422_detail_for_out_of_scope():
    from fastapi import HTTPException

    from app.main import assess

    with pytest.raises(HTTPException) as e:
        asyncio.run(assess(acc("자전거가 갑자기 도로로 나와서 부딪혔어요")))
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "out_of_scope" and e.value.detail["category"] == "자전거"
