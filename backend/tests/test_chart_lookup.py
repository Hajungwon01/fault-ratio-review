import asyncio

from app.agents.base import extract_citations
from app.agents.vehicle_a import VehicleAAgent
from app.core.chart_lookup import find_candidate_charts, get_catalog
from app.core.pipeline import FaultAssessmentPipeline
from app.models.schemas import AccidentInput, AccidentType


class RecordingLLM:
    mock_mode = True

    def __init__(self):
        self.calls = []

    async def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "차43-2 적용. 인용: 차43-2, 도로교통법 제19조"


def test_catalog_loaded_and_ratios_sum_to_100():
    charts = get_catalog().charts
    assert len(charts) >= 100
    for c in charts.values():
        for v in c["variants"]:
            assert v["base_ratio"]["a"] + v["base_ratio"]["b"] == 100


# 입력 → 상위 3개 후보 안에 기대 도표가 있어야 한다 (1위 정확도는 보장하지 않음)
CASES = [
    (AccidentType.LANE_CHANGE, "직진 중인데 옆 차로 차량이 방향지시등 없이 갑자기 끼어들어 충돌", "차43-2"),
    (AccidentType.REAR_END_COLLISION, "신호대기로 정차한 앞차를 뒤차가 추돌", "차41-1"),
    (AccidentType.STRAIGHT_VS_LEFT_TURN, "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌", "차2-2"),
    (AccidentType.OTHER, "주차장 통로를 지나가는데 주차칸에서 후진으로 나오던 차와 부딪힘", "차51-1"),
    (AccidentType.OTHER, "정차 중 운전자가 문을 열어 뒤에서 오던 차가 부딪힘", "차52-1"),
]


def test_candidate_charts_contain_expected():
    for ty, desc, expected in CASES:
        got = [m.chart_no for m in find_candidate_charts(AccidentInput(accident_type=ty, description=desc), 3)]
        assert expected in got, (desc, got)


def test_reference_charts_injected_into_advocate_prompt():
    llm = RecordingLLM()
    agent = VehicleAAgent(llm)
    ty, desc, _ = CASES[0]
    from app.core.chart_lookup import format_candidates

    ref = format_candidates(find_candidate_charts(AccidentInput(accident_type=ty, description=desc)))
    arg = asyncio.run(agent.run({"accident_input": desc, "reference_charts": ref, "round": 1}))
    system, user = llm.calls[0]
    assert "차43-2" in user and "가감요소" in user
    assert arg.cited_rules == ["차43-2", "도로교통법 제19조"]


def test_extract_citations_dedup():
    assert extract_citations("차12-1 그리고 차12-1, 도로교통법 제26조") == ["차12-1", "도로교통법 제26조"]


def test_pipeline_runs_end_to_end_mock():
    ty, desc, _ = CASES[0]
    report = asyncio.run(FaultAssessmentPipeline().run(AccidentInput(accident_type=ty, description=desc)))
    assert len(report.rounds) == 4
