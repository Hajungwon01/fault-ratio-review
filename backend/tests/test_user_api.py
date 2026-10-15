"""사용자용 API(/v1/assess, /v1/assess/{id}/answer) — 질문 단계의 결과 숨김, 세션 1회 답변, 만료, 내부 API 차단."""
import json
import re

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core.sessions import AlreadyAnswered, SessionNotFound, SessionStore, TooManySessions
from app.models.schemas import ClarifyingQuestion, FaultReport, QuestionOption, ReviewScore

DESC = {"description": "녹색 신호에 직진하는데 맞은편 차량이 좌회전하며 충돌했습니다"}
SECRETS = ("scene", "fault", "chart_no", "evidence", "flag", "reviews", "rounds", "needs_precise", "clarifications")


def review(name, fault, chart="차2-2", conf="high"):
    return ReviewScore(reviewer_name=name, perspective=name, fault_ratio_a=fault, fault_ratio_b=100 - fault,
                       rationale=f"{name} 근거", chart_no=chart, chart_a_is="A", scene_a_fault=fault, confidence=conf)


def question():
    return ClarifyingQuestion(
        question="충돌 직전 상대 방향 신호는 무엇이었나요?",
        options=[
            QuestionOption(label="녹색 신호였다", chart_no="차2-6", scene_a_fault=10),
            QuestionOption(label="적색 신호였다", chart_no="차2-2", scene_a_fault=0),
            QuestionOption(label="점멸 신호였다", chart_no="차2-5", scene_a_fault=50),
        ],
        evidence="블랙박스 영상, 신호 주기 기록",
    )


def report(with_question=True, faults=(10, 20, 10), note=""):
    return FaultReport(
        fault_ratio_a=faults[1], fault_ratio_b=100 - faults[1], basis="요약", deviation=4.7, spread=max(faults) - min(faults),
        flag_reasons=["위원 간 차이"], needs_precise_review=with_question,
        clarifications=[question()] if with_question else [], needs_clarification=with_question,
        clarification_note=note, reviews=[review("기준표 적용", faults[0]), review("현장 정황", faults[1]), review("유사사례", faults[2])],
    )


@pytest.fixture
def client(monkeypatch):
    """파이프라인은 가짜 리포트를 돌려주고, 세션 저장소는 테스트마다 새로 만든다."""
    holder = {"report": report()}

    async def fake_run(accident_input):
        return holder["report"]

    monkeypatch.setattr(main.pipeline, "run", fake_run)
    store = SessionStore()
    monkeypatch.setattr(main.service, "store", store)
    c = TestClient(main.app)
    c.holder, c.store = holder, store
    return c


def start(client):
    return client.post("/v1/assess", json=DESC)


def forbidden_in(text):
    return [s for s in SECRETS if s in text] + re.findall(r"차\d+-\d+", text)


# ---------- 질문 단계: 결과 숨김 ----------
def test_question_response_hides_charts_faults_and_internal_fields(client):
    r = start(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_answer" and body["session_id"] and body["expires_in_sec"] == 1800
    assert set(body) == {"status", "session_id", "expires_in_sec", "question", "notice"}      # result 등은 아직 없다
    assert set(body["question"]) == {"seed", "text", "options"}
    assert all(set(o) == {"id", "label"} for o in body["question"]["options"])
    assert forbidden_in(json.dumps(body, ensure_ascii=False)) == []
    labels = [o["label"] for o in body["question"]["options"]]
    assert labels[-1].startswith("모름") and body["question"]["options"][-1]["id"] == -1
    assert set(labels[:-1]) == {"녹색 신호였다", "적색 신호였다", "점멸 신호였다"}


def test_without_question_result_is_returned_with_range_and_disclaimer(client):
    client.holder["report"] = report(with_question=False, faults=(0, 10, 10), note="다른 도표도 가능하나 결과 차이는 작습니다")
    body = start(client).json()
    assert body["status"] == "done" and "session_id" not in body and "question" not in body
    res = body["result"]
    assert (res["my_fault_low"], res["my_fault_high"]) == (0, 10) and (res["other_fault_low"], res["other_fault_high"]) == (90, 100)
    assert res["assumption"] == "" and res["evidence_to_check"] == []
    assert res["other_possibilities"].startswith("다른 도표도") and "참고용" in res["disclaimer"] and "법적 효력" in res["disclaimer"]
    assert res["charts"] and res["charts"][0]["chart_no"] == "차2-2"
    assert set(body["result"]) == {"my_fault_low", "my_fault_high", "other_fault_low", "other_fault_high", "assumption", "explanation",
                                   "charts", "evidence_to_check", "other_possibilities", "uncertain", "disclaimer"}
    assert "flag_reasons" not in json.dumps(body) and "reviews" not in body["result"]


# ---------- 답변 ----------
def answer(client, body, option_label, **kw):
    q = body["question"]
    opt = next(o for o in q["options"] if o["label"] == option_label)
    return client.post(f"/v1/assess/{body['session_id']}/answer", json={"option_id": opt["id"], "seed": q["seed"]}, **kw)


@pytest.mark.parametrize("label,expected,chart", [("녹색 신호였다", 10, "차2-6"), ("적색 신호였다", 0, "차2-2")])
def test_answer_uses_selected_option_and_marks_it_as_a_statement(client, label, expected, chart):
    body = start(client).json()
    r = answer(client, body, label)
    assert r.status_code == 200
    res = r.json()["result"]
    assert (res["my_fault_low"], res["my_fault_high"]) == (expected, expected)
    assert "사용자 진술 기준" in res["assumption"] and label in res["assumption"]
    assert res["evidence_to_check"] == ["블랙박스 영상, 신호 주기 기록"]                    # 증거 안내는 답변 뒤에 공개
    assert [c["chart_no"] for c in res["charts"]] == [chart]


@pytest.mark.parametrize("option_id", [-1, None])
def test_unknown_answer_returns_range_over_all_options(client, option_id):
    body = start(client).json()
    r = client.post(f"/v1/assess/{body['session_id']}/answer", json={"option_id": option_id, "seed": body["question"]["seed"]})
    res = r.json()["result"]
    assert (res["my_fault_low"], res["my_fault_high"]) == (0, 50) and "확실하지 않다고" in res["assumption"]


def test_session_accepts_only_one_answer(client):
    body = start(client).json()
    assert answer(client, body, "녹색 신호였다").status_code == 200
    second = answer(client, body, "적색 신호였다")                       # 보기를 바꿔 결과를 비교하려는 시도
    assert second.status_code == 409 and second.json()["detail"]["code"] == "already_answered"


def test_unknown_or_expired_session_is_404_and_internal_report_is_dropped_after_answer(client):
    r = client.post("/v1/assess/없는세션/answer", json={"option_id": 0, "seed": 1})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "session_not_found"
    body = start(client).json()
    answer(client, body, "녹색 신호였다")
    assert client.store._sessions[body["session_id"]].report is None   # 답한 뒤에는 내부 결과를 보관하지 않는다


@pytest.mark.parametrize("option_id,seed_delta", [(9, 0), (-5, 0), (0, 1)])
def test_invalid_option_or_wrong_seed_is_422_and_session_stays_open(client, option_id, seed_delta):
    body = start(client).json()
    r = client.post(f"/v1/assess/{body['session_id']}/answer", json={"option_id": option_id, "seed": body["question"]["seed"] + seed_delta})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_option"
    assert answer(client, body, "녹색 신호였다").status_code == 200       # 잘못된 시도는 기회를 소모하지 않는다


def test_each_session_gets_a_different_seed_and_token(client):
    a, b = start(client).json(), start(client).json()
    assert a["session_id"] != b["session_id"] and len(a["session_id"]) >= 24


# ---------- 범위 밖 / 내부 API ----------
def test_out_of_scope_is_422(monkeypatch):
    c = TestClient(main.app)
    r = c.post("/v1/assess", json={"description": "횡단보도를 건너던 보행자를 제 차로 치었습니다 급하게 브레이크를 밟았지만"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "out_of_scope"


def test_mock_pipeline_end_to_end_returns_a_result_without_question():
    r = TestClient(main.app).post("/v1/assess", json=DESC)
    assert r.status_code == 200 and r.json()["status"] == "done" and "참고용" in r.json()["result"]["disclaimer"]


def test_internal_assess_can_be_disabled(monkeypatch):
    c = TestClient(main.app)
    assert c.post("/assess", json=DESC).status_code == 200
    monkeypatch.setattr(main.settings, "internal_api_enabled", False)
    assert c.post("/assess", json=DESC).status_code == 404
    assert c.post("/v1/assess", json=DESC).status_code == 200            # 사용자용은 영향 없음


def test_internal_assess_still_returns_the_full_report_for_development(client):
    body = client.post("/assess", json=DESC).json()
    assert body["clarifications"] and body["reviews"]                    # 개발·평가용이라 보기별 도표·과실이 보인다


# ---------- 세션 저장소 ----------
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_session_expires_after_ttl():
    clock = Clock()
    store = SessionStore(ttl_sec=60, clock=clock)
    sid = store.create(report(), question(), 1)
    clock.t = 59
    assert store.get(sid)
    clock.t = 61
    with pytest.raises(SessionNotFound):
        store.get(sid)
    assert len(store) == 0


def test_answered_session_is_marked_until_it_expires():
    clock = Clock()
    store = SessionStore(ttl_sec=60, clock=clock)
    sid = store.create(report(), question(), 1)
    store.mark_answered(sid)
    with pytest.raises(AlreadyAnswered):
        store.get(sid)
    clock.t = 61
    with pytest.raises(SessionNotFound):
        store.get(sid)


def test_store_limits_concurrent_sessions_and_frees_expired_ones():
    clock = Clock()
    store = SessionStore(ttl_sec=60, max_sessions=2, clock=clock)
    store.create(report(), question(), 1)
    store.create(report(), question(), 2)
    with pytest.raises(TooManySessions):
        store.create(report(), question(), 3)
    clock.t = 61                                                         # 만료되면 자리가 난다
    assert store.create(report(), question(), 4)


def test_too_many_sessions_is_503(client, monkeypatch):
    monkeypatch.setattr(main.service, "store", SessionStore(max_sessions=1))
    assert start(client).status_code == 200
    r = start(client)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "busy"
