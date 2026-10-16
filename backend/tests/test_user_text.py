"""사용자 설명 생성 — 도표 근거는 코드가 데이터에서, 모델 문장은 검증 후 사용(§9)."""
import pytest

from app.core.user_text import (
    MAX_SENTENCES,
    UNKNOWN_SUMMARY,
    chart_basis_entry,
    choose_summary,
    fallback_summary,
    unique_basis,
    validate_user_summary,
)


# ---------- 도표 근거 블록 (코드 생성) ----------
@pytest.mark.parametrize("chart,final,is_,expected_my", [
    ("차2-2", 0, "A", 0),        # 내가 도표 A(직진, 기본 0)
    ("차2-2", 100, "B", 100),    # 내가 도표 B → 뒤집은 기본 100
    ("차2-2", 95, None, 100),    # 역할을 모르면 두 방향 중 가까운 값
    ("차6-1", 80, "A", 80),      # 경우 (가)(나)(다) = 80/40/10 중 최종값에 가장 가까운 경우
    ("차6-1", 40, "A", 40),
    ("차6-1", 12, "A", 10),
])
def test_chart_basis_picks_the_base_closest_to_the_final_value(chart, final, is_, expected_my):
    b = chart_basis_entry(chart, final, is_)
    assert b.my_base == expected_my and b.other_base == 100 - expected_my
    assert b.label and b.situation and len(b.laws) <= 3


def test_chart_basis_unknown_chart_is_none_and_unique_basis_dedups():
    assert chart_basis_entry("차99-9", 10) is None
    a = chart_basis_entry("차2-2", 0, "A")
    assert unique_basis([a, None, chart_basis_entry("차2-2", 0, "A"), chart_basis_entry("차2-6", 10, "B")]) == [a, chart_basis_entry("차2-6", 10, "B")]


# ---------- 모델 문장 검증 ----------
B = [chart_basis_entry("차2-2", 5, "A")]       # 차2-2: 나 0 : 상대 100
GOOD = "고르신 답변을 기준으로 차2-2를 적용했습니다. 이 도표는 내 차량이 녹색 신호에 직진한 경우를 다루며, 나의 과실은 0~10%로 보았습니다."


def test_valid_summary_passes():
    assert validate_user_summary(GOOD, B, 0, 10, answered=True) == []
    assert validate_user_summary(GOOD.replace("차2-2", "차 2-2"), B, 0, 10, answered=True) == []        # 공백이 섞인 도표번호도 같은 번호로 본다


@pytest.mark.parametrize("text,expect", [
    ("", "비어 있음"),
    ("차2-6을 적용했습니다. 나의 과실은 10%입니다.", "허용되지 않은 도표번호"),
    ("나의 과실은 35%로 보았습니다.", "주어지지 않은 숫자"),
    ("위원 3명이 일치했습니다. 나의 과실은 10%입니다.", "내부 용어"),
    ("정황 A의 과실은 10%입니다.", "내부 용어"),
    ("확신이 높습니다.", "내부 용어"),
    ("상대 신호를 녹색으로 가정했습니다.", "내부 용어"),
    (" ".join(["문장입니다."] * (MAX_SENTENCES + 1)), f"{MAX_SENTENCES}문장 초과"),
])
def test_invalid_summaries_are_rejected(text, expect):
    assert any(expect in p for p in validate_user_summary(text, B, 0, 10))


def test_law_numbers_from_the_data_are_allowed_but_other_numbers_are_not():
    # 차2-2 의 관련 법규는 "도로교통법 제5조", "도로교통법 제25조" — 도표 데이터에 있는 번호라 숫자 검사에서 제외된다
    assert validate_user_summary("도로교통법 제25조(교차로 통행방법)가 관련됩니다. 나의 과실은 0~10%입니다.", B, 0, 10) == []
    assert any("주어지지 않은 숫자" in p for p in validate_user_summary("도로교통법 제44조가 관련됩니다.", B, 0, 10))


def test_choose_summary_uses_model_text_only_when_valid():
    assert choose_summary(GOOD, B, 0, 10, answered=True) == GOOD
    out = choose_summary("위원 3명이 일치했습니다.", B, 0, 10, answered=True)
    assert out.startswith("고르신 답변을 기준으로 차2-2") and "위원" not in out
    assert choose_summary("", B, 0, 10).startswith("이 사고는")


def test_fallback_summary_text_is_built_from_data_only():
    s = fallback_summary(B, 0, 10)
    assert B[0].situation in s and "나 0 : 상대 100" in s and "0~10%" in s
    assert "하나로 정하지 못해" in fallback_summary([], 20, 20) and "20%" in fallback_summary([], 20, 20)
    assert "정황" not in UNKNOWN_SUMMARY.replace("정황에 없는", "")                                  # 모름 문장에 내부 용어 없음


# ---------- 실측 D01·D03: 답하지 않았는데 "고르신 답변을 기준으로", 프롬프트 섹션 이름 노출 ----------
def test_claiming_an_answer_when_there_was_none_is_rejected():
    text = "고르신 답변을 기준으로 차2-2를 적용했습니다. 나의 과실은 0%로 보았습니다."
    assert any("답변하지 않았는데" in p for p in validate_user_summary(text, B, 0, 0))
    assert validate_user_summary(text, B, 0, 0, answered=True) == []
    assert any("답변하지 않았는데" in p for p in validate_user_summary("답변을 기준으로 봤습니다. 나의 과실은 0%입니다.", B, 0, 0))
    assert choose_summary(text, B, 0, 0).startswith("이 사고는")           # 질문이 없었으면 코드 문장으로 대체


def test_prompt_section_labels_leaking_into_the_text_are_rejected():
    for text in ("[적용한 기준표]에서는 직진 차량의 사고를 다룹니다. 나의 과실은 0%입니다.", "[최종 결론] 나의 과실은 0%입니다."):
        assert any("대괄호" in p for p in validate_user_summary(text, B, 0, 0))
    assert validate_user_summary("차2-2를 적용했고 나의 과실은 0%입니다 (신호 상태에 따라 달라질 수 있어요).", B, 0, 0) == []   # 소괄호는 허용


def test_other_and_unknown_summaries_state_their_limits():
    from app.core.user_text import OTHER_SUMMARY

    assert "보기에 없는 경우는 이 범위에 포함되지 않습니다" in UNKNOWN_SUMMARY
    assert "범위를 좁히지 못했습니다" in OTHER_SUMMARY and "증거" in OTHER_SUMMARY
