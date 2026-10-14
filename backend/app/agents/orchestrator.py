"""진행 에이전트(오케스트레이터) — 라운드 진행 관리 + 위원 의견 취합 + 심사 리포트 생성.

이 에이전트 자체는 "판정"을 내리지 않고, 검토위원 3인의 결과를 집계해
과실비율 중앙값과 편차를 계산하고, 편차가 크면 정밀 심사가 필요하다고 플래그만 남긴다.
실제 라운드 순서 제어(누구를 언제 부를지)는 core/pipeline.py 가 담당하고,
이 클래스는 "이미 다 끝난 결과를 받아서 리포트로 정리하는" 마지막 단계만 맡는다.

TODO(8주차): 편차 계산 방식(표준편차 vs 최대-최소 등)과 임계값을 확정.
"""
from __future__ import annotations

import statistics
from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import FaultReport

# "정밀 심사 필요" 판정 기준 — 둘 중 하나라도 넘으면 플래그. 둘 다 평가셋 실측 후 조정할 초기값이다 (#13).
#
# 표준편차만으로는 위원이 3명일 때 이상치 한 명을 놓친다: 한 명이 나머지와 d점 벗어나면 표준편차는 약 0.47*d 라서
# 임계값 15 는 d >= 32점이어야 켜진다. 실제 첫 측정에서 위원 A과실이 30/0/0 (30점 차)인데 편차 14.1 로 플래그가 안 켜졌다.
# 그래서 사람이 직관적으로 이해하는 "위원 간 최대-최소 차이(spread)"를 함께 쓴다.
SPREAD_THRESHOLD = 20      # 위원 A과실의 최대-최소 차이(%p)
DEVIATION_THRESHOLD = 15.0  # 위원 A과실의 모집단 표준편차
# 위원이 숫자는 일치해도 스스로 "확신 낮음"이라고 답한 경우: 애매한 사례(예: 도표가 둘 다 가능한 D09)는 위원 숫자가 우연히 일치하면
# 위 두 기준으로는 드러나지 않았다. 몇 명 이상이 low 일 때 켤지는 초기값이며 평가셋 실측 후 조정한다 (#13).
# (위원 평가가 도표 기본 비율과 어긋나 신뢰도가 low 로 내려간 경우는 pipeline 이 별도로 플래그를 켠다)
LOW_CONFIDENCE_REVIEWERS = 2


class ProceedingAgent(BaseAgent):
    name = "proceeding_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 검토위원 3인의 평가를 취합해 최종 과실비율 심사 리포트를 작성하는 진행 에이전트입니다.\n"
            "직접 판정을 내리지 않고, 위원들의 평가를 집계·요약하는 역할만 합니다."
        )

    async def run(self, context: dict[str, Any]) -> FaultReport:
        reviews = context["reviews"]  # list[ReviewScore]
        ratios_a = [r.fault_ratio_a for r in reviews]

        # 위원이 3명이면 한 명의 이탈값(역할 오류 등)에 평균은 크게 끌려가지만 중앙값은 영향을 받지 않는다. 2명이면 평균과 같다.
        avg_a = round(statistics.median(ratios_a))
        avg_b = 100 - avg_a
        deviation = round(statistics.pstdev(ratios_a), 1) if len(ratios_a) > 1 else 0.0
        spread = (max(ratios_a) - min(ratios_a)) if ratios_a else 0

        basis = await self.llm.complete(
            self.system_prompt,
            f"검토위원 평가: {reviews}\n중앙값 A과실: {avg_a}, 편차: {deviation}",
        )

        reasons: list[str] = []
        if spread >= SPREAD_THRESHOLD:
            reasons.append(f"위원 간 과실비율 차이가 큼 (최대-최소 {spread}%p)")
        if deviation > DEVIATION_THRESHOLD:
            reasons.append(f"위원 간 과실비율 편차가 큼 (표준편차 {deviation})")
        # 위원이 서로 다른 도표를 적용했다 = 정황으로 도표가 하나로 정해지지 않는 애매한 사고. 숫자가 우연히 일치해도 알린다.
        # (저장된 실행 64건 중 불일치는 애매 사례 D09 3건뿐이고 명백한 사례에서는 0건이었다)
        charts = sorted({r.chart_no for r in reviews if r.chart_no})
        if len(charts) > 1:
            reasons.append(f"위원이 서로 다른 도표를 적용함 ({', '.join(charts)})")
        low = [r.perspective for r in reviews if r.confidence == "low"]
        if len(low) >= LOW_CONFIDENCE_REVIEWERS:
            reasons.append(f"위원 {len(low)}명이 확신 낮음으로 답함 ({', '.join(low)})")

        return FaultReport(
            fault_ratio_a=avg_a,
            fault_ratio_b=avg_b,
            basis=basis,
            deviation=deviation,
            spread=spread,
            flag_reasons=reasons,
            needs_precise_review=bool(reasons),
            rounds=context.get("rounds", []),
            fact_check=context.get("fact_check"),
            reviews=reviews,
        )


# ---------------------------------------------------------------------------
# 질문 단계 (사고 당사자용): 정황에 없는 사실 때문에 결과가 갈리면 사용자에게 물을 질문 1개를 만든다
# ---------------------------------------------------------------------------
CLARIFY_MIN_DIFF = 10  # 보기별 정황 A 과실의 최대-최소가 이를 넘을 때만 사용자에게 묻는다 (이하는 참고 문구만)

CLARIFY_PROMPT = (
    "당신은 과실비율 검토에서 '사고 정황에 적혀 있지 않은 사실' 때문에 결과가 달라지는 지점을 찾아, 사용자에게 물을 질문 하나를 만드는 진행 에이전트입니다.\n\n"
    "[질문이 필요한 경우]\n"
    f"- 정황에 없는 사실 하나에 따라 적용 도표가 달라지거나, 같은 도표 안에서도 정황 A 과실이 {CLARIFY_MIN_DIFF}%p 넘게 달라질 때만 만드세요.\n"
    "- 정황으로 이미 정해지거나, 사실에 따라 달라져도 차이가 작으면 만들지 마세요 (questions 는 빈 목록).\n"
    "- 정황에 서술된 상황을 뒤집는 가능성(예: 일반 도로라고 읽히는데 합류 구간일 수도 있다)을 상상해 만들지 마세요. "
    "위원들이 서로 다른 도표를 골랐거나, 근거에서 \"정황에 없어 불확실\"이라고 쓴 사실만 질문으로 만드세요.\n"
    "- 가장 영향이 큰 질문 1개만 만드세요. 위원들의 결론과 근거를 참고해 어떤 사실이 비워져 있는지 찾으세요.\n\n"
    "[질문 작성 규칙 — 사용자가 유리한 답을 고르지 못하게]\n"
    "- 관찰하거나 기억할 수 있는 사실을 묻습니다 (신호 색, 표지 유무, 진입 순서, 속도 구간 등). "
    "\"신호위반이었나요?\"처럼 법적 결론을 묻거나 용어로 결과를 암시하지 마세요.\n"
    "- 보기(label)는 사실만 적고 과실비율·도표번호·\"유리/불리\" 같은 암시를 쓰지 마세요. 보기 길이와 어조를 비슷하게 하세요.\n"
    "- 보기는 서로 겹치지 않게 2~4개. \"모름\"은 시스템이 추가하므로 쓰지 마세요.\n"
    "- 보기마다 그 답이 맞을 때의 도표번호(chart_no)와 결과를 적습니다. 도표번호는 [인용 도표 상세]에 있는 것만 쓰세요.\n"
    "- scene_a_fault: 그 보기가 맞을 때 **정황의 A(나의 차량)** 입장의 최종 과실비율 (도표의 A 기준이 아니라 정황 기준). "
    "나의 차량이 도표의 A 인지 B 인지 먼저 대응시킨 뒤 계산하세요.\n"
    "- evidence: 그 사실을 확인할 수 있는 증거 (블랙박스, 신호 주기 기록, 경찰 조사 등)를 적습니다.\n"
    "- 주어진 [인용 도표 상세] 밖의 도표번호·비율을 만들어 쓰지 마세요."
)


def ambiguity_signals(reviews) -> list[str]:
    """질문 생성이 필요할 수 있다는 신호: 위원이 서로 다른 도표를 고름 / 확신이 high 가 아닌 위원 / 위원 간 차이 / 평가 경고."""
    signals = []
    if len({r.chart_no for r in reviews if r.chart_no}) > 1:
        signals.append("위원들이 서로 다른 도표를 적용함")
    if any(r.confidence != "high" for r in reviews):
        signals.append("확신이 high 가 아닌 위원이 있음")
    faults = [r.fault_ratio_a for r in reviews]
    if faults and max(faults) - min(faults) >= CLARIFY_MIN_DIFF:
        signals.append("위원 간 과실비율 차이가 있음")
    if any(r.warnings for r in reviews):
        signals.append("위원 평가에 경고가 있음")
    return signals


def _format_review_summary(reviews) -> str:
    return "\n".join(
        f"- {r.perspective} (확신 {r.confidence}): 도표 {r.chart_no or '-'}, 정황 A 과실 {r.scene_a_fault}\n  근거: {r.rationale.strip()[:300]}"
        for r in reviews
    )


async def plan_clarification(llm, context: dict[str, Any]):
    """위원 결과와 정황으로 질문을 만든다. 반환: (질문 목록(0~1개), 참고 문구, 경고 목록).

    질문이 필요 없으면 빈 목록. 보기가 도표 데이터와 어긋나면(없는 도표, 기본 비율과 크게 다른 값) 질문을 버리고 경고만 남긴다.
    보기별 결과 차이가 CLARIFY_MIN_DIFF 이하면 사용자에게 묻지 않고 참고 문구만 돌려준다.
    """
    from app.core.verdict_check import check_scene_vs_chart_any_role
    from app.models.schemas import ClarificationPlan

    # 위원들이 같은 도표·같은 값·높은 확신이면 정황이 충분히 정해진 사고로 보고 질문을 만들지 않는다 (명백한 사례에서 질문을 지어낸 오탐 방지)
    if not ambiguity_signals(context["reviews"]):
        return [], "", []

    user_prompt = (
        f"[사고 정황]\n{context['accident_input']}\n\n"
        f"[검토위원 결론]\n{_format_review_summary(context['reviews'])}\n\n"
        f"[인용 도표 상세]\n{context.get('cited_charts', '(없음)')}\n"
    )
    plan = await llm.complete_json(CLARIFY_PROMPT, user_prompt, ClarificationPlan, mock={"questions": []})
    if not plan.questions:
        return [], "", []

    q = plan.questions[0]
    warnings: list[str] = []
    # 보기의 도표는 A/B 에이전트나 위원이 실제로 인용한 도표로 제한한다 (정황과 안 맞는 도표가 보기로 끼어드는 것 방지)
    allowed = set(context.get("cited_chart_numbers") or []) | {r.chart_no for r in context["reviews"] if r.chart_no}
    if allowed:
        kept = [o for o in q.options if o.chart_no.strip() in allowed]
        if len(kept) < len(q.options):
            dropped = sorted({o.chart_no for o in q.options} - allowed)
            warnings.append(f"질문 단계: 인용되지 않은 도표를 쓴 보기를 제외했습니다 ({', '.join(dropped)})")
            if len(kept) < 2:
                return [], "", warnings
            q = q.model_copy(update={"options": kept})
    problems = [p for o in q.options for p in check_scene_vs_chart_any_role(o.chart_no.strip(), o.scene_a_fault)]
    if problems:
        warnings.append(f"질문 단계: 보기의 도표·과실이 도표 데이터와 맞지 않아 질문을 생략했습니다 ({problems[0]})")
        return [], "", warnings
    if len({o.label.strip() for o in q.options}) < len(q.options):
        warnings.append("질문 단계: 보기가 중복되어 질문을 생략했습니다")
        return [], "", warnings

    faults = [o.scene_a_fault for o in q.options]
    if max(faults) - min(faults) <= CLARIFY_MIN_DIFF:
        detail = ", ".join(f"{o.chart_no} 적용 시 A {o.scene_a_fault}" for o in q.options)
        return [], f"정황에 없는 사실에 따라 적용 도표가 달라질 수 있으나 결과 차이는 작습니다 ({detail})", warnings
    return [q], "", warnings
