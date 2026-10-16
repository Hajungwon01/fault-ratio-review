"""사용자용 설명(summary)을 쓰는 별도 호출.

진행 에이전트의 내부 요약(basis)과 한 호출로 묶었더니 모델이 내부 구조("위원들은 …", 사실검증 지적)를 따라 써서
실측 4건이 모두 검증에서 탈락했다. 그래서 위원·사실검증 정보는 아예 주지 않고, 정황 + 적용 도표 근거 + 최종 범위 +
확인되지 않은 사실만 주는 깨끗한 프롬프트로 따로 호출한다 (docs/USER_API_DESIGN.md §9).
"""
from __future__ import annotations

import logging

from app.core.clarify import ClarifiedResult
from app.core.user_text import chart_basis_entry, unique_basis
from app.models.schemas import FaultReport
from app.models.user_schemas import ChartBasis

logger = logging.getLogger(__name__)

USER_SUMMARY_PROMPT = (
    "당신은 교통사고 과실비율 확인 서비스에서 사고 당사자에게 결과를 설명하는 도우미입니다. 판정을 바꾸지 말고 주어진 결과를 쉽게 풀어 쓰세요.\n\n"
    "[규칙]\n"
    "- 3~5문장의 한국어 문단 하나.\n"
    "- 이 정황에 [적용한 기준표]가 왜 맞는지 정황의 내용과 연결해 설명하세요.\n"
    "- [확인되지 않은 사실]이 있으면 그 사실에 따라 결과가 달라질 수 있다고 쉬운 말로 알려 주세요. 있지 않은 사실을 단정하지 마세요.\n"
    "- [사용자 진술]이 없으면 사용자가 무언가를 답했다고 쓰지 마세요.\n"
    "- [사고 정황]처럼 대괄호로 된 항목 이름은 문장에 쓰지 마세요.\n"
    "- 숫자는 [적용한 기준표]와 [최종 결론]에 있는 값만 쓰세요. 도표번호도 주어진 것만 쓰세요. 판례나 법조항을 새로 만들어 쓰지 마세요.\n"
    "- 검토 과정이나 내부 절차(누가 어떻게 검토했는지, 검증, 오류 지적)는 설명하지 마세요. 결과와 이유만 쓰세요.\n"
    "- '내 차량', '상대 차량'으로 쓰세요. '위원', '검토위원', '정황 A/B', '도표 A/B', '확신', '플래그', '가정' 같은 말은 쓰지 마세요."
)


def _range_text(low: int, high: int) -> str:
    return f"{low}%" if low == high else f"{low}~{high}%"


def basis_for(report: FaultReport, resolved: ClarifiedResult | None) -> list[ChartBasis]:
    """도표 근거 블록. 모름이면 비운다(보기별 결과 비공개), 답변함이면 고른 도표만, 질문 없음이면 위원이 적용한 도표."""
    if resolved is not None:
        if resolved.chart_no is None:
            return []
        return unique_basis([chart_basis_entry(resolved.chart_no, resolved.scene_a_low)])
    return unique_basis(chart_basis_entry(r.chart_no, report.fault_ratio_a, r.chart_a_is) for r in report.reviews if r.chart_no)


def reviewers_range(report: FaultReport) -> tuple[int, int]:
    faults = [r.fault_ratio_a for r in report.reviews] or [report.fault_ratio_a]
    return min(faults), max(faults)


def unconfirmed_facts(report: FaultReport) -> list[str]:
    """위원들이 정황에 없는데 가정하고 판단한 사실(중복 제거) — 사용자에게는 '확인되지 않은 사실'로 전달한다."""
    seen, out = set(), []
    for r in report.reviews:
        for f in r.assumed_facts:
            key = f.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def build_prompt(accident_text: str, basis: list[ChartBasis], low: int, high: int,
                 *, unconfirmed: list[str] | None = None, statement: str = "") -> str:
    blocks = [f"[사고 정황]\n{accident_text}"]
    if statement:
        blocks.append(
            f"[사용자 진술]\n{statement}\n(사용자가 고른 답변입니다. 확인된 사실이 아니라 진술이므로 \"고르신 답변을 기준으로\"처럼 진술 기준임을 밝혀 쓰세요.)"
        )
    if basis:
        lines = []
        for b in basis:
            laws = " / ".join(b.laws) or "(없음)"
            lines.append(f"{b.chart_no} {b.label}\n도표 상황: {b.situation}\n기본 과실비율: 나 {b.my_base} : 상대 {b.other_base}\n관련 법규: {laws}")
        blocks.append("[적용한 기준표]\n" + "\n\n".join(lines))
    if unconfirmed:
        blocks.append("[확인되지 않은 사실]\n" + "\n".join(f"- {f}" for f in unconfirmed))
    blocks.append(f"[최종 결론]\n나의 과실 {_range_text(low, high)}, 상대 {_range_text(100 - high, 100 - low)}")
    return "\n\n".join(blocks) + "\n"


async def write_user_summary(llm, accident_text: str, basis: list[ChartBasis], low: int, high: int,
                             *, unconfirmed: list[str] | None = None, statement: str = "") -> str:
    """모델이 쓴 사용자용 설명 후보. mock·실패면 빈 문자열(호출 측이 코드 문장으로 대체). 검증은 호출 측이 한다."""
    if llm.mock_mode:
        return ""
    try:
        return await llm.complete(USER_SUMMARY_PROMPT, build_prompt(accident_text, basis, low, high, unconfirmed=unconfirmed, statement=statement))
    except Exception as e:  # noqa: BLE001 - 설명은 보조 산출물
        logger.warning("사용자용 설명 생성 실패: %s", e)
        return ""
