"""사실검증 에이전트 — 양측이 인용한 기준표·법조항 적용의 정확성 검증.

검증은 2단계: ① core/citation_check.py 규칙 기반(도표 존재·기본 과실비율·법조문 대응) → ② LLM + 도구(get_chart)로 나머지 확인.
flagged_citations 는 "A측/B측 <내용>" 형식이며, 규칙 경고는 "(경고)" 접두어로 구분한다.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.chart_tools import build_chart_tools
from app.core.citation_check import verify_citations
from app.models.schemas import FactCheckResult


class FactCheckerAgent(BaseAgent):
    name = "fact_checker_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 A/B차량측 에이전트가 인용한 과실비율 기준표 및 도로교통법 조항이 "
            "실제로 정확하게 적용되었는지 검증하는 사실검증 에이전트입니다.\n"
            "[절차] 각 주장이 인용한 도표번호마다 get_chart 도구로 실제 내용을 조회하고, 그 결과와 대조해 각 주장의 (1) 도표번호가 실제 목록에 있는지 (2) 기본 과실비율이 맞는지 "
            "(3) 인용한 가감요소가 해당 도표에 존재하고 사고 정황에 근거가 있는지, "
            "(4) 인용 도표의 '사고 상황'이 사고 정황의 두 차량 움직임과 실제로 맞는지, "
            "(5) **도표의 A/B 정의와 정황의 A/B 대응이 맞고, 말한 과실비율이 정황 A 기준으로 올바르게 환산됐는지** "
            "(도표 A 가 정황의 B 인데 도표 A 비율을 정황 A 비율처럼 쓴 경우가 대표적 오류)를 항목별로 확인하세요. "
            "정황과 맞지 않는 도표를 인용했다면 get_chart/search_charts 로 더 맞는 도표를 찾아 어떤 도표가 맞는지도 오류 내용에 적으세요. "
            "조회 결과와 다른 인용은 한 줄에 하나씩 `오류: <A측|B측> <내용>` 형식으로 나열하고, 오류가 없으면 `오류 없음`이라고 쓰세요."
        )

    async def run(self, context: dict[str, Any]) -> FactCheckResult:
        # 1) 규칙 기반 사전 검증 (도표 존재 / 기본 과실비율 / 법조문-도표 대응)
        rule_findings = {
            "A측": verify_citations(str(context.get("argument_a") or "")),
            "B측": verify_citations(str(context.get("argument_b") or "")),
        }
        rule_lines = [f"{side} {f}" for side, fs in rule_findings.items() for f in fs]

        # 2) LLM 검증: 규칙 검증 결과를 알려주고 도구로 나머지(가감요소 적용 적정성 등)를 확인
        user_prompt = (
            f"[참조 기준표]\n{context.get('reference_charts', '(없음)')}\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[규칙 기반 사전 검증 결과]\n{chr(10).join(rule_lines) or '지적 사항 없음'}\n\n"
            f"A차량측 주장: {context.get('argument_a')}\n"
            f"B차량측 주장: {context.get('argument_b')}\n"
        )
        trace: list[str] = []
        notes = await self.llm.run_agent(self.system_prompt, user_prompt, build_chart_tools(trace))
        llm_errors = [line.split(":", 1)[1].strip() for line in notes.splitlines() if line.strip().startswith("오류:")]

        rule_errors = [
            f"{side} {f.message}" for side, fs in rule_findings.items() for f in fs if f.level == "error"
        ]
        rule_warnings = [
            f"(경고) {side} {f.message}" for side, fs in rule_findings.items() for f in fs if f.level == "warning"
        ]
        # 같은 내용을 규칙·LLM 이 둘 다 지적하면 중복이 생길 수 있어 순서를 유지하며 제거
        flagged = list(dict.fromkeys(rule_errors + llm_errors + rule_warnings))
        return FactCheckResult(
            is_valid=not (rule_errors or llm_errors), notes=notes, flagged_citations=flagged
        )
