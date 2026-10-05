"""모든 에이전트가 공유하는 베이스 클래스.

각 에이전트는 run(context)만 구현하면 된다. 라운드 순서/병렬 실행은
core/pipeline.py 의 FaultAssessmentPipeline이 담당하므로, 에이전트 자체는
"주어진 컨텍스트로 LLM을 호출해 구조화된 결과를 낸다"는 책임만 진다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.chart_tools import build_chart_tools
from app.core.verdict_check import check_verdict
from app.core.llm_client import LLMClient
from app.models.schemas import AgentArgument


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """역할별 시스템 프롬프트. TODO(3~4주차부터 순차 작성)."""

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> Any:
        """context를 받아 이 에이전트의 산출물을 반환한다."""


def format_rounds(rounds: list | None) -> str:
    """A/B 논증 기록을 위원이 읽기 쉬운 텍스트로 만든다 (pydantic repr 노출 방지)."""
    if not rounds:
        return "(없음)"
    blocks = []
    for r in rounds:
        side = "A" if "vehicle_a" in r.agent_name else "B" if "vehicle_b" in r.agent_name else r.agent_name
        cited = f"\n인용: {', '.join(r.cited_rules)}" if r.cited_rules else ""
        blocks.append(f"[{side}측 {r.round}라운드]\n{r.content}{cited}")
    return "\n\n".join(blocks)


def format_fact_check(fc) -> str:
    if fc is None:
        return "(없음)"
    head = "인용 오류 없음" if fc.is_valid else "인용 오류 있음"
    flagged = "\n".join(f"- {f}" for f in fc.flagged_citations) or "- 지적 사항 없음"
    return f"{head}\n{flagged}"


class ReviewerAgent(BaseAgent):
    """검토위원 3종(기준표/현장정황/유사사례)의 공통 부모.

    perspective만 다르고 평가 로직 골격은 동일하므로 여기서 공유한다.
    응답은 ReviewerVerdict 스키마로 구조화 출력을 받고(파싱 실패 시 LLMClient 가 재시도),
    B 과실은 100 - A 로 계산한다. 관점별 체크리스트 프롬프트는 서브클래스의 system_prompt (#8~#10).
    """

    perspective: str = "unspecified"

    # 모든 위원에게 공통으로 붙는 출력 규칙. 관점별 프롬프트는 그대로 두고 뒤에 덧붙인다.
    OUTPUT_RULES = (
        "\n\n[출력 규칙]\n"
        "- 반드시 지정된 JSON 스키마로만 답하세요.\n"
        "- A/B 에이전트의 주장은 서로 유리하게 쓴 것이므로 그대로 따르지 말고, 당신의 관점에서 독립적으로 평가하세요.\n"
        "- 사실검증이 지적한 오류가 있는 주장은 근거로 삼지 마세요.\n"
        "- **도표 안의 A/B 는 도표가 정한 역할 이름이며 정황의 A(나)/B(상대)와 다를 수 있습니다.** 도표를 적용하기 전에 "
        "도표의 'A=…/B=…' 정의와 정황을 대조해 `도표 A = 정황 A/B` 대응을 확인하세요. "
        "A/B 에이전트가 이 대응을 틀렸다면 그 주장은 따르지 마세요.\n"
        "- **숫자는 맨 마지막에 정하세요.** JSON 필드 순서대로 먼저 checklist, rationale 로 판단을 끝낸 뒤 "
        "scene_a_fault, chart_no, chart_a_is 를 채우세요. 근거 문장과 숫자가 모순되면 안 됩니다.\n"
        "- scene_a_fault: **정황의 A(나의 차량)** 입장의 최종 과실비율(0~100 정수). 최종 결론이며, 도표의 A/B 가 아니라 "
        "**정황의 A/B** 기준입니다. B 과실은 100 에서 뺀 값이라 적지 않습니다.\n"
        "- chart_no: 적용한 도표번호 (적용하지 않았으면 빈 문자열). "
        "chart_a_is: 도표 안의 A차량이 정황의 A(나의 차량)이면 \"A\", 정황의 B(상대 차량)이면 \"B\", 도표를 적용하지 않았으면 \"none\". "
        "시스템이 이 값과 도표의 기본 과실비율로 scene_a_fault 가 말이 되는지 확인합니다.\n"
        "- 예) 도표 차41-1 은 도표 A=뒤차 추돌(과실 100), 도표 B=앞차(과실 0)입니다. 정황의 나(A)가 정차 중이던 앞차였다면 "
        "chart_a_is=\"B\", scene_a_fault=0 입니다 (도표 B 가 나이므로 도표 B 의 과실 0 이 곧 나의 과실). "
        "반대로 내가 뒤차로 추돌했다면 chart_a_is=\"A\", scene_a_fault=100 입니다.\n"
        "- 도표·정황 정보가 부족해 판단이 불확실하면 confidence 를 low 로 두세요.\n"
        "- checklist 에는 위 [체크리스트] 항목을 빠짐없이, 항목 이름을 그대로 써서 항목마다 하나씩 판단을 적으세요. "
        "정황에 없는 사실은 추측하지 말고 '정황 정보 없음'이라고 쓰세요.\n"
        "- 주어진 [인용 도표 상세]·[판례·조정사례] 밖의 도표번호·판례·비율을 만들어 인용하지 마세요."
    )

    def _extra_sections(self, context: dict[str, Any]) -> str:
        """관점별로 사용자 프롬프트에 덧붙일 자료. 서브클래스가 재정의한다."""
        return ""

    async def run(self, context: dict[str, Any]):
        from app.models.schemas import ReviewerVerdict, ReviewScore

        verdict = await self.llm.complete_json(
            self.system_prompt + self.OUTPUT_RULES,
            self._build_user_prompt(context),
            ReviewerVerdict,
            mock={
                "checklist": [],
                "rationale": f"[MOCK {self.perspective}] 실제 LLM 이 연결되지 않아 자리표시 평가를 반환합니다.",
                "scene_a_fault": 50,
                "chart_no": "",
                "chart_a_is": "none",
                "confidence": "low",
                "cited_rules": [],
            },
        )
        # 도표 기본 비율과 대조해 앞뒤가 안 맞으면 신뢰도를 낮추고 경고를 남긴다 (전원이 같이 틀려도 걸러낼 수 있게)
        problems = check_verdict(verdict)
        return ReviewScore(
            reviewer_name=self.name,
            perspective=self.perspective,
            fault_ratio_a=verdict.fault_ratio_a,
            fault_ratio_b=100 - verdict.fault_ratio_a,
            rationale=verdict.rationale,
            confidence="low" if problems else verdict.confidence,
            cited_rules=verdict.cited_rules,
            checklist=verdict.checklist,
            chart_no=verdict.chart_no.strip(),
            chart_a_is=verdict.chart_a_is,
            scene_a_fault=verdict.scene_a_fault,
            warnings=problems,
        )

    def _build_user_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"[{self.perspective} 관점에서 평가]\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[인용 도표 상세]\n{context.get('cited_charts', '(없음)')}\n\n"
            f"{self._extra_sections(context)}"
            f"[A/B 논증]\n{format_rounds(context.get('rounds'))}\n\n"
            f"[사실검증 결과]\n{format_fact_check(context.get('fact_check'))}\n"
        )


class AdvocateAgent(BaseAgent):
    """A/B차량측 에이전트 공통 부모. side("A"/"B")만 다르고 논증 골격은 대칭이다."""

    side: str = "A"

    @property
    def other(self) -> str:
        return "B" if self.side == "A" else "A"

    @property
    def system_prompt(self) -> str:
        me, other = self.side, self.other
        return (
            f"당신은 {me}차량측을 대리해 과실비율 심사에서 {me}차량의 과실을 최소화하는 논증을 만드는 에이전트입니다.\n\n"
            "[규칙]\n"
            "1. 근거는 [참조 기준표]와, get_chart/search_charts 도구로 조회한 도표의 기본 과실비율·가감요소·법규만 사용하세요. "
            "[참조 기준표]는 자동 검색된 후보일 뿐 틀릴 수 있으므로, 더 맞는 도표가 있을 것 같으면 search_charts 로 직접 찾으세요 "
            "(질의는 '후행 직진 대 선행 진로변경'처럼 도표 명칭 어투로 쓰면 잘 찾아집니다). "
            "인용하기 전에 get_chart 로 해당 도표의 내용을 확인하고, 조회하지 않은 도표번호·비율·조문을 지어내지 마세요.\n"
            f"2. 입력된 사고 정황에 실제로 있는 사실만 가감요소 근거로 삼으세요. "
            f"{other}차량의 가산 사유({other} 쪽 가감요소)와 {me}차량의 감산 사유({me} 쪽 음수 가감요소)를 정황에서 찾아 제시하세요.\n"
            "3. **도표 선택과 역할 대응** (가장 중요): ① 먼저 정황에서 두 차량이 각각 무엇을 하고 있었는지(직진/진로변경/좌회전/후진/정차 등)를 한 줄로 요약하세요. "
            "② 그 움직임에 맞는 도표를 고르되, 도표의 '사고 상황'·'적용(비적용)'이 정황과 맞는지 확인하세요 (맞지 않으면 다른 도표를 찾으세요). "
            "③ **도표 안의 A차량/B차량은 도표가 정한 역할 이름이며 정황의 A(나)/B(상대)와 다를 수 있습니다.** 도표의 'A=…/B=…' 정의를 읽고 "
            "`도표 A = 정황 A` 또는 `도표 A = 정황 B` 형식으로 대응을 첫머리에 밝히세요. "
            "④ 이후 모든 과실비율은 **정황의 A/B 기준**으로 환산해 쓰세요 (예: 도표 A 30 : B 70 인데 도표 A 가 정황의 B 라면, 정황 A 는 70 입니다). "
            "경우((가)(나)(다))도 하나 고르고, 정보가 부족해 확정할 수 없으면 그 점을 명시하세요.\n"
            "4. 2라운드에는 상대측 주장의 근거 오류(잘못된 도표 선택, 정황에 없는 가감요소, 역할 대응·계산 오류)를 짚어 반박하고, 자기 주장을 보강하세요. "
            "1라운드에서 고른 도표는 사실검증이 틀렸다고 지적했거나 정황과 맞지 않음이 분명할 때만 바꾸세요. "
            "내 과실을 줄이려고 정황과 맞지 않는 도표로 갈아타면 사실검증에서 감점됩니다.\n"
            "5. **계산 규칙**: 과실비율은 항상 합이 100 입니다. 도표 기준 A 과실 = 기본 A + (도표 A측 가감점) − (도표 B측 가감점)이며, "
            "한쪽을 올리면 다른 쪽이 내려갑니다. 예: 기본 A 30 : B 70 에서 B측 +10 이면 A 20 : B 80 입니다 "
            "(B 만 80 으로 올리고 A 를 30 으로 두면 합이 110 이 되어 틀립니다). 같은 [택일N] 표시의 가감요소는 하나만 쓰세요. "
            "결론은 `최종: 정황 A xx : 정황 B yy` 한 줄로 쓰세요.\n"
            f"6. 마지막 줄에 `인용: 차XX-X, 도로교통법 제N조` 형식으로 인용한 도표·조문을 쉼표로 나열하세요.\n"
            f"7. 당신은 {me}차량측 변호인입니다. {me}에 유리한 해석을 하되, 사실을 왜곡하면 사실검증에서 감점됩니다."
        )

    def _fact_check_block(self, context: dict[str, Any]) -> str:
        """2라운드: 사실검증이 지적한 사항을 측별로 나눠 전달한다 (1라운드에는 없음)."""
        fc = context.get("fact_check")
        if fc is None:
            return ""
        mine = [f for f in fc.flagged_citations if f.removeprefix("(경고) ").startswith(f"{self.side}측")]
        theirs = [f for f in fc.flagged_citations if f.removeprefix("(경고) ").startswith(f"{self.other}측")]
        return (
            "[사실검증 결과]\n"
            f"- {self.side}측(당신) 지적: {'; '.join(mine) or '없음'}  → 틀린 인용은 정정하고, 정정 불가한 주장은 철회하세요.\n"
            f"- {self.other}측(상대) 지적: {'; '.join(theirs) or '없음'}  → 상대 주장의 약점으로 활용하세요.\n\n"
        )

    async def run(self, context: dict[str, Any]) -> AgentArgument:
        round_no = context.get("round", 1)
        user_prompt = (
            f"[참조 기준표]\n{context.get('reference_charts', '(없음)')}\n\n"
            f"[사고 정황]\n{context.get('accident_input')}\n\n"
            f"[상대측({self.other}차량) 이전 주장]\n{context.get('opponent_argument')}\n\n"
            f"{self._fact_check_block(context)}"
            f"현재 라운드: {round_no}"
        )
        trace: list[str] = []
        content = await self.llm.run_agent(self.system_prompt, user_prompt, build_chart_tools(trace))
        return AgentArgument(
            agent_name=self.name,
            round=round_no,
            content=content,
            cited_rules=extract_citations(content),
            tool_calls=trace,
        )


def extract_citations(text: str) -> list[str]:
    """논증 본문에서 도표번호(차12-1)와 도로교통법 조문을 뽑는다."""
    import re

    charts = re.findall(r"차\d+-\d+", text)
    laws = [f"도로교통법 제{n}조" for n in re.findall(r"도로교통법\s*제(\d+)조", text)]
    return list(dict.fromkeys(charts + laws))
