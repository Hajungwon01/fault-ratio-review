"""A차량측 에이전트 — A차량 과실 최소화 근거 논증.

TODO(3~4주차): 과실비율 인정기준표를 근거로 실제 논증을 생성하도록 system_prompt 작성.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import AgentArgument


class VehicleAAgent(BaseAgent):
    name = "vehicle_a_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 A차량측을 대리해 과실비율 심사에서 A차량의 과실을 최소화하는 논증을 만드는 에이전트입니다.\n"
            "TODO: 표준 과실비율 인정기준표와 도로교통법 조항을 인용해 근거를 제시하도록 구체화."
        )

    async def run(self, context: dict[str, Any]) -> AgentArgument:
        round_no = context.get("round", 1)
        user_prompt = (
            f"사고 정황: {context.get('accident_input')}\n"
            f"상대측(B차량) 이전 주장: {context.get('opponent_argument')}\n"
            f"현재 라운드: {round_no}"
        )
        content = await self.llm.complete(self.system_prompt, user_prompt)
        return AgentArgument(agent_name=self.name, round=round_no, content=content)
