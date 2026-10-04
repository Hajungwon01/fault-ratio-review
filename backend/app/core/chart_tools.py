"""에이전트가 Tool Runner 루프 안에서 직접 호출하는 기준표 조회 도구.

도구 호출 기록(trace)을 에이전트별로 남기기 위해 호출마다 클로저로 만든다.
trace 는 FaultReport/프론트 공방 뷰에서 "어떤 도표를 실제로 조회했는지" 보여주는 데 쓴다.
"""
from __future__ import annotations

from anthropic import beta_async_tool

import asyncio

from app.config import get_settings
from app.core.chart_lookup import format_chart, get_catalog


def build_chart_tools(trace: list[str]) -> list:
    catalog = get_catalog()

    @beta_async_tool
    async def search_charts(query: str, limit: int = 5) -> str:
        """사고 상황 설명으로 과실비율 기준표(차 대 차 도표) 후보를 검색한다.

        Args:
            query: 사고 상황을 설명하는 한국어 문장 또는 키워드. 예: "후행 직진 차량과 선행 진로변경 차량 충돌"
            limit: 반환할 후보 수 (기본 5, 최대 10).
        """
        trace.append(f"search_charts({query!r})")
        # 임베딩 인코딩은 CPU 에서 수백 ms 걸리므로 이벤트 루프를 막지 않게 스레드로 넘긴다
        matches = await asyncio.to_thread(
            catalog.search, query, min(max(limit, 1), 10), get_settings().chart_search_mode
        )
        if not matches:
            return "검색 결과 없음. 다른 표현으로 다시 검색하세요."
        return "\n".join(f"{m.chart_no}  {m.chart['label']}  ({m.chart['category']})" for m in matches)

    @beta_async_tool
    async def get_chart(chart_no: str, include_notes: bool = False) -> str:
        """도표번호로 기본 과실비율, 경우((가)(나)(다)), 가감요소, 관련 법규를 조회한다.

        Args:
            chart_no: 도표번호. 예: "차43-2"
            include_notes: true면 수정요소 해설(가감요소 적용 기준 설명)까지 포함한다.
        """
        trace.append(f"get_chart({chart_no!r})")
        chart = catalog.get(chart_no.strip())
        if chart is None:
            return f"'{chart_no}' 는 존재하지 않는 도표번호입니다. search_charts 로 다시 찾으세요."
        return format_chart(chart_no.strip(), chart, with_notes=include_notes)

    return [search_charts, get_chart]
