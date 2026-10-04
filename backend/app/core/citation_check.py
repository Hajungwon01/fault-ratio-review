"""논증 텍스트의 기준표 인용을 규칙 기반으로 검증한다 (LLM 사실검증 이전 단계).

잡아내는 것
- error:   존재하지 않는 도표번호 (환각 인용)
- error:   도표번호 근처에서 주장한 과실비율이 그 도표의 어떤 경우(variant)와도 맞지 않음
- warning: 인용한 도로교통법 조문이 인용한 도표의 '관련 법규'에 없음 (다른 조문일 수 있어 경고만)

비율은 "차43-2 ... 30:70" 처럼 도표번호 뒤 60자 안에 `숫자:숫자` 로 쓰인 경우만 검사한다.
A:B 순서가 바뀌어 쓰였을 수 있어 뒤집힌 값도 허용한다. 그래서 놓치는 경우는 있어도(false negative)
멀쩡한 인용을 틀렸다고 하는 경우(false positive)는 줄이는 방향으로 설계했다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.chart_lookup import ChartCatalog, get_catalog

_CHART_RE = re.compile(r"차\d+-\d+")
_RATIO_RE = re.compile(r"(\d{1,3})\s*[:대]\s*(\d{1,3})")
_LAW_RE = re.compile(r"도로교통법\s*제\s*(\d+)\s*조")
_RATIO_WINDOW = 60
# "A 30 : B 80", "A차량 30% : B차량 80%" 처럼 A/B 를 명시한 비율 쌍
# 프롬프트가 지정한 출력 형식 "최종: 정황 A 30 : 정황 B 80" 도 잡아야 하므로 A/B 앞의 '정황' 접두어를 허용한다
_PAIR_RE = re.compile(r"(?:정황\s*)?A\s*(?:차량)?\s*(\d{1,3})\s*%?\s*[:：대]\s*(?:정황\s*)?B\s*(?:차량)?\s*(\d{1,3})")


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


def _valid_ratios(chart: dict) -> set[tuple[int, int]]:
    ratios = set()
    for v in chart["variants"]:
        a, b = v["base_ratio"]["a"], v["base_ratio"]["b"]
        ratios |= {(a, b), (b, a)}
    return ratios


def verify_citations(text: str, catalog: ChartCatalog | None = None) -> list[Finding]:
    catalog = catalog or get_catalog()
    findings: list[Finding] = []
    cited: dict[str, dict] = {}

    for m in _CHART_RE.finditer(text):
        no = m.group()
        chart = catalog.get(no)
        if chart is None:
            if not any(f.message.startswith(f"{no} ") for f in findings):
                findings.append(Finding("error", f"{no} 는 존재하지 않는 도표번호"))
            continue
        cited[no] = chart
        window = text[m.end() : m.end() + _RATIO_WINDOW]
        r = _RATIO_RE.search(window)
        if r:
            claimed = (int(r.group(1)), int(r.group(2)))
            if claimed[0] + claimed[1] == 100 and claimed not in _valid_ratios(chart):
                msg = f"{no} 의 기본 과실비율이 {claimed[0]}:{claimed[1]} 로 인용됐으나 실제는 " + ", ".join(
                    f"{a}:{b}" for a, b in sorted({(v['base_ratio']['a'], v['base_ratio']['b']) for v in chart['variants']})
                )
                if not any(f.message == msg for f in findings):
                    findings.append(Finding("error", msg))

    # 과실비율은 항상 합이 100 이다. 가감요소를 한쪽에만 더하고 반대쪽을 내리지 않은 계산("A 30 : B 80")을 잡는다.
    for m in _PAIR_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 100 and b <= 100 and a + b != 100:
            msg = (
                f"과실비율 합이 100이 아님: A {a} : B {b} (합 {a + b}). "
                "가감요소로 한쪽을 올리면 다른 쪽을 내려 합을 100으로 유지해야 함 (A 과실 = 기본 A + A측 가감점 − B측 가감점)"
            )
            if not any(f.message == msg for f in findings):
                findings.append(Finding("error", msg))

    if cited:
        known_articles = {a for c in cited.values() for law in c["laws"] for a in re.findall(r"제\s*(\d+)\s*조", law["title"])}
        for n in dict.fromkeys(_LAW_RE.findall(text)):
            if n not in known_articles:
                findings.append(
                    Finding("warning", f"도로교통법 제{n}조 는 인용된 도표({', '.join(cited)})의 관련 법규에 없음")
                )
    return findings
