"""과실비율정보포털(accident.knia.or.kr) '차 대 차' 도표 상세 페이지를 수집해
backend/app/data/fault_ratio_reference.json 의 chart_catalog 를 갱신한다.

도표 상세 페이지(myaccident-content?chartNo=차51-1&chartType=1)는 서버 렌더링 HTML이라
별도 API 없이 파싱할 수 있다. 존재하지 않는 chartNo 는 HTTP 500 을 돌려준다.

이 서버는 중간 인증서 체인이 불완전해 Python 기본 SSL 검증이 실패하므로,
OS 인증서 저장소를 쓰는 curl 로 받는다 (SSL 검증을 끄지 않는다).

실행:  python backend/scripts/scrape_knia_charts.py
의존성: beautifulsoup4, lxml (수집 스크립트 전용, 서버 런타임에는 불필요)
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://accident.knia.or.kr/myaccident-content?chartNo={no}&chartType=1"
DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "fault_ratio_reference.json"
MAX_GROUP = 70  # 차1 ~ 차70
MAX_SUB = 12
DELAY_SEC = 0.3


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch(chart_no: str) -> str | None:
    url = BASE.format(no=urllib.parse.quote(chart_no))
    proc = subprocess.run(
        ["curl", "-s", "-A", "Mozilla/5.0", "-w", "\n%{http_code}", url],
        capture_output=True,
    )
    body, _, code = proc.stdout.rpartition(b"\n")
    if code.strip() != b"200":
        return None
    return body.decode("utf-8")


def parse_modifiers(ul) -> list[dict]:
    mods = []
    if ul is None:
        return mods
    for li in ul.select(":scope > li"):
        label = norm(li.select_one(".center").get_text())
        a = int(norm(li.select_one(".side em.isA").get_text()))
        b = int(norm(li.select_one(".side em.isB").get_text()))
        # 가감요소는 한쪽 차량에만 걸린다. 값은 가산(+)/감산(-) 점수.
        side, adj = ("A", a) if a else ("B", b)
        mods.append({"side": side, "condition": re.sub(r"^[AB]\s+", "", label), "adjustment": adj})
    return mods


def parse_variants(soup) -> list[dict]:
    variants = []
    tabs = soup.select(".ratioHead .tabs button")
    if tabs:
        specs = [(norm(b.select_one("span").get_text()), int(b["data-ratio-a"]), int(b["data-ratio-b"])) for b in tabs]
    else:
        nums = [int(e.get_text()) for e in soup.select(".visual .score .scoreNum.pc")]
        specs = [("", nums[0], nums[1])]
    for i, (label, a, b) in enumerate(specs, start=1):
        status = soup.select_one(f"#case{i}_status") or (soup.select_one(".ratioHead .txtBox") if i == 1 else None)
        descs = [re.sub(r"^자동차\s*[AB]\s*:\s*", "", norm(p.get_text())) for p in status.select("p")] if status else ["", ""]
        score_ul = soup.select_one(f"#case{i}_score") or (soup.select_one(".factorList") if i == 1 else None)
        variants.append({
            "label": label,
            "base_ratio": {"a": a, "b": b},
            "vehicle_a": descs[0],
            "vehicle_b": descs[1] if len(descs) > 1 else "",
            "modifiers": parse_modifiers(score_ul),
        })
    return variants


def parse_acc_items(soup) -> dict[str, list[dict]]:
    out = {}
    for item in soup.select(".accSec .acc .item"):
        title = norm(item.select_one("button.head").get_text())
        entries = []
        for li in item.select(".caseList > li"):
            strong = li.find("strong")
            paras = [norm(p.get_text(" ")) for p in li.find_all("p")]
            entries.append({"title": norm(strong.get_text()) if strong else "", "text": "\n".join(p for p in paras if p)})
        out[title] = entries
    return out


def parse_chart(chart_no: str, html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    if not soup.select_one(".infoSec h3.title"):
        return None
    crumbs = [norm(e.get_text()) for e in soup.select(".subLocationInner > *")]
    crumbs = [c for c in crumbs if c and c != "홈" and c != chart_no]
    info = {norm(dt.get_text()): norm(dt.find_next("dd").get_text(" ")) for dt in soup.select(".infoList dt")}
    variants = parse_variants(soup)
    acc = parse_acc_items(soup)
    return {
        "category": " > ".join(crumbs[:-1]),
        "label": crumbs[-1] if crumbs else "",
        "situation": info.get("사고 상황", ""),
        "applicability": info.get("적용(비적용)", ""),
        "base_rationale": info.get("기본 과실 해설", ""),
        "base_ratio": variants[0]["base_ratio"],
        "variants": variants,
        "modifier_notes": acc.get("수정요소 해설", []),
        "laws": acc.get("관련 법규", []),
        "precedents": acc.get("판례·조정사례", []),
    }


def main() -> int:
    catalog = {}
    for g in range(1, MAX_GROUP + 1):
        misses = 0
        for s in range(1, MAX_SUB + 1):
            no = f"차{g}-{s}"
            html = fetch(no)
            parsed = parse_chart(no, html) if html else None
            time.sleep(DELAY_SEC)
            if parsed is None:
                misses += 1
                if misses >= 2:
                    break
                continue
            misses = 0
            catalog[no] = parsed
            print(f"{no}  {parsed['base_ratio']}  {parsed['label']}", flush=True)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data["chart_catalog"] = {
        "_note": (
            "accident.knia.or.kr '차 대 차' 도표 상세 페이지를 backend/scripts/scrape_knia_charts.py 로 수집. "
            "variants 는 (가)(나)(다) 처럼 기본 과실이 여러 개인 도표의 경우별 값이며 base_ratio 는 첫 variant. "
            "modifiers.adjustment 는 해당 side 차량의 과실 가산(+)/감산(-) 포인트."
        ),
        **catalog,
    }
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(catalog)}개 도표 저장 -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
