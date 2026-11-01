# 교통사고 과실비율 검증 서비스 (멀티 에이전트)

사고 당사자가 자신의 **과실비율이 기준에 맞는지 근거와 함께 확인**할 수 있게 하는 개인 프로젝트입니다.
사고 정황을 입력하면 7개 에이전트(A/B 차량측 논증, 사실검증, 검토위원 3종, 진행)가 손해보험협회 과실비율 인정기준(차 대 차 108개 도표)을 근거로 검토하고,
정황에 없는 사실 때문에 결과가 크게 갈리면 결과를 숨긴 채 사실 질문을 1개 던진 뒤 **과실 범위와 근거**를 보여 줍니다.

> **참고용 서비스입니다.** 법적 효력이 있는 판단이 아니며 실제 과실비율은 보험사·분쟁심의위원회·법원이 정합니다.
> 답변은 사실로 확정하지 않고 "사용자 진술 기준"으로만 표기합니다.
> 기준표 출처: 손해보험협회 과실비율정보포털(accident.knia.or.kr). 데이터는 공개 재배포하지 않습니다(이 저장소는 비공개).

(2026-10-14 주 사용자를 보험사 심사자에서 사고 당사자로 변경 — [기획서 §9](docs/deliverables/02_프로젝트_기획서.md))

## 현재 상태

| 영역 | 상태 |
|---|---|
| 기준표 데이터 | 차 대 차 108개 도표 수집·구조화 완료 (가감요소·법규·판례 포함). 보행자·자전거는 범위 밖 |
| 에이전트 파이프라인 | 7개 에이전트, 도구 호출 루프, 위원 3종 중앙값 집계, 정밀 심사 플래그 구현 |
| 사용자용 API·화면 | 입력 → 대기 → (질문) → 결과 흐름 구현 (`/v1/assess`) |
| 평가 도구 | 시나리오 일괄 평가, 포털식 baseline 비교, 검색 방식 비교 구현 |
| 테스트 | 자동 테스트 288개 통과 (외부 호출 없음) |
| 평가 데이터 | 개발용 시나리오 15건만 있음. **보고용 시나리오 세트·구어체 평가셋은 수집 중** → 개발용 수치는 보고용이 아닙니다 |

산출물 현황: [docs/deliverables/00_산출물_현황.md](docs/deliverables/00_산출물_현황.md) · 일정: [01_WBS.md](docs/deliverables/01_WBS.md)

## 실행 방법

### 백엔드

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # OPENAI_API_KEY, OPENAI_MODEL 입력 (키가 없으면 mock 모드)
python scripts/verify_llm.py   # 선택: 키·모델 연동 확인
uvicorn app.main:app --reload --port 8000
```

http://localhost:8000/health 로 동작을 확인합니다. **키가 없으면 mock 모드**로 전체 흐름이 돌지만 응답은 가짜이며 질문은 생성되지 않습니다.
실제 모델을 쓰면 요청 한 번에 LLM 호출이 약 10~17회, 수십 초(최근 평균 약 35~47초) 걸리고 비용이 듭니다.

### 프론트엔드

`frontend/index.html` 을 브라우저로 열면 됩니다(백엔드가 `localhost:8000` 에서 실행 중이어야 함). 다른 주소는 `index.html?api=http://주소:포트` 로 지정합니다. 빌드 도구는 없습니다.

### 테스트

```bash
cd backend
python -m pytest -q        # 외부 호출 없음, 약 15초
```

### 임베딩 검색(선택)

`CHART_SEARCH_MODE=hybrid` 는 로컬 KURE-v1 임베딩을 씁니다. `pip install -r requirements-embedding.txt` 가 필요하고 모델 약 2GB를 최초 1회 내려받습니다.
설치하지 않으면 키워드 검색으로 자동 대체됩니다. 방식별 비교: [10_벡터_검색_비교_결과서.md](docs/deliverables/10_벡터_검색_비교_결과서.md)

### 평가 도구 (실제 API 호출 시 비용 발생)

```bash
cd backend
python scripts/run_scenarios.py --dry-run                      # 호출 없이 시나리오·설정 검증
python scripts/run_scenarios.py --limit 1 --models <모델ID>    # 1건으로 호출 수·토큰 확인 후 전체 실행
python scripts/run_scenarios.py --resume --price-in <단가> --price-out <단가>   # 중단 재개, 비용 계산
python scripts/run_baseline.py                                 # 포털식 단순 매칭 baseline (비용 없음)
python scripts/eval_chart_search.py -v                         # 검색 방식 비교 (비용 없음)
```

결과는 `backend/eval/results/` 에 저장되며 저장소에는 올리지 않습니다. 자세한 계획과 결과: [11_테스트_계획_및_결과_보고서.md](docs/deliverables/11_테스트_계획_및_결과_보고서.md)

## 주요 설정 (`backend/.env`)

| 변수 | 설명 |
|---|---|
| `LLM_PROVIDER` | `openai`(기본) 또는 `anthropic` |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | 모델 ID 는 코드에 고정하지 않고 직접 지정 |
| `CHART_SEARCH_MODE` | `keyword`(기본) / `embedding` / `hybrid` |
| `INTERNAL_API_ENABLED` | 내부·개발용 `/assess`(보기별 도표·과실이 그대로 나감). **사용자에게 공개하는 환경에서는 `false`** |

전체 목록은 [09_시스템_구성도.md](docs/deliverables/09_시스템_구성도.md) §7. **API 키는 `.env` 에만 두고 저장소에 올리지 않습니다**(`.gitignore` 로 제외).

## 구조

```
backend/app/
  main.py           FastAPI 엔트리 (/v1/assess, /v1/assess/{id}/answer, /health, 내부용 /assess)
  agents/           7개 에이전트와 진행 에이전트(질문 계획 포함)
  core/             파이프라인, LLM 클라이언트, 기준표 조회·검색, 검증 코드, 사용자용 흐름·세션
  models/           Pydantic 스키마
  evaluation/       시나리오 평가·baseline
  data/             fault_ratio_reference.json (108개 도표)
backend/scripts/    평가·수집·연동 확인 스크립트
backend/eval/       평가 시나리오와 평가셋 (결과는 results/, git 제외)
backend/tests/      자동 테스트
frontend/           순수 HTML/CSS/JS 화면
docs/deliverables/  산출물 문서 (기획서·요구사항·설계·보고서)
```

문서 안내: 화면 [08_화면설계서](docs/deliverables/08_화면설계서.md) · 구성 [09_시스템_구성도](docs/deliverables/09_시스템_구성도.md) · AI 구조 [07_AI_시스템_아키텍처](docs/deliverables/07_AI_시스템_아키텍처.md) · API [docs/USER_API_DESIGN.md](docs/USER_API_DESIGN.md) · 개요 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 기준표 데이터 수집

과실비율정보포털에 공식 API는 없지만 도표 상세 페이지가 서버 렌더링 HTML이라 도표번호로 직접 파싱할 수 있습니다.
`backend/scripts/scrape_knia_charts.py` (`pip install beautifulsoup4 lxml`)로 108개 도표의 기본 과실(가)(나)(다), 경우별 가감요소, 법규, 판례를 수집해 `fault_ratio_reference.json` 에 저장합니다.
이 서버는 중간 인증서 체인이 불완전해 Python 기본 SSL 검증이 실패하므로 OS 인증서를 쓰는 `curl` 로 수신합니다(검증을 끄지 않음). 서비스 실행 중에는 포털에 접속하지 않습니다.

## 알려진 한계

- **평가 근거가 약합니다**: 보고용 시나리오 세트가 오기 전이며 개발용 13/13 은 그 세트에 맞춰 고친 결과입니다. 모델은 `gpt-4.1` 위주, 반복 횟수가 적습니다.
- **모호한 입력의 질문 품질이 불안정합니다**: 정황과 모순되는 보기는 걸러내지만 보기가 빠짐없는지는 점검하지 못합니다(실측 D14).
- **사용자 답변은 검증되지 않습니다**: 과실이 적게 나오는 답을 고를 동기가 있어 결과 숨김·보기 섞기·"모름"으로 완화할 뿐 막지 못합니다.
- **개인정보**: 입력한 사고 정황이 외부 LLM(OpenAI)으로 전송됩니다. 개인 식별 정보를 적지 않도록 안내하고, 공개 서비스에는 처리 고지가 필요합니다.
- **운영 준비가 안 됐습니다**: CORS 전체 허용·인증 없음, 세션이 서버 메모리라 재시작 시 사라짐, 동기 호출(수십 초~2분), 배포 구성 없음.
- 지원 범위는 차 대 차이며 보행자·자전거·킥보드·단독사고는 거절합니다. 실기기·Safari·스크린 리더 확인은 하지 못했습니다.
