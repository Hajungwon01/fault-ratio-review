# 과실비율 산정 심사 프로세스 자동화 (갠플 프로젝트)

보험사의 "사고 접수 → 손해사정사 검토 → 과실비율 산정 → 보험금 지급" 프로세스 중,
**과실비율 판단 단계**를 7개 에이전트로 자동화하는 개인 프로젝트입니다.

전체 기획 배경, 시장조사, BM, 데이터 전략은 `docs/` 및 기획서(.docx)를 참고하세요.
이 저장소는 기획서의 "시스템 구성 기획" 파트를 실제 코드로 구현합니다.

## 진행 일정 (풀 스코프, 2026-09-07 ~ 2026-11-30, 총 12주)

| 주차 | 기간 | 목표 |
|---|---|---|
| 1~2주 | 9/7 ~ 9/20 | 공통 아키텍처 개발 (오늘 여기) |
| 3~4주 | 9/21 ~ 10/4 | 기준표 데이터 구조화 + A/B차량측 에이전트 |
| 5주 | 10/5 ~ 10/11 | 사실검증 + 라운드제 토론 |
| 6~7주 | 10/12 ~ 10/25 | 검토위원 3종 구현 |
| 8주 | 10/26 ~ 11/1 | 취합·리포트 + 프론트엔드 결합 |
| 9~10주 | 11/2 ~ 11/15 | 블랙박스 이미지 인식 기능 |
| 11주 | 11/16 ~ 11/22 | 테스트 시나리오 검증 |
| 12주 | 11/23 ~ 11/30 | 마무리 및 데모 준비 |

## 오늘(9/7) 진행한 것

- 프로젝트 폴더 구조 확정 (`backend/`, `frontend/`, `docs/`)
- 백엔드: FastAPI 기반 오케스트레이션 서버 골격
  - 사고 정황 입력/에이전트 논증/검토위원 평가/최종 리포트에 대한 Pydantic 스키마 정의
  - 7개 에이전트의 베이스 클래스 + 각 에이전트 스텁 (역할 프롬프트는 3~4주차부터 실제 작성)
  - LLM 호출을 감싸는 `LLMClient` (API 키가 없으면 목(mock) 응답을 반환해, 프롬프트 없이도 파이프라인 전체를 오늘 바로 실행/테스트 가능)
  - `FaultAssessmentPipeline`: 1차 논증(A/B 병렬) → 사실검증 → 2라운드 반박 → 검토위원 3종 병렬 평가 → 진행 에이전트 집계까지 이어지는 라운드 흐름의 뼈대
  - `/health`, `/assess` API 엔드포인트
- 프론트엔드: 사고 정황 입력 폼 뼈대 (사고 유형/신호/차선/속도/설명 입력 → `/assess` 호출 → 결과 JSON 표시), 빌드 도구 없이 순수 HTML/JS
- 과실비율 기준표 참조용 데이터 파일 스텁 (`backend/app/data/fault_ratio_reference.json`) — 3~4주차에 실제 기준표로 채울 예정

## 실행 방법

### 백엔드

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # 필요 시 ANTHROPIC_API_KEY 입력 (없어도 mock 모드로 동작)
uvicorn app.main:app --reload --port 8000
```

브라우저에서 http://localhost:8000/health 로 동작 확인.

### 프론트엔드

`frontend/index.html` 파일을 브라우저로 바로 열면 됩니다 (백엔드가 `localhost:8000`에서 실행 중이어야 함).

## 오늘(9/7) 검증 완료

- [x] `pip install -r requirements.txt` 로 의존성 설치 확인 (venv 대신 `--user` 설치로 진행 — 이 환경에서 venv 생성이 불안정했음. 로컬 개발 시엔 README 상단 방법대로 venv 사용 권장)
- [x] mock 모드로 파이프라인 end-to-end 실행 확인 (`asyncio`로 직접 호출 + `uvicorn`으로 실제 HTTP 요청 둘 다 테스트, `/assess` 정상 응답)
- [x] git 저장소 초기 커밋 완료

## 과실비율 기준표 데이터 수집 (완료: 차 대 차 108개 도표)

`accident.knia.or.kr` (과실비율정보포털)에 공식 API는 없지만, 도표 상세 페이지
(`/myaccident-content?chartNo=차51-1&chartType=1`)가 서버 렌더링 HTML이라 chartNo로 직접 접근해 파싱할 수 있습니다.
(10/1: 마법사 수작업 순회 대신 이 방식으로 전환 → 가감요소·법규·판례까지 한 번에 확보)

- 수집 스크립트: `backend/scripts/scrape_knia_charts.py` (`pip install beautifulsoup4 lxml` 후 실행, 재실행 시 `chart_catalog` 전체 갱신)
- 이 서버는 중간 인증서 체인이 불완전해 Python 기본 SSL 검증이 실패하므로 OS 인증서를 쓰는 `curl`로 수신 (검증은 끄지 않음)
- 결과: `backend/app/data/fault_ratio_reference.json` → `chart_catalog` (차1~차61, 108개)
  - `situation` / `applicability` / `base_rationale`: 사고 상황·적용 범위·기본 과실 해설
  - `variants[]`: 기본 과실 (가)(나)(다) 경우별 비율 + 경우별 가감요소(`modifiers`: side/condition/adjustment). 12개 도표가 다중 variant
  - `modifier_notes` / `laws` / `precedents`: 수정요소 해설, 관련 법규, 판례·조정사례
- 검증: 모든 variant의 A+B=100, 108개 전부 가감요소 보유. 기존 수기 3건(차2-2/43-2/41-1)과 일치.
  기존 차33-1/33-2는 variant 하나만 담고 있었던 것을 바로잡음 (유턴: 상시유턴구역 20:80 / 신호유턴 100:0)
- 범위 밖: 자동차 대 보행자 / 자전거 도표 (필요 시 같은 방식으로 확장)

## 다음 할 일

- [x] 기준표 조회 모듈 (`core/chart_lookup.py`): 사고 입력 → 후보 도표 상위 3개(2-gram+IDF 매칭) → A/B·사실검증 에이전트 프롬프트에 주입. A/B 에이전트 실제 시스템 프롬프트 작성(`AdvocateAgent`). `cd backend && python -m pytest tests`
- [ ] 실제 ANTHROPIC_API_KEY로 A/B 논증 품질 확인 + 프롬프트 튜닝 (조회 정확도: 테스트 사례 상위 3개 안 적중, 1위는 약 절반 → 부족하면 LLM 기반 도표 선택으로 보강)

- [ ] 프론트엔드 실시간 공방 뷰(라운드별 진행 상태 표시) 최소 버전 붙이기
- [ ] `.env` 채워서 실제 ANTHROPIC_API_KEY 연동 경로도 한 번 확인 (지금은 mock 경로만 검증됨)
