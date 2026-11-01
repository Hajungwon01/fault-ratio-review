# 아키텍처 개요

## 파이프라인 흐름

1. 사고 당사자가 사고 정황 입력 (`AccidentInput`) — 지원 범위(차 대 차) 밖이면 중단
2. `VehicleAAgent` / `VehicleBAgent` 가 병렬로 1차 논증 생성
3. `FactCheckerAgent` 가 양측 인용 기준표·법조항 적용 검증
4. A/B 에이전트 2라운드 반박
5. `ReviewerStandardAgent` / `ReviewerSceneAgent` / `ReviewerPrecedentAgent` 가 병렬로 독립 평가
6. `ProceedingAgent`(진행 에이전트, 오케스트레이터 역할)가 결과를 취합해
   "몇 대 몇" `FaultReport` 생성(위원 중앙값) + 위원 간 편차가 크거나 다른 도표를 적용하면 `needs_precise_review` 플래그
7. 위원이 모호함 신호를 보이면 `plan_clarification` 이 정황에 없는 사실 질문 1개를 만들고, 정황과 모순되는 보기를 점검 호출로 제외
8. 사용자용 흐름(`AssessmentService`): 질문이 필요하면 결과·도표·과실을 숨긴 채 질문만 반환, 답변을 받아 범위와 근거를 반환 (자세히: `docs/USER_API_DESIGN.md`)

## 코드 구조

```
backend/app/
  main.py            FastAPI 엔트리포인트 (/health, /v1/assess, /v1/assess/{id}/answer, 내부용 /assess)
  config.py          환경 설정 (.env 로드)
  models/schemas.py  Pydantic 스키마 (입력/논증/평가/리포트)
  core/llm_client.py LLM 호출 래퍼 (키 없으면 mock 응답)
  core/tooling.py      프로바이더 중립 도구 클래스 (OpenAI/Anthropic 형식으로 변환)
  core/chart_tools.py  에이전트용 도구(search_charts/get_chart) — 도구 호출 루프에서 호출
  core/chart_lookup.py 기준표 조회 (사고 입력 → 후보 도표 → 프롬프트용 요약)
  core/pipeline.py   라운드 진행 로직 (FaultAssessmentPipeline)
  agents/base.py     BaseAgent, ReviewerAgent 공통 클래스
  agents/*.py        7개 에이전트 구현체와 진행 에이전트(질문 계획 포함)
  core/clarify.py, sessions.py, user_flow.py, user_summary.py, user_text.py  사용자용 질문·세션·설명 생성과 검증
  core/verdict_check.py, citation_check.py, scope_check.py  위원 값·인용·지원 범위 검증 코드
  evaluation/        시나리오 평가와 포털식 baseline
  data/fault_ratio_reference.json  과실비율 기준표 (차 대 차 108개 도표, scripts/scrape_knia_charts.py 로 수집)
```

## 에이전트 실행 방식 (도구 호출 루프)

- LLM 프로바이더는 `LLM_PROVIDER`(openai 기본 | anthropic)로 고른다. `LLMClient`가 프로바이더별 백엔드를 감싸고, mock 처리와 구조화 출력의 검증·재시도는 공통으로 처리한다.
  OpenAI는 Responses API(함수 호출 루프, `text.format` json_schema strict), Anthropic은 SDK Tool Runner를 쓴다.
- 논증(A/B)·사실검증 에이전트는 `LLMClient.run_agent()` 로 **도구 호출 루프**를 돈다.
  모델이 `search_charts`/`get_chart` 를 스스로 호출해 기준표를 확인하고, 더 이상 도구를 부르지 않으면 종료한다.
  파이프라인이 미리 넣어주는 후보 도표는 "힌트"일 뿐이고, 인용 근거는 실제 조회한 도표여야 한다.
- 도구 호출 기록은 `AgentArgument.tool_calls` 에 남는다 (공방 뷰는 Stretch 로 미구현).
- 라운드 순서·병렬 실행(`asyncio.gather`)은 여전히 코드(`FaultAssessmentPipeline`)가 통제한다.
  위원/진행 에이전트는 현재 도구 없는 단발 호출(`complete`).
- 에이전트가 서로를 호출하는 자율 오케스트레이션(Managed Agents multiagent 등)은 도입하지 않았다 — 심사 흐름이 고정이라
  코드 통제가 재현성·디버깅에 유리하다. 필요해지면 재검토.

## 설계 원칙

- **에이전트는 얇게, 파이프라인이 흐름을 통제한다.** 각 에이전트는 `run(context) -> dict` 형태의
  단일 진입점만 가지며, 라운드 순서·병렬 실행은 `FaultAssessmentPipeline`이 담당한다.
- **LLM 키 없이도 전체 파이프라인이 돌아가야 한다.** `LLMClient`가 mock 모드를 지원하므로,
  프롬프트를 아직 안 짰거나 API 키가 없어도 구조적으로 끝까지 실행/테스트할 수 있다.
- **이미지 인식은 별도 에이전트가 아니라 입력 전처리로 취급한다.** (Stretch, 미구현 — #22) 블랙박스 이미지가
  들어오면, 비전 지원 LLM으로 상황을 텍스트 요약한 뒤 기존 `AccidentInput.description`에
  합쳐 넣는 방식으로 확장한다 (신규 CV 모델 학습 없음).
