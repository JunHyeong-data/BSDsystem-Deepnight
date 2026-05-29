# AI 코딩 도구 활용 회고 — BSDsystem-Deepnight

> 본 문서는 capstone 진행 중 AI 코딩 도구를 어떻게 활용했는지, 무엇을 직접 판단했는지,
> 검증은 어떻게 했는지를 정리한 회고. 다른 학생/연구자가 비슷한 도구를 도입할 때 참고용.

---

## 1. 사용 도구

- **Claude Code** (Anthropic) — 터미널 통합 페어 프로그래밍 AI
  - 워크플로: 자연어로 작업 요청 → AI 가 파일 읽기·수정·실행 도구를 직접 호출 → 본인이 결과 검토·승인
  - 메모리 시스템에 프로젝트 컨텍스트 (좌표계, 학습 상태, 환경 quirk) 저장 → 세션 간 일관된 컨텍스트 유지
- **GitHub Copilot** (선택) — IDE inline 자동완성. 단순 보일러플레이트 한정

## 2. 셋업 방법

```bash
# Claude Code CLI 설치 후 프로젝트 루트에서
cd ~/BSDSystem
claude  # 인터랙티브 세션 시작

# 세션 내 주요 명령:
#   /init   — CLAUDE.md 생성 (프로젝트 컨벤션 문서)
#   /review — 현재 변경분 코드 리뷰
#   /fast   — 더 빠른 응답 모드 (Opus)
```

프로젝트 루트의 [`CLAUDE.md`](../CLAUDE.md) 에 4가지 행동 가이드 (Think Before Coding / Simplicity First /
Surgical Changes / Goal-Driven Execution) 를 작성 → AI 가 매 응답마다 이 원칙을 따름.
"불필요한 추상화 / 미요청 기능 추가 / 인접 코드 무단 수정" 같은 LLM 의 흔한 실패 패턴 억제.

## 3. AI 에 위임한 작업

| 카테고리 | 예시 |
|---|---|
| **반복 보일러플레이트** | argparse 인자 추가, dataclass 필드 정의, docstring 통일 |
| **수식 코딩** | fisheye equidistant 역투영 (`r = f·θ`), 회전행렬 합성. 본인이 수식 검증 후 채택 |
| **다중 파일 일관성** | README ↔ 코드 ↔ config 의 수치/문구 불일치 탐지. 예: "lateral 2.5 m" 주석 vs yaml `lateral_max: 3.5` |
| **버그 후보 식별** | `zip(detections, track_ids)` 의 length-mismatch 위험, dead code (`CoordBSDInterface`) 위치 등 |
| **리팩토링 (작은 범위)** | demo_tracker 의 inline 3-stage 로직을 `BSDInterface` 로 승격 |
| **명령어/스크립트 작성** | 평가 스크립트 (`evaluate_day_vs_night.py`), git workflow |

## 4. 본인이 직접 판단한 영역 (AI 위임 불가)

- **연구 방향 / 문제 정의** — "BSD scope에 보행자를 포함시킬지", "SGLDet 을 ablation 으로 negative 입증 vs 제거"
- **아키텍처 결정** — Part A (학습) / Part B (추론) 분리, ROS2 vs main.py 동시 지원 구조
- **물리·도메인 파라미터** — BSD zone 거리 (`lateral_min/max`, `forward_max`, `rear_max`), 카메라 mount 위치, MORAI 좌표계 의 좌수계 특성
- **학습 전략** — Plain FT + Scenario-level split 채택 (ablation 결과 기반)
- **교수님 피드백 수용/거절** — pedestrian class 제거 vs 유지, day-vs-night 비교 우선순위
- **scope 정직화** — README 에 limitation 명시할 것인지, 어느 정도 인정할 것인지

> **핵심 구분**: AI 는 **"어떻게(How)"** 의 답을 잘 내지만, **"무엇을(What)·왜(Why)"** 는 본인이 정해야 함.

## 5. 검증 방법

AI 가 생성한 코드는 *항상* 다음 4단계를 거침:

1. **읽고 의미 파악** — 단순 복붙 금지. 본인이 한 줄씩 의도를 이해 못 하면 채택 안 함.
2. **물리적 sanity check** — 좌표 변환은 (Z=0 평면 교차점, 광축 방향) 같은 hand-crafted 테스트.
3. **실데이터 동작 확인** — toy input 대신 실제 MORAI / 학습 데이터로 돌려서 결과 확인. 예: velocity 로직 추가 후 가상 시나리오 (dX/dt = +0.59 → DANGER) 로 행동 검증.
4. **컨텍스트 누락 인지** — AI 가 모르는 프로젝트 quirk (예: MORAI 의 좌수계, `cv2.Rodrigues` 미사용 이유) 는 본인이 명시적으로 알려주고 메모리에 박아 둠.

## 6. 한계와 주의사항

- **"되는 것 같다" 의 함정** — AI 가 자신 있게 틀리는 경우 있음 (특히 컨텍스트 가정이 어긋날 때). 항상 본인이 실데이터로 동작 확인.
- **AI 의 보수성 vs 본인의 판단** — AI 는 안전한 fallback 추가, 광범위한 error handling 을 선호. CLAUDE.md 의 "Simplicity First" 로 이를 명시적으로 억제.
- **컨텍스트 유실** — 세션이 길어지면 초기 결정/제약을 잊을 수 있음. 메모리 시스템에 핵심 사실 (좌표계, BSD zone, 학습 상태) 저장으로 보완.
- **연구 방향 결정 불가** — "어느 데이터를 더 모을지", "어느 가설을 검증할지" 는 AI 가 모름. 본인이 결정 후 AI 에게 실행을 위임하는 게 옳음.

## 7. 본 capstone 에서의 실제 사용 비율 (감각)

| 작업 유형 | 비율 (대략) | 본인 vs AI 비중 |
|---|---|---|
| 코드 작성 (스크립트, 모듈) | 60% | 본인 40 / AI 60 (AI 가 초안, 본인이 검토·수정) |
| 디버깅 | 15% | 본인 60 / AI 40 (root cause 찾기는 본인, fix 적용은 AI) |
| 리팩토링 / 코드 정리 | 10% | 본인 30 / AI 70 |
| 문서화 (README, docstring) | 10% | 본인 50 / AI 50 |
| 의사결정 / 연구 방향 | 5% | 본인 100 / AI 0 (AI 는 옵션 제시까지만) |

## 8. 한 줄 요약

> AI 는 **숙련된 페어 프로그래머**처럼 다룰 것 — 무엇을·왜는 본인이 정하고, 어떻게의 초안을 위임하되 모든 결과는 본인이 검증·책임진다.
