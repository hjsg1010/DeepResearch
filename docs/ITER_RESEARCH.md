# IterResearch (Heavy) 모드 상세 문서

## 1. 개요

### 1.1 IterResearch란?

**IterResearch**는 Tongyi DeepResearch에서 제공하는 **Test-Time Scaling** 전략입니다. 동일한 질문에 대해 에이전트를 여러 번 독립적으로 실행하고, 결과를 집계하여 모델의 최대 성능을 이끌어냅니다.

이 접근법은 다음 원칙에 기반합니다:
- **다양성**: 여러 번의 독립 실행은 다양한 추론 경로를 탐색
- **신뢰성**: 여러 결과 중 일관된 답변이 더 신뢰할 수 있음
- **최적화**: 추론 시간에 더 많은 계산을 투입하여 성능 향상

### 1.2 Heavy 모드의 특징

README.md에서 언급된 바와 같이:
> "an IterResearch-based 'Heavy' mode, which uses a test-time scaling strategy to unlock the model's maximum performance ceiling"

**Heavy 모드의 핵심**:
1. 복수 실행 (Multiple Rollouts)
2. Test-Time Scaling
3. 결과 집계 및 평가

---

## 2. 이론적 배경

### 2.1 Test-Time Scaling

Test-Time Scaling은 모델 학습 후 추론 단계에서 추가 계산을 투입하여 성능을 향상시키는 기법입니다.

```
기존 방식:
  질문 → [단일 실행] → 답변

Test-Time Scaling:
  질문 → [실행1] → 답변1
       → [실행2] → 답변2  → [집계] → 최종 답변
       → [실행3] → 답변3
```

### 2.2 왜 다중 롤아웃이 효과적인가?

1. **탐색 공간 확장**: LLM은 확률적으로 토큰을 생성하므로, 여러 번 실행하면 다른 추론 경로를 탐색
2. **오류 완화**: 단일 실행의 실수를 여러 시도로 보완
3. **신뢰도 향상**: 일관된 답변이 여러 번 나오면 정답일 확률 높음

### 2.3 평가 메트릭

| 메트릭 | 설명 | 계산 방법 |
|--------|------|-----------|
| **Pass@K** | K번 시도 중 최소 1번 성공 비율 | `(성공한 질문 수) / (전체 질문 수)` |
| **Best@1** | 각 라운드 중 최고 성공률 | `max(라운드별 성공률)` |
| **Avg@K** | K개 라운드의 평균 성공률 | `mean(라운드별 성공률)` |

**예시**:
```
질문: "파이썬에서 리스트 정렬 방법은?"
- Round 1: 정답 ✓
- Round 2: 오답 ✗
- Round 3: 정답 ✓

Pass@1: 1.0 (첫 번째가 정답)
Pass@3: 1.0 (3번 중 1번 이상 정답)
Best@1: 1.0 (라운드별 최고 = 100%)
Avg@3: 0.67 (2/3 정답)
```

---

## 3. 아키텍처

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         IterResearch                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MultiRolloutExecutor                        │    │
│  │                                                          │    │
│  │  질문 ───┬─── Rollout 1 (iter1) ───┐                    │    │
│  │          ├─── Rollout 2 (iter2) ───┼─── 결과 수집       │    │
│  │          └─── Rollout 3 (iter3) ───┘                    │    │
│  │                                                          │    │
│  │  ThreadPoolExecutor / AsyncIO 기반 병렬 실행             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ResultAggregator                            │    │
│  │                                                          │    │
│  │  - FIRST_SUCCESS: 첫 번째 성공 선택                      │    │
│  │  - MAJORITY_VOTE: 다수결                                 │    │
│  │  - BEST_CONFIDENCE: 신뢰도 기반                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MetricsCalculator                           │    │
│  │                                                          │    │
│  │  - Pass@1, Pass@3                                        │    │
│  │  - Best@1, Avg@3                                         │    │
│  │  - 라운드별 성공률                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 부분 롤아웃 (Partial Rollout) - ParallelMuse 스타일

```
초기 롤아웃:
  [시작] → [Step 1] → [Step 2*] → [Step 3] → [Step 4*] → [답변]
                ↑ PPL 높음              ↑ PPL 높음

부분 롤아웃 (Step 2에서 분기):
  [시작] → [Step 1] → [새로운 Step 2'] → [Step 3'] → [답변']

부분 롤아웃 (Step 4에서 분기):
  [시작] → [Step 1] → [Step 2] → [Step 3] → [새로운 Step 4'] → [답변'']
```

**PPL (Perplexity) 기반 분기점 탐지**:
- `think_ppl`: 생각(Think) 영역의 불확실성
- `tool_call_ppl`: 도구 호출(Tool Call) 영역의 불확실성
- `all_ppl`: 전체 응답의 불확실성

---

## 4. 구현 상세

### 4.1 모듈 구조

```
langgraph_agent/iter_research/
├── __init__.py           # 모듈 진입점
├── config.py             # 설정 클래스
├── rollout.py            # 다중 롤아웃 실행기
├── aggregator.py         # 결과 집계 및 메트릭
├── entropy.py            # PPL/엔트로피 계산
└── partial_rollout.py    # 부분 롤아웃
```

### 4.2 주요 클래스

#### IterResearchConfig

```python
@dataclass
class IterResearchConfig:
    # 기본 롤아웃 설정
    rollout_count: int = 3           # 롤아웃 수
    max_workers: int = 20            # 병렬 워커 수

    # 부분 샘플링 설정
    partial_sampling_mode: PartialSamplingMode = PartialSamplingMode.NONE
    initial_rollout_num: int = 1     # 초기 롤아웃 수
    partial_sampling_topk: int = 2   # 분기점 개수
    partial_sampling_times_per_pos: int = 3  # 분기점당 샘플링 횟수

    # 결과 집계 설정
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FIRST_SUCCESS
```

#### MultiRolloutExecutor

```python
class MultiRolloutExecutor:
    def run_single_question(
        self,
        question: str,
        answer: str = "",
    ) -> List[RolloutResult]:
        """단일 질문에 대해 다중 롤아웃 실행"""

    def run_batch(
        self,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, List[RolloutResult]]:
        """배치 질문 처리"""
```

#### MetricsCalculator

```python
class MetricsCalculator:
    def calculate(
        self,
        results_by_question: Dict[str, List[RolloutResult]],
    ) -> EvaluationMetrics:
        """Pass@K, Best@1, Avg@K 계산"""
```

### 4.3 부분 샘플링 모드

| 모드 | 설명 |
|------|------|
| `NONE` | 부분 샘플링 비활성화 |
| `ALL_PPL` | 전체 응답 PPL 기반 |
| `THINK_PPL` | Think 영역 PPL 기반 |
| `TOOL_CALL_PPL` | Tool Call 영역 PPL 기반 |
| `MIXED_PPL` | Think + Tool Call 혼합 |

---

## 5. 사용법

### 5.1 기본 사용

```python
from langgraph_agent.iter_research import (
    run_iter_research,
    MetricsCalculator,
)

# 단일 질문에 대해 3회 롤아웃
results = run_iter_research(
    question="API 설계 가이드라인은 무엇인가요?",
    answer="",  # 정답 (평가용)
    rollout_count=3,
    max_workers=3,
)

# 결과 확인
for r in results:
    print(f"Rollout {r.rollout_idx}: {r.termination}")
    print(f"  예측: {r.prediction[:100]}...")
```

### 5.2 배치 처리

```python
from langgraph_agent.iter_research import (
    IterResearchConfig,
    MultiRolloutExecutor,
    MetricsCalculator,
)

# 설정
config = IterResearchConfig(
    rollout_count=3,
    max_workers=10,
    output_dir="./outputs",
)

# 실행기 생성
executor = MultiRolloutExecutor(config)

# 질문 리스트
questions = [
    {"question": "질문1", "answer": "정답1"},
    {"question": "질문2", "answer": "정답2"},
]

# 배치 실행
results = executor.run_batch(questions, output_prefix="experiment")

# 메트릭 계산
calculator = MetricsCalculator()
metrics = calculator.calculate(results)

print(metrics)
# === 평가 결과 ===
# Pass@1: 75.00%
# Pass@3: 90.00%
# Best@1: 80.00%
# Avg@3: 76.67%
```

### 5.3 부분 샘플링 (고급)

```python
from langgraph_agent.iter_research import (
    run_iter_research_with_partial,
    PartialSamplingMode,
)

# 부분 샘플링과 함께 실행
results = run_iter_research_with_partial(
    question="복잡한 연구 질문",
    answer="",
    initial_rollout_num=1,
    partial_sampling_mode=PartialSamplingMode.TOOL_CALL_PPL,
    partial_sampling_topk=2,
    partial_sampling_times_per_pos=3,
)

# 총 실행: 1 + (1 * 2 * 3) = 7개 롤아웃
print(f"총 롤아웃 수: {len(results)}")
```

### 5.4 비동기 실행

```python
import asyncio
from langgraph_agent.iter_research import arun_iter_research

async def main():
    results = await arun_iter_research(
        question="질문",
        rollout_count=3,
        max_workers=3,
    )
    return results

results = asyncio.run(main())
```

---

## 6. 원본 구현체와의 비교

### 6.1 run_multi_react.py (원본)

```python
# 원본 DeepResearch의 다중 롤아웃 실행
output_files = {i: f"iter{i}.jsonl" for i in range(1, roll_out_count + 1)}

for rollout_idx in range(1, roll_out_count + 1):
    for item in items:
        tasks_to_run_all.append({
            "item": item,
            "rollout_idx": rollout_idx,
        })

with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = {executor.submit(agent._run, task): task for task in tasks}
```

### 6.2 langgraph_agent 구현

```python
# LangGraph 기반 다중 롤아웃 실행
executor = MultiRolloutExecutor(config)
results = executor.run_single_question(question, answer)

# 또는 비동기
results = await AsyncMultiRolloutExecutor(config).run_single_question_async(question)
```

### 6.3 주요 차이점

| 항목 | 원본 (DeepResearch) | 본 구현 (langgraph_agent) |
|------|---------------------|--------------------------|
| 프레임워크 | qwen_agent 기반 | LangGraph 기반 |
| LLM 호출 | vLLM 서버 직접 연결 | OpenAI 호환 API |
| 도구 | Serper, Jina 등 외부 API | MCP Mock 도구 |
| 병렬화 | ThreadPoolExecutor | ThreadPoolExecutor + AsyncIO |
| 상태 관리 | 메시지 리스트 | LangGraph StateGraph |

---

## 7. 평가 시스템

### 7.1 evaluate_deepsearch_official.py 분석

원본 평가 스크립트의 핵심 로직:

```python
# 3개 라운드 파일 로드
round1_file = "iter1.jsonl"
round2_file = "iter2.jsonl"
round3_file = "iter3.jsonl"

# LLM Judge로 정답 판정
for item in items:
    judgement = call_llm_judge(item)

# 메트릭 계산
pass_at_3 = calculate_pass_at_k(results, k=3)
best_at_1 = calculate_best_pass_at_1(results)
avg_at_3 = calculate_avg_pass_at_3(results)
```

### 7.2 본 구현의 평가

```python
from langgraph_agent.iter_research import (
    evaluate_iter_research_results,
    MetricsCalculator,
)

# 파일에서 결과 로드 및 평가
file_paths = {1: "iter1.jsonl", 2: "iter2.jsonl", 3: "iter3.jsonl"}
metrics = evaluate_iter_research_results(file_paths)

print(f"Pass@3: {metrics.pass_at_3 * 100:.2f}%")
print(f"Best@1: {metrics.best_at_1 * 100:.2f}%")
print(f"Avg@3: {metrics.avg_at_3 * 100:.2f}%")
```

---

## 8. 설정 가이드

### 8.1 기본 IterResearch (권장)

```python
config = IterResearchConfig(
    rollout_count=3,
    max_workers=3,
    max_turns_per_rollout=100,
)
```

### 8.2 부분 샘플링 (고급)

```python
config = IterResearchConfig(
    rollout_count=1,
    initial_rollout_num=1,
    partial_sampling_mode=PartialSamplingMode.TOOL_CALL_PPL,
    partial_sampling_topk=2,
    partial_sampling_rounds=1,
    partial_sampling_times_per_pos=3,
    sampling_budget=7,  # 1 + 1*2*3 = 7
)
```

### 8.3 대규모 배치 처리

```python
config = IterResearchConfig(
    rollout_count=3,
    max_workers=20,
    output_dir="./large_batch_outputs",
    save_intermediate=True,
)
```

---

## 9. 제한 사항 및 주의사항

### 9.1 리소스 고려

- **CPU/메모리**: 병렬 롤아웃은 리소스를 많이 사용
- **API 호출**: 롤아웃 수 × 턴 수만큼 LLM 호출
- **시간**: 단일 실행 대비 N배 시간 소요

### 9.2 권장 사항

1. **롤아웃 수**: 3~5개가 적절 (너무 많으면 비효율)
2. **병렬 워커**: CPU 코어 수 × 2 이하
3. **부분 샘플링**: 초기 롤아웃 결과가 좋지 않을 때만 사용

### 9.3 현재 제한

- PPL 기반 분기점 탐지는 logprobs 지원이 필요
- 일부 OpenAI 호환 서버에서는 logprobs 미지원
- Mock 도구 사용으로 실제 검색 결과 없음

---

## 10. 향후 개선 계획

1. **실제 MCP 서버 연동**: Mock 도구를 실제 서비스로 교체
2. **LLM Judge 통합**: 자동 정답 판정 기능
3. **스트리밍 지원**: 실시간 진행 상황 표시
4. **분산 처리**: 여러 서버에서 병렬 실행
5. **캐싱**: 동일 질문에 대한 결과 재사용

---

## 11. 참고 자료

- [Tongyi DeepResearch Technical Report](https://arxiv.org/pdf/2510.24701)
- [ParallelMuse: Agentic Parallel Thinking](https://arxiv.org/pdf/2510.24698)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [원본 DeepResearch GitHub](https://github.com/Alibaba-NLP/DeepResearch)
