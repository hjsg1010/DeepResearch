"""
IterResearch (Heavy) 모드 모듈

Test-Time Scaling 전략을 통해 모델의 최대 성능을 이끌어내는
IterResearch 기능을 제공합니다.

주요 기능:
- 다중 롤아웃 실행 (Multi-Rollout)
- 결과 집계 및 평가 (Pass@K, Best@1, Avg@K)
- PPL 기반 불확실성 분석
- 부분 롤아웃 (Partial Rollout)

사용 예시:
```python
from langgraph_agent.iter_research import (
    run_iter_research,
    IterResearchConfig,
    MetricsCalculator,
)

# 기본 IterResearch 실행
results = run_iter_research(
    question="LLM 에이전트 개발 가이드는?",
    rollout_count=3,
)

# 결과 평가
calculator = MetricsCalculator()
metrics = calculator.calculate({"question": results})
print(metrics)
```
"""

from langgraph_agent.iter_research.config import (
    IterResearchConfig,
    PartialSamplingMode,
    AggregationStrategy,
    RolloutResult,
    AggregatedResult,
    StepPPL,
)

from langgraph_agent.iter_research.rollout import (
    MultiRolloutExecutor,
    AsyncMultiRolloutExecutor,
    RolloutTask,
    run_iter_research,
    arun_iter_research,
)

from langgraph_agent.iter_research.aggregator import (
    ResultAggregator,
    MetricsCalculator,
    EvaluationMetrics,
    load_results_from_files,
    evaluate_iter_research_results,
)

from langgraph_agent.iter_research.entropy import (
    EntropyCalculator,
    UncertaintyDetector,
    TokenEntropy,
    StepUncertainty,
    compute_step_ppl_from_logprobs,
)

from langgraph_agent.iter_research.partial_rollout import (
    PartialRolloutExecutor,
    PartialRolloutRunner,
    PartialRolloutResult,
    BranchPoint,
    run_iter_research_with_partial,
)

__all__ = [
    # Config
    "IterResearchConfig",
    "PartialSamplingMode",
    "AggregationStrategy",
    "RolloutResult",
    "AggregatedResult",
    "StepPPL",

    # Rollout
    "MultiRolloutExecutor",
    "AsyncMultiRolloutExecutor",
    "RolloutTask",
    "run_iter_research",
    "arun_iter_research",

    # Aggregation & Metrics
    "ResultAggregator",
    "MetricsCalculator",
    "EvaluationMetrics",
    "load_results_from_files",
    "evaluate_iter_research_results",

    # Entropy & Uncertainty
    "EntropyCalculator",
    "UncertaintyDetector",
    "TokenEntropy",
    "StepUncertainty",
    "compute_step_ppl_from_logprobs",

    # Partial Rollout
    "PartialRolloutExecutor",
    "PartialRolloutRunner",
    "PartialRolloutResult",
    "BranchPoint",
    "run_iter_research_with_partial",
]
