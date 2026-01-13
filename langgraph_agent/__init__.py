"""
LangGraph 기반 DeepResearch Agent

이 모듈은 Alibaba DeepResearch의 아키텍처를 LangGraph 형태로 재구현한 것입니다.
OpenAI 호환 API를 직접 사용하며, MCP 형태의 retrieve/confluence search 도구를 지원합니다.

주요 기능:
- LangGraph StateGraph 기반 ReAct 에이전트
- OpenAI 호환 API 직접 호출 (langchain-openai 미사용)
- MCP 스타일 도구 (retrieve, confluence_search)
- IterResearch (Heavy) 모드: Test-Time Scaling 전략

사용 예시:
```python
# 기본 연구 실행
from langgraph_agent import run_research
result = run_research("LLM 에이전트 개발 가이드는?")
print(result.answer)

# IterResearch 모드 (다중 롤아웃)
from langgraph_agent.iter_research import run_iter_research
results = run_iter_research("질문", rollout_count=3)
```
"""

from langgraph_agent.graph import (
    create_research_graph,
    run_research,
    arun_research,
    ResearchAgent,
)
from langgraph_agent.config import AgentConfig, LLMConfig, MCPConfig

# IterResearch 모듈
from langgraph_agent.iter_research import (
    IterResearchConfig,
    run_iter_research,
    arun_iter_research,
    MultiRolloutExecutor,
    MetricsCalculator,
    EvaluationMetrics,
)

__version__ = "0.2.0"

__all__ = [
    # 기본 연구
    "create_research_graph",
    "run_research",
    "arun_research",
    "ResearchAgent",

    # 설정
    "AgentConfig",
    "LLMConfig",
    "MCPConfig",

    # IterResearch
    "IterResearchConfig",
    "run_iter_research",
    "arun_iter_research",
    "MultiRolloutExecutor",
    "MetricsCalculator",
    "EvaluationMetrics",
]
