"""
LangGraph 기반 DeepResearch Agent

이 모듈은 Alibaba DeepResearch의 아키텍처를 LangGraph 형태로 재구현한 것입니다.
OpenAI 호환 API를 직접 사용하며, MCP 형태의 retrieve/confluence search 도구를 지원합니다.
"""

from langgraph_agent.graph import create_research_graph, run_research
from langgraph_agent.config import AgentConfig

__version__ = "0.1.0"
__all__ = ["create_research_graph", "run_research", "AgentConfig"]
