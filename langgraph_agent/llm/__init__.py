"""
LLM 호출 모듈

OpenAI 호환 API를 직접 사용하는 클라이언트를 제공합니다.
"""

from langgraph_agent.llm.openai_client import OpenAICompatibleClient

__all__ = ["OpenAICompatibleClient"]
