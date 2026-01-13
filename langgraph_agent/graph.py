"""
LangGraph 워크플로우 정의

DeepResearch 에이전트의 실행 그래프를 구성합니다.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent.state import AgentState, ResearchResult, create_initial_state
from langgraph_agent.nodes import AgentNodes, should_continue, should_continue_after_tool
from langgraph_agent.llm.openai_client import OpenAICompatibleClient
from langgraph_agent.tools.base import ToolRegistry
from langgraph_agent.tools.retrieve import RetrieveTool, RetrieveByIdTool
from langgraph_agent.tools.confluence import (
    ConfluenceSearchTool,
    ConfluenceGetPageTool,
    ConfluenceListSpacesTool
)
from langgraph_agent.config import AgentConfig
from langgraph_agent.prompts import build_system_prompt

logger = logging.getLogger(__name__)


def create_tool_registry() -> ToolRegistry:
    """
    기본 도구 레지스트리 생성

    Returns:
        초기화된 도구 레지스트리
    """
    registry = ToolRegistry()

    # MCP Mock 도구 등록
    registry.register(RetrieveTool())
    registry.register(RetrieveByIdTool())
    registry.register(ConfluenceSearchTool())
    registry.register(ConfluenceGetPageTool())
    registry.register(ConfluenceListSpacesTool())

    logger.info(f"도구 레지스트리 초기화: {registry.list_tools()}")

    return registry


def create_research_graph(
    config: Optional[AgentConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None
) -> StateGraph:
    """
    연구 에이전트 그래프 생성

    Args:
        config: 에이전트 설정 (None이면 기본값 사용)
        tool_registry: 도구 레지스트리 (None이면 기본 도구 사용)
        checkpointer: LangGraph 체크포인터 (상태 저장용)

    Returns:
        컴파일된 StateGraph
    """
    # 설정 초기화
    if config is None:
        config = AgentConfig()

    # 도구 레지스트리 초기화
    if tool_registry is None:
        tool_registry = create_tool_registry()

    # LLM 클라이언트 초기화
    llm_client = OpenAICompatibleClient(config.llm)

    # 노드 객체 생성
    nodes = AgentNodes(llm_client, tool_registry, config)

    # 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("tool", nodes.tool_node)

    # 엣지 추가
    # 시작점 -> agent
    workflow.set_entry_point("agent")

    # agent -> (조건부) -> tool / end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool": "tool",
            "agent": "agent",  # 도구 호출 없이 계속
            "end": END
        }
    )

    # tool -> (조건부) -> agent / end
    workflow.add_conditional_edges(
        "tool",
        should_continue_after_tool,
        {
            "agent": "agent",
            "end": END
        }
    )

    # 그래프 컴파일
    if checkpointer:
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()

    logger.info("연구 에이전트 그래프 생성 완료")

    return compiled


def run_research(
    question: str,
    config: Optional[AgentConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True
) -> ResearchResult:
    """
    연구 실행

    Args:
        question: 사용자 질문
        config: 에이전트 설정
        tool_registry: 도구 레지스트리
        verbose: 상세 출력 여부

    Returns:
        ResearchResult: 연구 결과
    """
    import time

    # 설정 초기화
    if config is None:
        config = AgentConfig()

    # 도구 레지스트리
    if tool_registry is None:
        tool_registry = create_tool_registry()

    # 시스템 프롬프트 생성
    tools = list(tool_registry.get_all().values())
    system_prompt = build_system_prompt(tools)

    # 초기 상태 생성
    initial_state = create_initial_state(
        question=question,
        system_prompt=system_prompt,
        max_turns=config.max_turns
    )

    # 그래프 생성 및 실행
    graph = create_research_graph(config, tool_registry)

    if verbose:
        logger.info(f"연구 시작: {question[:100]}...")

    # 그래프 실행
    start_time = time.time()
    final_state = None

    try:
        for step in graph.stream(initial_state):
            if verbose:
                for node_name, node_state in step.items():
                    if node_name != "__end__":
                        turn = node_state.get("current_turn", "?")
                        logger.info(f"[Step] 노드: {node_name}, 턴: {turn}")

            # 마지막 상태 저장
            for node_name, node_state in step.items():
                if node_name != "__end__":
                    # 상태 병합
                    if final_state is None:
                        final_state = dict(initial_state)
                    for key, value in node_state.items():
                        if key == "messages":
                            final_state["messages"] = final_state.get("messages", []) + value
                        else:
                            final_state[key] = value

    except Exception as e:
        logger.error(f"연구 실행 오류: {e}")
        if final_state is None:
            final_state = dict(initial_state)
        final_state["termination_reason"] = f"error: {str(e)}"
        final_state["is_complete"] = True

    execution_time = time.time() - start_time

    if verbose:
        logger.info(f"연구 완료: {execution_time:.1f}초, 종료사유: {final_state.get('termination_reason', 'unknown')}")

    # 결과 생성
    result = ResearchResult(
        question=question,
        answer=final_state.get("answer", ""),
        messages=final_state.get("messages", []),
        termination_reason=final_state.get("termination_reason", "unknown"),
        total_turns=final_state.get("current_turn", 0),
        total_tokens=final_state.get("total_tokens", 0),
        execution_time=execution_time,
        metadata=final_state.get("metadata", {})
    )

    return result


async def arun_research(
    question: str,
    config: Optional[AgentConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True
) -> ResearchResult:
    """
    비동기 연구 실행

    Args:
        question: 사용자 질문
        config: 에이전트 설정
        tool_registry: 도구 레지스트리
        verbose: 상세 출력 여부

    Returns:
        ResearchResult: 연구 결과
    """
    import asyncio
    import time

    # 설정 초기화
    if config is None:
        config = AgentConfig()

    if tool_registry is None:
        tool_registry = create_tool_registry()

    # 시스템 프롬프트 생성
    tools = list(tool_registry.get_all().values())
    system_prompt = build_system_prompt(tools)

    # 초기 상태 생성
    initial_state = create_initial_state(
        question=question,
        system_prompt=system_prompt,
        max_turns=config.max_turns
    )

    # 그래프 생성
    graph = create_research_graph(config, tool_registry)

    if verbose:
        logger.info(f"[Async] 연구 시작: {question[:100]}...")

    start_time = time.time()
    final_state = None

    try:
        async for step in graph.astream(initial_state):
            if verbose:
                for node_name, node_state in step.items():
                    if node_name != "__end__":
                        turn = node_state.get("current_turn", "?")
                        logger.info(f"[Async Step] 노드: {node_name}, 턴: {turn}")

            for node_name, node_state in step.items():
                if node_name != "__end__":
                    if final_state is None:
                        final_state = dict(initial_state)
                    for key, value in node_state.items():
                        if key == "messages":
                            final_state["messages"] = final_state.get("messages", []) + value
                        else:
                            final_state[key] = value

    except Exception as e:
        logger.error(f"[Async] 연구 실행 오류: {e}")
        if final_state is None:
            final_state = dict(initial_state)
        final_state["termination_reason"] = f"error: {str(e)}"
        final_state["is_complete"] = True

    execution_time = time.time() - start_time

    if verbose:
        logger.info(f"[Async] 연구 완료: {execution_time:.1f}초")

    result = ResearchResult(
        question=question,
        answer=final_state.get("answer", ""),
        messages=final_state.get("messages", []),
        termination_reason=final_state.get("termination_reason", "unknown"),
        total_turns=final_state.get("current_turn", 0),
        total_tokens=final_state.get("total_tokens", 0),
        execution_time=execution_time,
        metadata=final_state.get("metadata", {})
    )

    return result


class ResearchAgent:
    """
    연구 에이전트 클래스

    편리한 사용을 위한 래퍼 클래스입니다.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        에이전트 초기화

        Args:
            config: 에이전트 설정
            tool_registry: 도구 레지스트리
        """
        self.config = config or AgentConfig()
        self.tool_registry = tool_registry or create_tool_registry()
        self.graph = create_research_graph(self.config, self.tool_registry)

        # 통계
        self.total_runs = 0
        self.successful_runs = 0

    def run(self, question: str, verbose: bool = True) -> ResearchResult:
        """
        동기 연구 실행

        Args:
            question: 사용자 질문
            verbose: 상세 출력 여부

        Returns:
            ResearchResult: 연구 결과
        """
        self.total_runs += 1

        result = run_research(
            question=question,
            config=self.config,
            tool_registry=self.tool_registry,
            verbose=verbose
        )

        if result.termination_reason == "answer_found":
            self.successful_runs += 1

        return result

    async def arun(self, question: str, verbose: bool = True) -> ResearchResult:
        """
        비동기 연구 실행

        Args:
            question: 사용자 질문
            verbose: 상세 출력 여부

        Returns:
            ResearchResult: 연구 결과
        """
        self.total_runs += 1

        result = await arun_research(
            question=question,
            config=self.config,
            tool_registry=self.tool_registry,
            verbose=verbose
        )

        if result.termination_reason == "answer_found":
            self.successful_runs += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        통계 반환

        Returns:
            통계 정보 딕셔너리
        """
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": self.successful_runs / self.total_runs if self.total_runs > 0 else 0,
            "available_tools": self.tool_registry.list_tools()
        }

    def add_tool(self, tool) -> None:
        """
        도구 추가

        Args:
            tool: 추가할 도구 인스턴스
        """
        self.tool_registry.register(tool)
        # 그래프 재생성
        self.graph = create_research_graph(self.config, self.tool_registry)
