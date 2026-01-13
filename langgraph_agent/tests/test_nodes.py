#!/usr/bin/env python3
"""
노드 테스트

LangGraph 노드 기능을 테스트합니다.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph_agent.nodes import AgentNodes, should_continue, should_continue_after_tool
from langgraph_agent.state import AgentState, create_initial_state
from langgraph_agent.config import AgentConfig


class TestParsingFunctions:
    """파싱 함수 테스트"""

    def setup_method(self):
        """테스트 설정"""
        from langgraph_agent.config import LLMConfig
        from langgraph_agent.llm.openai_client import OpenAICompatibleClient
        from langgraph_agent.tools.base import ToolRegistry

        config = AgentConfig()
        llm_client = OpenAICompatibleClient(config.llm)
        tool_registry = ToolRegistry()

        self.nodes = AgentNodes(llm_client, tool_registry, config)

    def test_parse_tool_calls_single(self):
        """단일 도구 호출 파싱"""
        content = """
<think>Let me search for relevant documents.</think>
<tool_call>
{"name": "retrieve", "arguments": {"query": "API design"}}
</tool_call>
"""
        tool_calls = self.nodes._parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "retrieve"
        assert tool_calls[0]["arguments"]["query"] == "API design"

    def test_parse_tool_calls_multiple(self):
        """다중 도구 호출 파싱"""
        content = """
<tool_call>
{"name": "retrieve", "arguments": {"query": "first query"}}
</tool_call>
<tool_call>
{"name": "confluence_search", "arguments": {"query": "second query"}}
</tool_call>
"""
        tool_calls = self.nodes._parse_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "retrieve"
        assert tool_calls[1]["name"] == "confluence_search"

    def test_parse_tool_calls_none(self):
        """도구 호출 없음"""
        content = "Just a regular response without any tool calls."
        tool_calls = self.nodes._parse_tool_calls(content)

        assert len(tool_calls) == 0

    def test_parse_answer(self):
        """답변 파싱"""
        content = """
<think>Based on the information gathered...</think>
<answer>
This is the final answer to your question.
It can span multiple lines.
</answer>
"""
        answer = self.nodes._parse_answer(content)

        assert answer is not None
        assert "final answer" in answer
        assert "multiple lines" in answer

    def test_parse_answer_none(self):
        """답변 없음"""
        content = "Response without answer tags."
        answer = self.nodes._parse_answer(content)

        assert answer is None


class TestRoutingFunctions:
    """라우팅 함수 테스트"""

    def test_should_continue_to_end(self):
        """완료 시 end로 라우팅"""
        state = AgentState(
            messages=[],
            question="test",
            current_turn=1,
            max_turns=100,
            start_time=0,
            total_tokens=0,
            pending_tool_calls=[],
            last_response="",
            answer="Final answer",
            termination_reason="answer_found",
            is_complete=True,
            metadata={}
        )

        result = should_continue(state)
        assert result == "end"

    def test_should_continue_to_tool(self):
        """도구 호출 시 tool로 라우팅"""
        state = AgentState(
            messages=[],
            question="test",
            current_turn=1,
            max_turns=100,
            start_time=0,
            total_tokens=0,
            pending_tool_calls=[{"name": "retrieve", "arguments": {}}],
            last_response="",
            answer=None,
            termination_reason=None,
            is_complete=False,
            metadata={}
        )

        result = should_continue(state)
        assert result == "tool"

    def test_should_continue_to_agent(self):
        """기본 agent로 라우팅"""
        state = AgentState(
            messages=[],
            question="test",
            current_turn=1,
            max_turns=100,
            start_time=0,
            total_tokens=0,
            pending_tool_calls=[],
            last_response="",
            answer=None,
            termination_reason=None,
            is_complete=False,
            metadata={}
        )

        result = should_continue(state)
        assert result == "agent"

    def test_should_continue_after_tool_to_agent(self):
        """도구 실행 후 agent로 라우팅"""
        state = AgentState(
            messages=[],
            question="test",
            current_turn=1,
            max_turns=100,
            start_time=0,
            total_tokens=0,
            pending_tool_calls=[],
            last_response="",
            answer=None,
            termination_reason=None,
            is_complete=False,
            metadata={}
        )

        result = should_continue_after_tool(state)
        assert result == "agent"


class TestStateCreation:
    """상태 생성 테스트"""

    def test_create_initial_state(self):
        """초기 상태 생성"""
        question = "테스트 질문입니다."
        system_prompt = "You are a helpful assistant."

        state = create_initial_state(question, system_prompt, max_turns=50)

        assert state["question"] == question
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "system"
        assert state["messages"][1]["role"] == "user"
        assert state["current_turn"] == 0
        assert state["max_turns"] == 50
        assert state["is_complete"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
