"""
LangGraph 노드 정의

각 노드는 상태를 받아서 상태 업데이트를 반환합니다.
"""

import json
import json5
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from langgraph_agent.state import AgentState, Message, ToolCall
from langgraph_agent.llm.openai_client import OpenAICompatibleClient
from langgraph_agent.tools.base import ToolRegistry, ToolResult
from langgraph_agent.config import AgentConfig
from langgraph_agent.prompts import CONTEXT_LIMIT_PROMPT

logger = logging.getLogger(__name__)


class AgentNodes:
    """
    LangGraph 에이전트 노드 모음

    각 메서드는 LangGraph 노드로 사용됩니다.
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        tool_registry: ToolRegistry,
        config: AgentConfig
    ):
        """
        노드 초기화

        Args:
            llm_client: LLM 클라이언트
            tool_registry: 도구 레지스트리
            config: 에이전트 설정
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config

    def _parse_tool_calls(self, content: str) -> List[ToolCall]:
        """
        LLM 응답에서 도구 호출 파싱

        Args:
            content: LLM 응답 내용

        Returns:
            파싱된 도구 호출 리스트
        """
        tool_calls = []

        # <tool_call>...</tool_call> 패턴 찾기
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            try:
                # json5를 사용하여 유연한 JSON 파싱
                parsed = json5.loads(match.strip())
                if "name" in parsed:
                    tool_calls.append(ToolCall(
                        name=parsed.get("name", ""),
                        arguments=parsed.get("arguments", {})
                    ))
            except Exception as e:
                logger.warning(f"도구 호출 파싱 실패: {e}, 내용: {match[:100]}")

        return tool_calls

    def _parse_answer(self, content: str) -> Optional[str]:
        """
        LLM 응답에서 최종 답변 파싱

        Args:
            content: LLM 응답 내용

        Returns:
            파싱된 답변 (없으면 None)
        """
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()
        return None

    def _convert_messages_for_api(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        메시지를 OpenAI API 형식으로 변환

        Args:
            messages: 내부 메시지 리스트

        Returns:
            API 호출용 메시지 리스트
        """
        api_messages = []
        for msg in messages:
            api_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            api_messages.append(api_msg)
        return api_messages

    def agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        에이전트 노드: LLM을 호출하여 다음 액션 결정

        Args:
            state: 현재 상태

        Returns:
            상태 업데이트
        """
        current_turn = state["current_turn"] + 1
        logger.info(f"[Agent] 턴 {current_turn} 시작")

        # 시간 초과 체크
        elapsed_time = time.time() - state["start_time"]
        if elapsed_time > self.config.max_execution_time:
            logger.warning(f"[Agent] 시간 초과: {elapsed_time:.0f}초")
            return {
                "is_complete": True,
                "termination_reason": "timeout",
                "answer": "시간 초과로 답변을 완료하지 못했습니다.",
                "current_turn": current_turn
            }

        # 턴 수 초과 체크
        if current_turn > state["max_turns"]:
            logger.warning(f"[Agent] 턴 수 초과: {current_turn}")
            return {
                "is_complete": True,
                "termination_reason": "max_turns_exceeded",
                "answer": "최대 턴 수 초과로 답변을 완료하지 못했습니다.",
                "current_turn": current_turn
            }

        # 컨텍스트 길이 체크
        messages = state["messages"]
        token_count = self.llm_client.count_messages_tokens(
            self._convert_messages_for_api(messages)
        )

        if token_count > self.config.max_tokens_context:
            logger.warning(f"[Agent] 컨텍스트 길이 초과: {token_count}")
            # 강제 답변 요청
            messages = messages + [{"role": "user", "content": CONTEXT_LIMIT_PROMPT}]

        # LLM 호출
        try:
            api_messages = self._convert_messages_for_api(messages)
            response = self.llm_client.chat(api_messages)
            content = response.content

            # 토큰 사용량 업데이트
            tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0

            logger.debug(f"[Agent] LLM 응답: {content[:200]}...")

        except Exception as e:
            logger.error(f"[Agent] LLM 호출 실패: {e}")
            return {
                "is_complete": True,
                "termination_reason": "llm_error",
                "answer": f"LLM 호출 중 오류 발생: {str(e)}",
                "current_turn": current_turn
            }

        # <tool_response>가 응답에 포함되어 있으면 제거 (stop sequence 관련)
        if "<tool_response>" in content:
            content = content.split("<tool_response>")[0]

        # 응답 메시지 추가
        new_message = Message(
            role="assistant",
            content=content
        )

        # 답변 파싱
        answer = self._parse_answer(content)
        if answer:
            logger.info(f"[Agent] 최종 답변 감지")
            return {
                "messages": [new_message],
                "last_response": content,
                "answer": answer,
                "is_complete": True,
                "termination_reason": "answer_found",
                "current_turn": current_turn,
                "total_tokens": state["total_tokens"] + tokens_used,
                "pending_tool_calls": []
            }

        # 도구 호출 파싱
        tool_calls = self._parse_tool_calls(content)
        if tool_calls:
            logger.info(f"[Agent] 도구 호출 감지: {[tc['name'] for tc in tool_calls]}")
            return {
                "messages": [new_message],
                "last_response": content,
                "pending_tool_calls": tool_calls,
                "current_turn": current_turn,
                "total_tokens": state["total_tokens"] + tokens_used
            }

        # 도구 호출도 답변도 없는 경우
        logger.warning("[Agent] 도구 호출이나 답변 없음")
        return {
            "messages": [new_message],
            "last_response": content,
            "pending_tool_calls": [],
            "current_turn": current_turn,
            "total_tokens": state["total_tokens"] + tokens_used
        }

    def tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        도구 노드: 대기 중인 도구 호출 실행

        Args:
            state: 현재 상태

        Returns:
            상태 업데이트
        """
        pending_calls = state.get("pending_tool_calls", [])

        if not pending_calls:
            logger.warning("[Tool] 실행할 도구 호출 없음")
            return {"pending_tool_calls": []}

        results = []
        for tool_call in pending_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            logger.info(f"[Tool] 도구 실행: {tool_name}")
            logger.debug(f"[Tool] 인자: {tool_args}")

            # 도구 실행
            result = self.tool_registry.call_tool(tool_name, tool_args)

            if result.success:
                result_content = result.content
            else:
                result_content = f"[Error] {result.error}"

            results.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result_content
            })

            logger.debug(f"[Tool] 결과: {result_content[:200]}...")

        # 결과를 하나의 메시지로 포맷팅
        result_parts = []
        for r in results:
            result_parts.append(f"Tool: {r['tool_name']}\nResult:\n{r['result']}")

        combined_result = "\n\n---\n\n".join(result_parts)
        tool_response = f"<tool_response>\n{combined_result}\n</tool_response>"

        new_message = Message(
            role="user",
            content=tool_response
        )

        return {
            "messages": [new_message],
            "pending_tool_calls": []
        }


def should_continue(state: AgentState) -> str:
    """
    조건부 라우팅 함수: 다음 노드 결정

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름 ("agent", "tool", "end")
    """
    # 완료된 경우
    if state.get("is_complete", False):
        return "end"

    # 대기 중인 도구 호출이 있는 경우
    if state.get("pending_tool_calls"):
        return "tool"

    # 기본: 에이전트 노드로
    return "agent"


def should_continue_after_tool(state: AgentState) -> str:
    """
    도구 실행 후 라우팅 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름 ("agent", "end")
    """
    if state.get("is_complete", False):
        return "end"

    return "agent"
