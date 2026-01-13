"""
LangGraph State 정의

에이전트의 상태를 관리하는 TypedDict 및 관련 유틸리티를 정의합니다.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import operator


class Message(TypedDict, total=False):
    """메시지 구조"""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_result: Optional[str]


class ToolCall(TypedDict):
    """도구 호출 정보"""
    name: str
    arguments: Dict[str, Any]


class AgentState(TypedDict):
    """
    LangGraph 에이전트 상태

    Attributes:
        messages: 대화 메시지 히스토리
        question: 사용자의 원래 질문
        current_turn: 현재 턴 번호
        max_turns: 최대 턴 수
        start_time: 시작 시간 (timestamp)
        total_tokens: 총 사용 토큰 수
        pending_tool_calls: 실행 대기 중인 도구 호출
        last_response: 마지막 LLM 응답
        answer: 최종 답변 (있는 경우)
        termination_reason: 종료 사유
        is_complete: 완료 여부
        metadata: 추가 메타데이터
    """
    messages: Annotated[List[Message], operator.add]
    question: str
    current_turn: int
    max_turns: int
    start_time: float
    total_tokens: int
    pending_tool_calls: List[ToolCall]
    last_response: str
    answer: Optional[str]
    termination_reason: Optional[str]
    is_complete: bool
    metadata: Dict[str, Any]


def create_initial_state(
    question: str,
    system_prompt: str,
    max_turns: int = 100
) -> AgentState:
    """
    초기 상태 생성

    Args:
        question: 사용자 질문
        system_prompt: 시스템 프롬프트
        max_turns: 최대 턴 수

    Returns:
        AgentState: 초기화된 상태
    """
    import time

    return AgentState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        question=question,
        current_turn=0,
        max_turns=max_turns,
        start_time=time.time(),
        total_tokens=0,
        pending_tool_calls=[],
        last_response="",
        answer=None,
        termination_reason=None,
        is_complete=False,
        metadata={}
    )


@dataclass
class ResearchResult:
    """연구 결과 구조체"""
    question: str
    answer: str
    messages: List[Message]
    termination_reason: str
    total_turns: int
    total_tokens: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "question": self.question,
            "answer": self.answer,
            "messages": self.messages,
            "termination_reason": self.termination_reason,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }

    @classmethod
    def from_state(cls, state: AgentState) -> "ResearchResult":
        """상태에서 결과 생성"""
        import time

        return cls(
            question=state["question"],
            answer=state.get("answer", ""),
            messages=state["messages"],
            termination_reason=state.get("termination_reason", "unknown"),
            total_turns=state["current_turn"],
            total_tokens=state["total_tokens"],
            execution_time=time.time() - state["start_time"],
            metadata=state.get("metadata", {})
        )
