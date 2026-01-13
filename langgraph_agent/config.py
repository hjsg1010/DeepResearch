"""
에이전트 설정 모듈

OpenAI 호환 API 설정 및 에이전트 파라미터를 관리합니다.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class LLMConfig:
    """LLM 호출 설정"""
    base_url: str = "http://localhost:8000/v1"  # OpenAI 호환 API 엔드포인트
    api_key: str = "your-api-key-here"  # API 키
    model_name: str = "openai/gpt-oss-120b"  # 모델 이름

    # 생성 파라미터
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 10000
    presence_penalty: float = 1.1
    timeout: float = 600.0

    # 재시도 설정
    max_retries: int = 10
    base_sleep_time: float = 1.0
    max_sleep_time: float = 30.0

    # Stop 시퀀스
    stop_sequences: list = field(default_factory=lambda: ["\n<tool_response>", "<tool_response>"])


@dataclass
class MCPConfig:
    """MCP 도구 설정"""
    retrieve_endpoint: str = "http://localhost:9000/mcp/retrieve"
    confluence_endpoint: str = "http://localhost:9001/mcp/confluence"
    timeout: float = 30.0
    max_results: int = 10


@dataclass
class AgentConfig:
    """전체 에이전트 설정"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # 에이전트 실행 제한
    max_turns: int = 100  # 최대 턴 수
    max_tokens_context: int = 110 * 1024  # 최대 컨텍스트 토큰 수 (110K)
    max_execution_time: int = 150 * 60  # 최대 실행 시간 (150분)

    # 로깅 설정
    verbose: bool = True
    log_dir: str = "./logs"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """환경 변수에서 설정 로드"""
        llm_config = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", "your-api-key-here"),
            model_name=os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
            top_p=float(os.getenv("LLM_TOP_P", "0.95")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "10000")),
            presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", "1.1")),
        )

        mcp_config = MCPConfig(
            retrieve_endpoint=os.getenv("MCP_RETRIEVE_ENDPOINT", "http://localhost:9000/mcp/retrieve"),
            confluence_endpoint=os.getenv("MCP_CONFLUENCE_ENDPOINT", "http://localhost:9001/mcp/confluence"),
        )

        return cls(
            llm=llm_config,
            mcp=mcp_config,
            max_turns=int(os.getenv("MAX_TURNS", "100")),
            verbose=os.getenv("VERBOSE", "true").lower() == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "llm": {
                "base_url": self.llm.base_url,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "presence_penalty": self.llm.presence_penalty,
            },
            "mcp": {
                "retrieve_endpoint": self.mcp.retrieve_endpoint,
                "confluence_endpoint": self.mcp.confluence_endpoint,
            },
            "max_turns": self.max_turns,
            "max_tokens_context": self.max_tokens_context,
            "max_execution_time": self.max_execution_time,
        }


def get_current_date() -> str:
    """현재 날짜 반환"""
    return datetime.now().strftime("%Y-%m-%d")
