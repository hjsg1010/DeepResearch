"""
OpenAI 호환 API 클라이언트

langchain-openai를 사용하지 않고 openai 패키지를 직접 사용합니다.
이를 통해 vLLM, LocalAI, Ollama 등 OpenAI 호환 서버에 연결할 수 있습니다.
"""

import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from langgraph_agent.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 응답 구조체"""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    logprobs: Optional[Any] = None
    raw_response: Optional[Any] = None


class OpenAICompatibleClient:
    """
    OpenAI 호환 API 클라이언트

    이 클라이언트는 OpenAI API 형식을 따르는 모든 서버와 통신할 수 있습니다.
    주요 특징:
    - 자동 재시도 (지수 백오프)
    - 로깅 및 에러 처리
    - Stop 시퀀스 지원
    - 토큰 사용량 추적
    """

    def __init__(self, config: LLMConfig):
        """
        클라이언트 초기화

        Args:
            config: LLM 설정 객체
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.total_tokens_used = 0
        self.total_calls = 0

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        logprobs: bool = False,
    ) -> LLMResponse:
        """
        채팅 완성 API 호출

        Args:
            messages: 대화 메시지 리스트
            temperature: 샘플링 온도 (None이면 config 값 사용)
            max_tokens: 최대 생성 토큰 수 (None이면 config 값 사용)
            stop: Stop 시퀀스 리스트 (None이면 config 값 사용)
            logprobs: 로그 확률 반환 여부

        Returns:
            LLMResponse: LLM 응답

        Raises:
            RuntimeError: 모든 재시도 실패 시
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        stop = stop if stop is not None else self.config.stop_sequences

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"LLM 호출 시도 {attempt + 1}/{self.config.max_retries}")

                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    max_tokens=max_tokens,
                    presence_penalty=self.config.presence_penalty,
                    stop=stop,
                    logprobs=logprobs,
                )

                content = response.choices[0].message.content

                if content and content.strip():
                    self.total_calls += 1
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens

                    logger.debug(f"LLM 호출 성공: {len(content)} 문자")

                    return LLMResponse(
                        content=content.strip(),
                        finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else None,
                        logprobs=response.choices[0].logprobs if logprobs else None,
                        raw_response=response,
                    )
                else:
                    logger.warning(f"시도 {attempt + 1}: 빈 응답 수신")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.error(f"시도 {attempt + 1}: API 오류 - {e}")
            except Exception as e:
                logger.error(f"시도 {attempt + 1}: 예상치 못한 오류 - {e}")

            # 재시도 전 대기 (지수 백오프)
            if attempt < self.config.max_retries - 1:
                sleep_time = min(
                    self.config.base_sleep_time * (2 ** attempt) + random.uniform(0, 1),
                    self.config.max_sleep_time
                )
                logger.info(f"{sleep_time:.2f}초 후 재시도...")
                time.sleep(sleep_time)

        raise RuntimeError(f"LLM 호출 실패: {self.config.max_retries}회 재시도 후에도 실패")

    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정

        Args:
            text: 토큰 수를 계산할 텍스트

        Returns:
            추정 토큰 수
        """
        # tiktoken을 사용한 정확한 토큰 카운팅
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # tiktoken이 없으면 대략적인 추정
            return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        메시지 리스트의 총 토큰 수 계산

        Args:
            messages: 메시지 리스트

        Returns:
            총 토큰 수
        """
        total = 0
        for msg in messages:
            # 역할 토큰 (약 4토큰)
            total += 4
            # 내용 토큰
            if "content" in msg and msg["content"]:
                total += self.count_tokens(msg["content"])
        return total

    def get_stats(self) -> Dict[str, Any]:
        """
        사용 통계 반환

        Returns:
            호출 횟수 및 토큰 사용량 통계
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens_used": self.total_tokens_used,
            "config": {
                "model": self.config.model_name,
                "base_url": self.config.base_url,
            }
        }

    def reset_stats(self) -> None:
        """사용 통계 초기화"""
        self.total_calls = 0
        self.total_tokens_used = 0


class AsyncOpenAICompatibleClient:
    """
    비동기 OpenAI 호환 API 클라이언트

    병렬 처리가 필요한 경우 사용합니다.
    """

    def __init__(self, config: LLMConfig):
        """
        클라이언트 초기화

        Args:
            config: LLM 설정 객체
        """
        from openai import AsyncOpenAI

        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        비동기 채팅 완성 API 호출

        Args:
            messages: 대화 메시지 리스트
            temperature: 샘플링 온도
            max_tokens: 최대 생성 토큰 수
            stop: Stop 시퀀스 리스트

        Returns:
            LLMResponse: LLM 응답
        """
        import asyncio

        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        stop = stop if stop is not None else self.config.stop_sequences

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    max_tokens=max_tokens,
                    presence_penalty=self.config.presence_penalty,
                    stop=stop,
                )

                content = response.choices[0].message.content

                if content and content.strip():
                    return LLMResponse(
                        content=content.strip(),
                        finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else None,
                    )

            except Exception as e:
                logger.error(f"비동기 LLM 호출 실패 (시도 {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                sleep_time = min(
                    self.config.base_sleep_time * (2 ** attempt) + random.uniform(0, 1),
                    self.config.max_sleep_time
                )
                await asyncio.sleep(sleep_time)

        raise RuntimeError(f"비동기 LLM 호출 실패: {self.config.max_retries}회 재시도 후에도 실패")
