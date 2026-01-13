#!/usr/bin/env python3
"""
LangGraph DeepResearch Agent 기본 사용 예제

이 예제는 에이전트의 기본적인 사용법을 보여줍니다.
"""

import logging
import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph_agent.config import AgentConfig, LLMConfig
from langgraph_agent.graph import ResearchAgent, run_research


def example_basic():
    """기본 사용 예제"""
    print("=" * 60)
    print("예제 1: 기본 사용")
    print("=" * 60)

    # 설정 생성 (Mock 모드로 실행하려면 실제 LLM 없이도 테스트 가능)
    config = AgentConfig(
        llm=LLMConfig(
            base_url="http://localhost:8000/v1",  # OpenAI 호환 서버
            api_key="your-api-key",
            model_name="openai/gpt-oss-120b",
            temperature=0.6,
        ),
        max_turns=50,
        verbose=True
    )

    # 질문
    question = "사내 LLM 에이전트 개발 가이드에 대해 알려줘"

    print(f"\n질문: {question}\n")

    # 연구 실행 (실제 LLM 서버가 필요)
    # result = run_research(question, config=config)
    # print(f"답변: {result.answer}")

    print("(실제 실행하려면 LLM 서버가 필요합니다)")


def example_with_custom_tools():
    """커스텀 도구 추가 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 커스텀 도구 추가")
    print("=" * 60)

    from langgraph_agent.tools.base import BaseTool, ToolParameter, ToolResult, ToolRegistry
    from langgraph_agent.tools.retrieve import RetrieveTool
    from langgraph_agent.tools.confluence import ConfluenceSearchTool

    # 커스텀 도구 정의
    class CustomSearchTool(BaseTool):
        name = "custom_search"
        description = "커스텀 검색을 수행합니다."
        parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="검색 쿼리",
                required=True
            )
        ]

        def call(self, params):
            query = params.get("query", "")
            # 실제 구현에서는 외부 API 호출 등
            return ToolResult(
                success=True,
                content=f"'{query}'에 대한 커스텀 검색 결과입니다.",
                metadata={"query": query}
            )

    # 도구 레지스트리 생성
    registry = ToolRegistry()
    registry.register(RetrieveTool())
    registry.register(ConfluenceSearchTool())
    registry.register(CustomSearchTool())

    print(f"등록된 도구: {registry.list_tools()}")

    # 도구 테스트
    result = registry.call_tool("custom_search", {"query": "테스트"})
    print(f"테스트 결과: {result.content}")


def example_tool_test():
    """도구 독립 테스트 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 도구 독립 테스트")
    print("=" * 60)

    from langgraph_agent.tools.retrieve import RetrieveTool
    from langgraph_agent.tools.confluence import ConfluenceSearchTool

    # Retrieve 도구 테스트
    retrieve_tool = RetrieveTool()
    print(f"\n도구: {retrieve_tool.name}")
    print(f"설명: {retrieve_tool.description}")

    result = retrieve_tool.call({"query": "LLM 에이전트"})
    print(f"\n검색 결과:\n{result.content[:500]}...")

    # Confluence 도구 테스트
    confluence_tool = ConfluenceSearchTool()
    print(f"\n\n도구: {confluence_tool.name}")

    result = confluence_tool.call({
        "query": "CI/CD",
        "space_key": "TECH"
    })
    print(f"\n검색 결과:\n{result.content[:500]}...")


def example_config():
    """설정 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 설정 관리")
    print("=" * 60)

    # 기본 설정
    config = AgentConfig()
    print(f"기본 설정: {config.to_dict()}")

    # 환경 변수에서 설정 로드
    # os.environ["LLM_BASE_URL"] = "http://custom-server:8000/v1"
    # config = AgentConfig.from_env()

    # 커스텀 설정
    custom_config = AgentConfig(
        llm=LLMConfig(
            base_url="http://internal-llm.company.com/v1",
            api_key="internal-api-key",
            model_name="company/llm-model",
            temperature=0.7,
            max_tokens=8000,
        ),
        max_turns=30,
        max_tokens_context=100 * 1024,
        verbose=True
    )

    print(f"\n커스텀 설정:")
    print(f"  - LLM: {custom_config.llm.model_name}")
    print(f"  - Base URL: {custom_config.llm.base_url}")
    print(f"  - Max turns: {custom_config.max_turns}")


def example_async():
    """비동기 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 비동기 실행")
    print("=" * 60)

    print("""
비동기 실행 예제 코드:

import asyncio
from langgraph_agent.graph import arun_research, ResearchAgent

async def main():
    # 방법 1: 함수 직접 사용
    result = await arun_research("질문")
    print(result.answer)

    # 방법 2: 에이전트 클래스 사용
    agent = ResearchAgent()
    result = await agent.arun("질문")
    print(result.answer)

    # 방법 3: 여러 질문 병렬 처리
    questions = ["질문1", "질문2", "질문3"]
    tasks = [arun_research(q) for q in questions]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.answer)

asyncio.run(main())
""")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 예제 실행
    example_basic()
    example_with_custom_tools()
    example_tool_test()
    example_config()
    example_async()

    print("\n" + "=" * 60)
    print("모든 예제 완료!")
    print("=" * 60)
