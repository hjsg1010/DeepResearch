"""
프롬프트 템플릿

에이전트에서 사용하는 시스템 프롬프트 및 도구 설명을 정의합니다.
"""

from typing import List, Dict, Any
from datetime import datetime

from langgraph_agent.tools.base import BaseTool


def get_current_date() -> str:
    """현재 날짜 반환"""
    return datetime.now().strftime("%Y-%m-%d")


# 시스템 프롬프트 템플릿
SYSTEM_PROMPT_TEMPLATE = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Guidelines

1. **Thorough Research**: Before providing an answer, search for relevant information using available tools. Use multiple queries if necessary to gather comprehensive information.

2. **Source Verification**: Cross-reference information from multiple sources when possible. If information conflicts, note the discrepancy.

3. **Structured Response**: Organize your findings logically. Use headings, bullet points, or numbered lists for clarity.

4. **Citation**: When referencing specific documents or pages, mention the source.

5. **Thinking Process**: Before each action, briefly explain your reasoning. Use <think></think> tags for your internal reasoning.

6. **Answer Format**: When you have gathered sufficient information, provide your final answer within <answer></answer> tags.

7. **Language**: Respond in the same language as the user's query.

Current date: {current_date}
"""


def build_system_prompt(tools: List[BaseTool]) -> str:
    """
    도구 정의를 포함한 시스템 프롬프트 생성

    Args:
        tools: 사용 가능한 도구 리스트

    Returns:
        완성된 시스템 프롬프트
    """
    tool_definitions = "\n".join(tool.get_prompt_description() for tool in tools)

    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_definitions=tool_definitions,
        current_date=get_current_date()
    )


# 도구 설명 템플릿 (도구별 상세 설명)
RETRIEVE_TOOL_DESCRIPTION = """{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "사내 문서 저장소에서 관련 문서를 검색합니다. 기술 문서, 가이드라인, 정책 문서 등을 검색할 수 있습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리 (키워드 또는 자연어 질문)"
                },
                "filters": {
                    "type": "object",
                    "description": "검색 필터 (예: {'department': 'AI/ML Team', 'tags': ['llm']})"
                },
                "max_results": {
                    "type": "integer",
                    "description": "최대 반환 결과 수 (기본값: 5)"
                }
            },
            "required": ["query"]
        }
    }
}"""

CONFLUENCE_TOOL_DESCRIPTION = """{
    "type": "function",
    "function": {
        "name": "confluence_search",
        "description": "Confluence 위키에서 문서를 검색합니다. 사내 위키, 프로젝트 문서, 가이드라인 등을 검색할 수 있습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리 (키워드 또는 문장)"
                },
                "space_key": {
                    "type": "string",
                    "description": "검색할 Confluence 스페이스 키 (TECH, PROJ, HR, GUIDE)"
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "필터링할 라벨 목록"
                },
                "max_results": {
                    "type": "integer",
                    "description": "최대 반환 결과 수 (기본값: 5)"
                }
            },
            "required": ["query"]
        }
    }
}"""


# 컨텍스트 길이 초과 시 사용하는 프롬프트
CONTEXT_LIMIT_PROMPT = """You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:

<think>your final thinking</think>
<answer>your answer</answer>"""


# 웹페이지 내용 추출 프롬프트 (기존 repo의 EXTRACTOR_PROMPT와 유사)
CONTENT_EXTRACTOR_PROMPT = """Please process the following document content and user goal to extract relevant information:

## **Document Content**
{document_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the document content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**"""


# 검색 쿼리 생성 프롬프트
QUERY_GENERATION_PROMPT = """Based on the user's question, generate effective search queries.

User Question: {question}

Generate 1-3 search queries that would help find relevant information. Consider:
1. Key concepts and terms
2. Related technical terms
3. Alternative phrasings

Output as JSON array:
["query1", "query2", "query3"]"""


# 답변 합성 프롬프트
ANSWER_SYNTHESIS_PROMPT = """Based on the information gathered from various sources, synthesize a comprehensive answer to the user's question.

## User Question
{question}

## Gathered Information
{gathered_info}

## Guidelines
1. Synthesize information from all sources
2. Highlight key findings
3. Note any conflicting information
4. Provide actionable recommendations if applicable
5. Cite sources where appropriate

Provide your answer within <answer></answer> tags."""


def get_tool_call_format_example() -> str:
    """도구 호출 형식 예시 반환"""
    return """
Example tool call:
<tool_call>
{"name": "retrieve", "arguments": {"query": "API design guidelines"}}
</tool_call>

Example with multiple arguments:
<tool_call>
{"name": "confluence_search", "arguments": {"query": "deployment process", "space_key": "TECH", "max_results": 3}}
</tool_call>
"""
