# LangGraph DeepResearch Agent 아키텍처 문서

## 1. 개요

이 문서는 Alibaba DeepResearch 에이전트를 LangGraph 형태로 재구현한 아키텍처를 설명합니다.

### 1.1 주요 설계 원칙

1. **langchain-openai 미사용**: OpenAI 호환 API를 `openai` 패키지로 직접 호출
2. **회사망 환경 적합**: 외부 검색 도구 대신 MCP 기반 내부 도구 사용
3. **LangGraph 기반**: StateGraph를 활용한 선언적 워크플로우
4. **확장 가능성**: 도구 추가 및 설정 변경 용이

### 1.2 기술 스택

| 구분 | 기술 |
|------|------|
| 워크플로우 프레임워크 | LangGraph |
| LLM 호출 | OpenAI Python SDK (직접 사용) |
| 도구 프로토콜 | MCP (Model Context Protocol) |
| 상태 관리 | TypedDict 기반 State |

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 입력                              │
│                    (질문/연구 주제)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   ResearchAgent                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 LangGraph StateGraph                  │  │
│  │                                                       │  │
│  │  ┌─────────────┐         ┌─────────────┐             │  │
│  │  │             │         │             │             │  │
│  │  │ agent_node  │◄───────►│  tool_node  │             │  │
│  │  │   (LLM)     │         │  (도구실행)  │             │  │
│  │  │             │         │             │             │  │
│  │  └──────┬──────┘         └─────────────┘             │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │  ┌─────────────┐                                     │  │
│  │  │    END      │                                     │  │
│  │  │  (답변완료)  │                                     │  │
│  │  └─────────────┘                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────┼───────────────────────────────┐ │
│  │                       ▼                                │ │
│  │  ┌─────────────────────────────────────────────────┐  │ │
│  │  │             OpenAI Compatible Client            │  │ │
│  │  │                                                  │  │ │
│  │  │  - base_url: http://localhost:8000/v1           │  │ │
│  │  │  - model: openai/gpt-oss-120b                   │  │ │
│  │  │  - 자동 재시도, 지수 백오프                       │  │ │
│  │  └─────────────────────────────────────────────────┘  │ │
│  │                                                        │ │
│  │  ┌─────────────────────────────────────────────────┐  │ │
│  │  │              Tool Registry (MCP)                │  │ │
│  │  │                                                  │  │ │
│  │  │  - retrieve: 문서 저장소 검색                     │  │ │
│  │  │  - confluence_search: 위키 검색                  │  │ │
│  │  │  - (확장 가능)                                   │  │ │
│  │  └─────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ResearchResult                            │
│                                                              │
│  - question: 원래 질문                                       │
│  - answer: 최종 답변                                         │
│  - messages: 대화 히스토리                                   │
│  - termination_reason: 종료 사유                             │
│  - total_turns / total_tokens / execution_time              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 LangGraph 워크플로우

```
              ┌──────────────┐
              │    START     │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
         ┌───►│  agent_node  │◄───────────────┐
         │    │   (LLM 호출)  │                │
         │    └──────┬───────┘                │
         │           │                         │
         │           ▼                         │
         │    ┌──────────────┐                │
         │    │ should_      │                │
         │    │ continue?    │                │
         │    └──────┬───────┘                │
         │           │                         │
         │     ┌─────┼─────┐                  │
         │     │     │     │                  │
         │     ▼     ▼     ▼                  │
         │   tool  agent  end                 │
         │     │     │                        │
         │     │     └────────────────────────┘
         │     │
         │     ▼
         │  ┌──────────────┐
         │  │  tool_node   │
         │  │  (도구 실행)  │
         │  └──────┬───────┘
         │         │
         └─────────┘
```

---

## 3. 핵심 컴포넌트

### 3.1 State (상태)

`AgentState`는 에이전트의 전체 상태를 관리합니다.

```python
class AgentState(TypedDict):
    messages: List[Message]          # 대화 히스토리
    question: str                     # 원래 질문
    current_turn: int                 # 현재 턴 번호
    max_turns: int                    # 최대 턴 수
    start_time: float                 # 시작 시간
    total_tokens: int                 # 총 토큰 사용량
    pending_tool_calls: List[ToolCall]  # 대기 중인 도구 호출
    last_response: str                # 마지막 LLM 응답
    answer: Optional[str]             # 최종 답변
    termination_reason: Optional[str] # 종료 사유
    is_complete: bool                 # 완료 여부
    metadata: Dict[str, Any]          # 추가 메타데이터
```

### 3.2 Nodes (노드)

#### agent_node
- **역할**: LLM을 호출하여 다음 액션 결정
- **입력**: 현재 State
- **출력**: State 업데이트 (메시지, 도구 호출, 답변 등)
- **로직**:
  1. 시간/턴 제한 확인
  2. 컨텍스트 길이 확인
  3. LLM 호출
  4. 응답 파싱 (도구 호출 / 답변)
  5. State 업데이트 반환

#### tool_node
- **역할**: 대기 중인 도구 호출 실행
- **입력**: 현재 State (pending_tool_calls)
- **출력**: State 업데이트 (도구 결과 메시지)
- **로직**:
  1. 도구 호출 순차 실행
  2. 결과 수집
  3. `<tool_response>` 형식으로 메시지 추가

### 3.3 Edges (엣지)

#### 조건부 라우팅: should_continue
```python
def should_continue(state: AgentState) -> str:
    if state.get("is_complete"):
        return "end"
    if state.get("pending_tool_calls"):
        return "tool"
    return "agent"
```

---

## 4. LLM 호출 방식

### 4.1 OpenAI 호환 API 직접 사용

langchain-openai 대신 `openai` 패키지를 직접 사용합니다.

```python
from openai import OpenAI

client = OpenAI(
    api_key=config.api_key,
    base_url=config.base_url,  # OpenAI 호환 서버
    timeout=config.timeout,
)

response = client.chat.completions.create(
    model=config.model_name,  # "openai/gpt-oss-120b"
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    max_tokens=10000,
    presence_penalty=1.1,
    stop=["\n<tool_response>", "<tool_response>"],
)
```

### 4.2 재시도 로직

```python
for attempt in range(max_retries):
    try:
        response = client.chat.completions.create(...)
        if response.choices[0].message.content:
            return response
    except (APIError, APIConnectionError, APITimeoutError) as e:
        sleep_time = min(base_sleep_time * (2 ** attempt), max_sleep_time)
        time.sleep(sleep_time)
```

### 4.3 설정 가능한 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| base_url | http://localhost:8000/v1 | OpenAI 호환 서버 주소 |
| api_key | your-api-key-here | API 인증 키 |
| model_name | openai/gpt-oss-120b | 모델 이름 |
| temperature | 0.6 | 샘플링 온도 |
| top_p | 0.95 | Top-p 샘플링 |
| max_tokens | 10000 | 최대 생성 토큰 |
| presence_penalty | 1.1 | 반복 회피 페널티 |

---

## 5. MCP 도구 구현

### 5.1 도구 인터페이스

```python
class BaseTool(ABC):
    name: str
    description: str
    parameters: List[ToolParameter]

    @abstractmethod
    def call(self, params: Dict[str, Any]) -> ToolResult:
        """도구 실행"""
        pass
```

### 5.2 구현된 도구

#### RetrieveTool (문서 검색)
- **이름**: `retrieve`
- **기능**: 사내 문서 저장소에서 관련 문서 검색
- **파라미터**:
  - `query` (필수): 검색 쿼리
  - `filters` (선택): 필터 조건
  - `max_results` (선택): 최대 결과 수

#### ConfluenceSearchTool (위키 검색)
- **이름**: `confluence_search`
- **기능**: Confluence 위키에서 문서 검색
- **파라미터**:
  - `query` (필수): 검색 쿼리
  - `space_key` (선택): 스페이스 키
  - `labels` (선택): 라벨 필터

### 5.3 도구 레지스트리

```python
registry = ToolRegistry()
registry.register(RetrieveTool())
registry.register(ConfluenceSearchTool())

# 도구 호출
result = registry.call_tool("retrieve", {"query": "API 설계"})
```

---

## 6. 프롬프트 구조

### 6.1 시스템 프롬프트

```
You are a deep research assistant...

# Tools
<tools>
{도구 정의 JSON}
</tools>

For each function call, return a json object:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

When ready to answer:
<answer>your final answer</answer>

Current date: {날짜}
```

### 6.2 도구 응답 형식

```
<tool_response>
Tool: retrieve
Result:
{검색 결과}
</tool_response>
```

---

## 7. 파일 구조

```
langgraph_agent/
├── __init__.py           # 모듈 진입점
├── config.py             # 설정 관리
├── state.py              # State 정의
├── prompts.py            # 프롬프트 템플릿
├── nodes.py              # LangGraph 노드
├── graph.py              # 그래프 정의 및 실행
├── main.py               # CLI 진입점
│
├── llm/
│   ├── __init__.py
│   └── openai_client.py  # OpenAI 호환 클라이언트
│
├── tools/
│   ├── __init__.py
│   ├── base.py           # 도구 기본 클래스
│   ├── retrieve.py       # 문서 검색 도구
│   └── confluence.py     # Confluence 검색 도구
│
├── utils/
│   ├── __init__.py
│   └── parsing.py        # 파싱 유틸리티
│
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   └── test_nodes.py
│
├── examples/
│   └── basic_usage.py
│
└── requirements.txt
```

---

## 8. 확장 방법

### 8.1 새 도구 추가

```python
from langgraph_agent.tools.base import BaseTool, ToolParameter, ToolResult

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "My custom tool description"
    parameters = [
        ToolParameter(
            name="param1",
            type="string",
            description="Parameter description",
            required=True
        )
    ]

    def call(self, params):
        # 구현
        return ToolResult(success=True, content="결과")

# 등록
registry.register(MyCustomTool())
```

### 8.2 LLM 설정 변경

```python
config = AgentConfig(
    llm=LLMConfig(
        base_url="http://my-llm-server/v1",
        api_key="my-key",
        model_name="my-model",
    )
)
```

---

## 9. 제한 사항 및 주의사항

1. **토큰 제한**: 기본 컨텍스트 110K 토큰 제한
2. **시간 제한**: 기본 150분 실행 시간 제한
3. **턴 제한**: 기본 100턴 제한
4. **Mock 도구**: 현재 도구는 Mock 구현이며 실제 MCP 서버 연동 필요

---

## 10. 참고 자료

- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 문서](https://platform.openai.com/docs/api-reference)
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/)
- [DeepResearch 원본 레포지토리](https://github.com/Alibaba-NLP/DeepResearch)
