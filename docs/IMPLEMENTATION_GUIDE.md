# 구현 가이드

## 1. 빠른 시작

### 1.1 설치

```bash
# 의존성 설치
pip install -r langgraph_agent/requirements.txt
```

### 1.2 환경 변수 설정

```bash
export LLM_BASE_URL="http://your-llm-server:8000/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="openai/gpt-oss-120b"
```

### 1.3 기본 실행

```python
from langgraph_agent import run_research, AgentConfig

# 기본 설정으로 실행
result = run_research("시스템 아키텍처에 대해 알려줘")
print(result.answer)

# 커스텀 설정으로 실행
config = AgentConfig.from_env()
config.max_turns = 50
result = run_research("질문", config=config)
```

### 1.4 CLI 실행

```bash
# 단일 질문
python -m langgraph_agent.main "질문 내용"

# 대화형 모드
python -m langgraph_agent.main --interactive

# 상세 로깅
python -m langgraph_agent.main --verbose "질문"
```

---

## 2. 주요 구현 내용

### 2.1 LangGraph 기반 워크플로우

기존 DeepResearch의 while 루프 기반 ReAct 패턴을 LangGraph의 StateGraph로 변환했습니다.

**기존 방식 (react_agent.py):**
```python
while num_llm_calls_available > 0:
    content = self.call_server(messages, planning_port)
    messages.append({"role": "assistant", "content": content})
    if '<tool_call>' in content:
        result = self.custom_call_tool(tool_name, tool_args)
        messages.append({"role": "user", "content": result})
    if '<answer>' in content:
        break
```

**LangGraph 방식 (graph.py):**
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", nodes.agent_node)
workflow.add_node("tool", nodes.tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, ...)
workflow.add_conditional_edges("tool", should_continue_after_tool, ...)
```

### 2.2 OpenAI 호환 API 직접 사용

langchain-openai를 사용하지 않고 openai 패키지를 직접 사용합니다.

**설계 이유:**
1. 의존성 최소화
2. 더 세밀한 제어 가능
3. 회사 내부 LLM 서버와의 호환성 보장

**구현:**
```python
from openai import OpenAI

class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(self, messages: List[Dict], ...) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            ...
        )
        return LLMResponse(content=response.choices[0].message.content)
```

### 2.3 MCP Mock 도구

외부 검색 API (Serper, Jina 등) 대신 MCP 기반 내부 도구를 Mock으로 구현했습니다.

**구현된 도구:**
- `retrieve`: 사내 문서 저장소 검색
- `retrieve_by_id`: 문서 ID로 직접 조회
- `confluence_search`: Confluence 위키 검색
- `confluence_get_page`: Confluence 페이지 직접 조회
- `confluence_list_spaces`: 스페이스 목록 조회

**Mock 데이터 예시:**
```python
MOCK_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "시스템 아키텍처 설계 문서",
        "content": "...",
        "metadata": {
            "author": "김아키텍트",
            "department": "Platform Engineering",
            "tags": ["architecture", "microservices"]
        }
    },
    ...
]
```

---

## 3. 설정 상세

### 3.1 LLM 설정 (LLMConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| base_url | str | http://localhost:8000/v1 | API 엔드포인트 |
| api_key | str | your-api-key-here | API 키 |
| model_name | str | openai/gpt-oss-120b | 모델 이름 |
| temperature | float | 0.6 | 샘플링 온도 |
| top_p | float | 0.95 | Top-p 샘플링 |
| max_tokens | int | 10000 | 최대 생성 토큰 |
| presence_penalty | float | 1.1 | 반복 회피 |
| timeout | float | 600.0 | 요청 타임아웃 (초) |
| max_retries | int | 10 | 최대 재시도 횟수 |

### 3.2 MCP 설정 (MCPConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| retrieve_endpoint | str | http://localhost:9000/mcp/retrieve | Retrieve 엔드포인트 |
| confluence_endpoint | str | http://localhost:9001/mcp/confluence | Confluence 엔드포인트 |
| timeout | float | 30.0 | 요청 타임아웃 |
| max_results | int | 10 | 최대 결과 수 |

### 3.3 에이전트 설정 (AgentConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| max_turns | int | 100 | 최대 턴 수 |
| max_tokens_context | int | 112640 | 최대 컨텍스트 토큰 |
| max_execution_time | int | 9000 | 최대 실행 시간 (초) |
| verbose | bool | True | 상세 로깅 |

---

## 4. 프롬프트 엔지니어링

### 4.1 시스템 프롬프트 구조

```
[역할 정의]
You are a deep research assistant...

[도구 정의]
<tools>
{"type": "function", "function": {"name": "retrieve", ...}}
{"type": "function", "function": {"name": "confluence_search", ...}}
</tools>

[호출 형식]
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

[가이드라인]
1. Thorough Research: ...
2. Source Verification: ...
3. Structured Response: ...
4. Citation: ...
5. Thinking Process: <think></think>
6. Answer Format: <answer></answer>
7. Language: 사용자 언어로 응답

[메타 정보]
Current date: YYYY-MM-DD
```

### 4.2 도구 호출 형식

```xml
<think>사고 과정...</think>
<tool_call>
{"name": "retrieve", "arguments": {"query": "검색어"}}
</tool_call>
```

### 4.3 답변 형식

```xml
<think>최종 정리...</think>
<answer>
최종 답변 내용
</answer>
```

---

## 5. 상태 관리

### 5.1 AgentState 필드

```python
class AgentState(TypedDict):
    # 대화 관련
    messages: List[Message]          # 전체 대화 히스토리
    question: str                     # 원래 질문

    # 실행 제어
    current_turn: int                 # 현재 턴
    max_turns: int                    # 최대 턴
    start_time: float                 # 시작 시각
    total_tokens: int                 # 누적 토큰

    # 도구 호출
    pending_tool_calls: List[ToolCall]  # 대기 중인 호출

    # 결과
    last_response: str                # 마지막 응답
    answer: Optional[str]             # 최종 답변
    termination_reason: Optional[str] # 종료 사유
    is_complete: bool                 # 완료 여부

    # 기타
    metadata: Dict[str, Any]
```

### 5.2 메시지 누적 방식

LangGraph의 `Annotated[List, operator.add]`를 사용하여 메시지가 자동으로 누적됩니다.

```python
messages: Annotated[List[Message], operator.add]
```

노드에서 반환:
```python
return {"messages": [new_message]}  # 기존 메시지에 추가됨
```

---

## 6. 테스트

### 6.1 도구 테스트

```bash
python -m pytest langgraph_agent/tests/test_tools.py -v
```

### 6.2 노드 테스트

```bash
python -m pytest langgraph_agent/tests/test_nodes.py -v
```

### 6.3 전체 테스트

```bash
python -m pytest langgraph_agent/tests/ -v
```

### 6.4 예제 실행

```bash
python langgraph_agent/examples/basic_usage.py
```

---

## 7. 디버깅

### 7.1 로깅 활성화

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 7.2 상세 모드

```bash
python -m langgraph_agent.main --verbose "질문"
```

### 7.3 상태 확인

```python
for step in graph.stream(initial_state):
    print(f"Step: {step}")
    for node_name, node_state in step.items():
        print(f"  Node: {node_name}")
        print(f"  Turn: {node_state.get('current_turn')}")
        print(f"  Pending tools: {node_state.get('pending_tool_calls')}")
```

---

## 8. 성능 최적화

### 8.1 토큰 사용량 관리

- 컨텍스트 길이 모니터링
- 필요시 대화 히스토리 요약
- 도구 결과 길이 제한

### 8.2 재시도 최적화

- 지수 백오프 (1초 → 2초 → 4초 → ...)
- 최대 30초 대기
- 최대 10회 재시도

### 8.3 병렬 처리 (향후)

- 비동기 LLM 호출
- 여러 질문 동시 처리

---

## 9. 트러블슈팅

### 9.1 LLM 연결 실패

**증상**: `APIConnectionError` 발생

**해결**:
1. base_url 확인
2. 서버 상태 확인
3. 네트워크 연결 확인

### 9.2 도구 호출 파싱 실패

**증상**: `도구 호출 파싱 실패` 로그

**해결**:
1. LLM 응답 형식 확인
2. JSON 구문 오류 확인
3. 프롬프트 개선 검토

### 9.3 컨텍스트 길이 초과

**증상**: `컨텍스트 길이 초과` 경고

**해결**:
1. max_tokens_context 조정
2. 대화 히스토리 정리
3. 도구 결과 요약 적용
