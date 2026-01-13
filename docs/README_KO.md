# LangGraph DeepResearch Agent

## 개요

이 프로젝트는 [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) 에이전트를 **LangGraph** 형태로 재구현한 것입니다.

### 주요 특징

- **LangGraph 기반 워크플로우**: StateGraph를 활용한 선언적 에이전트 구조
- **OpenAI 호환 API 직접 사용**: langchain-openai 미사용, openai 패키지 직접 호출
- **회사망 환경 적합**: 외부 API 대신 MCP 기반 내부 도구 사용
- **확장 가능**: 도구 추가 및 설정 변경 용이

---

## 설치

```bash
# 의존성 설치
pip install -r langgraph_agent/requirements.txt
```

## 환경 설정

```bash
# 환경 변수 설정
export LLM_BASE_URL="http://your-llm-server:8000/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="openai/gpt-oss-120b"
```

---

## 빠른 시작

### Python에서 사용

```python
from langgraph_agent import run_research, AgentConfig

# 기본 실행
result = run_research("시스템 아키텍처에 대해 알려줘")
print(result.answer)

# 커스텀 설정
config = AgentConfig.from_env()
result = run_research("질문", config=config)
```

### CLI 사용

```bash
# 단일 질문
python -m langgraph_agent.main "질문 내용"

# 대화형 모드
python -m langgraph_agent.main --interactive

# 상세 로깅
python -m langgraph_agent.main --verbose "질문"
```

---

## 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                    사용자 질문                         │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph StateGraph                     │
│                                                       │
│   ┌─────────────┐         ┌─────────────┐            │
│   │ agent_node  │◄───────►│  tool_node  │            │
│   │  (LLM 호출) │         │ (도구 실행) │            │
│   └──────┬──────┘         └─────────────┘            │
│          │                                            │
│          ▼                                            │
│   ┌─────────────┐                                    │
│   │    END      │                                    │
│   └─────────────┘                                    │
└──────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ OpenAI Client   │    │  Tool Registry  │
    │ (호환 API 직접) │    │  (MCP 도구들)  │
    └─────────────────┘    └─────────────────┘
```

---

## 핵심 구성 요소

### 1. State (상태)

```python
class AgentState(TypedDict):
    messages: List[Message]          # 대화 히스토리
    question: str                     # 질문
    current_turn: int                 # 현재 턴
    pending_tool_calls: List[ToolCall]  # 대기 중인 도구 호출
    answer: Optional[str]             # 최종 답변
    is_complete: bool                 # 완료 여부
    ...
```

### 2. Nodes (노드)

- **agent_node**: LLM 호출, 응답 파싱
- **tool_node**: 도구 실행, 결과 포맷팅

### 3. Tools (도구)

- **retrieve**: 사내 문서 저장소 검색
- **confluence_search**: Confluence 위키 검색
- (확장 가능)

---

## 구현된 도구

| 도구 | 설명 |
|------|------|
| `retrieve` | 문서 저장소 검색 |
| `retrieve_by_id` | 문서 ID로 직접 조회 |
| `confluence_search` | Confluence 검색 |
| `confluence_get_page` | Confluence 페이지 조회 |
| `confluence_list_spaces` | 스페이스 목록 |

**참고**: 현재 Mock 구현이며, 실제 MCP 서버 연동 필요

---

## 설정

```python
from langgraph_agent import AgentConfig
from langgraph_agent.config import LLMConfig

config = AgentConfig(
    llm=LLMConfig(
        base_url="http://localhost:8000/v1",
        api_key="your-key",
        model_name="openai/gpt-oss-120b",
        temperature=0.6,
    ),
    max_turns=100,
    verbose=True
)
```

---

## 테스트

```bash
# 도구 테스트
python -m pytest langgraph_agent/tests/test_tools.py -v

# 노드 테스트
python -m pytest langgraph_agent/tests/test_nodes.py -v

# 전체 테스트
python -m pytest langgraph_agent/tests/ -v
```

---

## 문서

- [아키텍처 문서](ARCHITECTURE.md)
- [구현 가이드](IMPLEMENTATION_GUIDE.md)
- [MCP 도구 가이드](MCP_TOOLS.md)

---

## 파일 구조

```
langgraph_agent/
├── __init__.py           # 모듈 진입점
├── config.py             # 설정 관리
├── state.py              # State 정의
├── prompts.py            # 프롬프트
├── nodes.py              # LangGraph 노드
├── graph.py              # 그래프 정의
├── main.py               # CLI
├── llm/                  # LLM 클라이언트
├── tools/                # MCP 도구
├── utils/                # 유틸리티
├── tests/                # 테스트
└── examples/             # 예제
```

---

## 라이선스

이 프로젝트는 원본 DeepResearch 프로젝트의 라이선스를 따릅니다.

---

## 기여

버그 리포트, 기능 제안, PR을 환영합니다.
