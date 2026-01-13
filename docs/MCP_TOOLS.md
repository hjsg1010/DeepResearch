# MCP 도구 가이드

## 1. 개요

MCP(Model Context Protocol)는 LLM 에이전트가 외부 도구와 상호작용하기 위한 프로토콜입니다.
이 프로젝트에서는 회사 내부 시스템과 연동하기 위해 MCP 스타일의 도구를 구현합니다.

현재 구현된 도구는 **Mock 버전**이며, 실제 배포 시 MCP 서버와 연동해야 합니다.

---

## 2. 구현된 도구

### 2.1 RetrieveTool (문서 검색)

#### 기능
사내 문서 저장소에서 키워드 기반으로 관련 문서를 검색합니다.

#### 파라미터

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| query | string | O | 검색 쿼리 (키워드 또는 자연어) |
| filters | object | X | 필터 조건 (부서, 태그 등) |
| max_results | integer | X | 최대 결과 수 (기본: 5) |
| include_content | boolean | X | 내용 포함 여부 (기본: True) |

#### 사용 예시

```json
{
  "name": "retrieve",
  "arguments": {
    "query": "API 설계 가이드",
    "filters": {"department": "Backend Team"},
    "max_results": 3
  }
}
```

#### 반환 형식

```
'API 설계 가이드'에 대한 검색 결과 (2건):

## 1. API 설계 가이드라인
- 문서 ID: doc-002
- 작성자: 이개발자
- 부서: Backend Team
- 태그: api, rest, guidelines
- 최종 수정일: 2024-05-15

### 내용:
# API 설계 가이드라인
...
```

---

### 2.2 RetrieveByIdTool (문서 직접 조회)

#### 기능
문서 ID로 특정 문서를 직접 조회합니다.

#### 파라미터

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| document_id | string | O | 조회할 문서 ID |

#### 사용 예시

```json
{
  "name": "retrieve_by_id",
  "arguments": {
    "document_id": "doc-001"
  }
}
```

---

### 2.3 ConfluenceSearchTool (Confluence 검색)

#### 기능
Confluence 위키에서 문서를 검색합니다.

#### 파라미터

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| query | string | O | 검색 쿼리 |
| space_key | string | X | 스페이스 키 (TECH, PROJ, HR, GUIDE) |
| labels | array | X | 라벨 필터 목록 |
| max_results | integer | X | 최대 결과 수 (기본: 5) |
| include_content | boolean | X | 내용 포함 여부 (기본: True) |

#### 사용 예시

```json
{
  "name": "confluence_search",
  "arguments": {
    "query": "배포 프로세스",
    "space_key": "TECH",
    "labels": ["ci-cd"]
  }
}
```

#### 반환 형식

```
Confluence 검색 결과: '배포 프로세스' (1건)

## 1. CI/CD 파이프라인 구성 가이드
- 페이지 ID: 12346
- 스페이스: 기술 문서 (TECH)
- 작성자: DevOps Team
- 라벨: ci-cd, devops, github-actions
- 최종 수정일: 2024-08-01

### 내용:
# CI/CD 파이프라인 구성 가이드
...
```

---

### 2.4 ConfluenceGetPageTool (페이지 직접 조회)

#### 기능
Confluence 페이지 ID로 특정 페이지를 직접 조회합니다.

#### 파라미터

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| page_id | string | O | 조회할 페이지 ID |

#### 사용 예시

```json
{
  "name": "confluence_get_page",
  "arguments": {
    "page_id": "12345"
  }
}
```

---

### 2.5 ConfluenceListSpacesTool (스페이스 목록)

#### 기능
사용 가능한 Confluence 스페이스 목록을 조회합니다.

#### 파라미터

없음

#### 사용 예시

```json
{
  "name": "confluence_list_spaces",
  "arguments": {}
}
```

#### 반환 형식

```
# 사용 가능한 Confluence 스페이스

- **TECH**: 기술 문서
  - 기술 관련 문서 공간
- **PROJ**: 프로젝트
  - 프로젝트 관련 문서
- **HR**: 인사/총무
  - 인사 및 총무 관련 문서
- **GUIDE**: 가이드
  - 각종 가이드 문서
```

---

## 3. 실제 MCP 서버 연동

### 3.1 MCP 서버 구조

실제 배포 시에는 다음과 같은 MCP 서버가 필요합니다:

```
┌─────────────────────────────────────────────────┐
│              MCP 서버 (예: FastAPI)              │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  /mcp/retrieve                          │    │
│  │  - 문서 저장소 연동                       │    │
│  │  - Elasticsearch / Vector DB 검색        │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  /mcp/confluence                        │    │
│  │  - Confluence REST API 연동             │    │
│  │  - 인증 처리                             │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 3.2 실제 도구로 교체 방법

1. **MCP 클라이언트 구현**

```python
import httpx

class MCPClient:
    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=timeout)

    def call(self, tool_name: str, params: dict) -> dict:
        response = self.client.post(
            f"{self.endpoint}/{tool_name}",
            json=params
        )
        return response.json()
```

2. **실제 RetrieveTool 구현**

```python
class RealRetrieveTool(BaseTool):
    name = "retrieve"
    description = "사내 문서 저장소에서 문서를 검색합니다."

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    def call(self, params: Dict[str, Any]) -> ToolResult:
        try:
            response = self.mcp_client.call("retrieve", params)
            return ToolResult(
                success=True,
                content=self._format_response(response),
                metadata=response.get("metadata", {})
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )
```

3. **레지스트리에 등록**

```python
mcp_client = MCPClient("http://mcp-server:9000")
registry = ToolRegistry()
registry.register(RealRetrieveTool(mcp_client))
registry.register(RealConfluenceSearchTool(mcp_client))
```

---

## 4. Mock 데이터

### 4.1 문서 저장소 (Retrieve)

| 문서 ID | 제목 | 부서 | 태그 |
|---------|------|------|------|
| doc-001 | 시스템 아키텍처 설계 문서 | Platform Engineering | architecture, microservices |
| doc-002 | API 설계 가이드라인 | Backend Team | api, rest, guidelines |
| doc-003 | ML 모델 배포 파이프라인 | AI/ML Team | mlops, deployment, ml |
| doc-004 | 보안 정책 및 가이드 | Security Team | security, policy, compliance |
| doc-005 | LLM 에이전트 개발 가이드 | AI/ML Team | llm, agent, ai, development |

### 4.2 Confluence 스페이스

| 스페이스 키 | 이름 | 설명 |
|-------------|------|------|
| TECH | 기술 문서 | 기술 관련 문서 공간 |
| PROJ | 프로젝트 | 프로젝트 관련 문서 |
| HR | 인사/총무 | 인사 및 총무 관련 문서 |
| GUIDE | 가이드 | 각종 가이드 문서 |

### 4.3 Confluence 페이지

| 페이지 ID | 제목 | 스페이스 | 라벨 |
|-----------|------|----------|------|
| 12345 | 신규 입사자 온보딩 가이드 | HR | onboarding, guide |
| 12346 | CI/CD 파이프라인 구성 가이드 | TECH | ci-cd, devops |
| 12347 | 데이터베이스 마이그레이션 절차 | TECH | database, migration |
| 12348 | 코드 리뷰 가이드라인 | GUIDE | code-review, best-practices |
| 12349 | LLM 서비스 운영 가이드 | TECH | llm, vllm, operations |
| 12350 | 프로젝트 알파 - 기술 명세서 | PROJ | project, alpha, spec |

---

## 5. 새 도구 추가 가이드

### 5.1 도구 클래스 작성

```python
from langgraph_agent.tools.base import BaseTool, ToolParameter, ToolResult

class MyNewTool(BaseTool):
    name = "my_new_tool"
    description = "도구 설명"

    parameters = [
        ToolParameter(
            name="param1",
            type="string",
            description="파라미터 설명",
            required=True
        ),
        ToolParameter(
            name="param2",
            type="integer",
            description="선택 파라미터",
            required=False,
            default=10
        )
    ]

    def call(self, params: Dict[str, Any]) -> ToolResult:
        param1 = params.get("param1")
        param2 = params.get("param2", 10)

        # 로직 구현
        result = f"처리 결과: {param1}, {param2}"

        return ToolResult(
            success=True,
            content=result,
            metadata={"processed": True}
        )
```

### 5.2 도구 등록

```python
from langgraph_agent.tools import ToolRegistry
from my_tools import MyNewTool

registry = ToolRegistry()
registry.register(MyNewTool())
```

### 5.3 에이전트에서 사용

```python
from langgraph_agent import ResearchAgent

agent = ResearchAgent(tool_registry=registry)
result = agent.run("내 새 도구를 사용해서 분석해줘")
```

---

## 6. 도구 개발 모범 사례

### 6.1 에러 처리

```python
def call(self, params: Dict[str, Any]) -> ToolResult:
    try:
        # 파라미터 검증
        if not params.get("query"):
            return ToolResult(
                success=False,
                content="",
                error="query 파라미터가 필요합니다"
            )

        # 실행
        result = self._execute(params)

        return ToolResult(success=True, content=result)

    except TimeoutError:
        return ToolResult(
            success=False,
            content="",
            error="요청 시간 초과"
        )
    except Exception as e:
        return ToolResult(
            success=False,
            content="",
            error=f"예상치 못한 오류: {str(e)}"
        )
```

### 6.2 결과 포맷팅

```python
def _format_results(self, results: List[Dict]) -> str:
    if not results:
        return "검색 결과가 없습니다."

    parts = [f"검색 결과 ({len(results)}건):\n"]

    for i, item in enumerate(results, 1):
        parts.append(f"\n## {i}. {item['title']}")
        parts.append(f"- ID: {item['id']}")
        parts.append(f"- 내용: {item['content'][:200]}...")

    return "\n".join(parts)
```

### 6.3 로깅

```python
import logging

logger = logging.getLogger(__name__)

class MyTool(BaseTool):
    def call(self, params: Dict[str, Any]) -> ToolResult:
        logger.info(f"[{self.name}] 호출: {params}")

        result = self._execute(params)

        logger.debug(f"[{self.name}] 결과: {len(result)} 문자")

        return ToolResult(success=True, content=result)
```
