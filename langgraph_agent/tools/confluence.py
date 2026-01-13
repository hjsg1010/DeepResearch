"""
MCP Confluence Search Tool (Mock 구현)

Confluence 위키에서 문서를 검색하는 MCP 도구입니다.
실제 환경에서는 Confluence REST API와 연동됩니다.

이 mock 구현은 실제 Confluence 연동 전 테스트 및 개발용입니다.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph_agent.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# Mock Confluence 데이터
MOCK_CONFLUENCE_SPACES = {
    "TECH": {
        "name": "기술 문서",
        "description": "기술 관련 문서 공간"
    },
    "PROJ": {
        "name": "프로젝트",
        "description": "프로젝트 관련 문서"
    },
    "HR": {
        "name": "인사/총무",
        "description": "인사 및 총무 관련 문서"
    },
    "GUIDE": {
        "name": "가이드",
        "description": "각종 가이드 문서"
    }
}

MOCK_CONFLUENCE_PAGES = [
    {
        "id": "12345",
        "title": "신규 입사자 온보딩 가이드",
        "space_key": "HR",
        "content": """# 신규 입사자 온보딩 가이드

## 첫째 주
1. **계정 설정**
   - 사내 이메일 설정
   - Slack 가입
   - GitHub 계정 연동
   - VPN 설정

2. **필수 교육 이수**
   - 보안 교육 (2시간)
   - 사내 시스템 교육 (1시간)
   - 팀 소개

## 둘째 주
1. **개발 환경 설정**
   - 개발 도구 설치
   - 저장소 접근 권한 요청
   - 로컬 개발 환경 구축

2. **첫 번째 태스크**
   - 간단한 버그 수정
   - 코드 리뷰 참여

## 체크리스트
- [ ] 계정 설정 완료
- [ ] 필수 교육 이수
- [ ] 개발 환경 구축
- [ ] 멘토 배정
- [ ] 첫 PR 제출
""",
        "author": "HR Team",
        "created_date": "2024-01-10",
        "last_modified": "2024-07-15",
        "labels": ["onboarding", "guide", "new-hire"]
    },
    {
        "id": "12346",
        "title": "CI/CD 파이프라인 구성 가이드",
        "space_key": "TECH",
        "content": """# CI/CD 파이프라인 구성 가이드

## 개요
GitHub Actions 기반 CI/CD 파이프라인 구성 방법

## 파이프라인 구조

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: npm run build

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: ./deploy.sh
```

## 환경 변수 설정
- GitHub Secrets에 민감한 정보 저장
- `.env.example` 파일로 필요한 변수 문서화

## 배포 전략
1. **Staging**: develop 브랜치 푸시 시 자동 배포
2. **Production**: main 브랜치 머지 후 수동 승인

## 모니터링
- Slack 알림 연동
- 배포 실패 시 자동 롤백
""",
        "author": "DevOps Team",
        "created_date": "2024-02-20",
        "last_modified": "2024-08-01",
        "labels": ["ci-cd", "devops", "github-actions"]
    },
    {
        "id": "12347",
        "title": "데이터베이스 마이그레이션 절차",
        "space_key": "TECH",
        "content": """# 데이터베이스 마이그레이션 절차

## 사전 준비
1. 현재 스키마 백업
2. 마이그레이션 스크립트 리뷰
3. 롤백 계획 수립

## 마이그레이션 단계

### 1. 스테이징 환경 테스트
```bash
# 마이그레이션 드라이런
python manage.py migrate --plan

# 마이그레이션 실행
python manage.py migrate
```

### 2. 데이터 검증
- 레코드 수 비교
- 주요 쿼리 결과 검증
- 성능 테스트

### 3. 프로덕션 적용
1. 유지보수 공지
2. 트래픽 차단 (필요시)
3. 백업 생성
4. 마이그레이션 실행
5. 검증
6. 트래픽 복구

## 롤백 절차
```bash
# 롤백
python manage.py migrate app_name 0001_previous_migration
```

## 체크리스트
- [ ] 백업 완료
- [ ] 스테이징 테스트 완료
- [ ] 롤백 스크립트 준비
- [ ] 관련 팀 공지
- [ ] 모니터링 대시보드 확인
""",
        "author": "DBA Team",
        "created_date": "2024-03-05",
        "last_modified": "2024-06-20",
        "labels": ["database", "migration", "procedure"]
    },
    {
        "id": "12348",
        "title": "코드 리뷰 가이드라인",
        "space_key": "GUIDE",
        "content": """# 코드 리뷰 가이드라인

## 목적
- 코드 품질 향상
- 지식 공유
- 버그 사전 발견

## 리뷰어 가이드

### 무엇을 확인할 것인가?
1. **기능적 정확성**: 요구사항 충족 여부
2. **코드 품질**: 가독성, 유지보수성
3. **테스트 커버리지**: 적절한 테스트 존재
4. **보안**: 잠재적 보안 취약점
5. **성능**: 명백한 성능 이슈

### 피드백 작성법
- 구체적으로 작성
- 개선 방향 제시
- 긍정적인 부분도 언급
- 질문 형태로 토론 유도

## 작성자 가이드

### PR 작성 시
1. 작은 단위로 분리
2. 명확한 제목과 설명
3. 셀프 리뷰 먼저 수행
4. 테스트 통과 확인

### 피드백 수용
- 열린 마음으로 수용
- 불명확한 점은 질문
- 변경 사항 설명

## 리뷰 SLA
- 일반 PR: 24시간 이내
- 긴급 PR: 4시간 이내
- 대규모 변경: 48시간 이내
""",
        "author": "Engineering Team",
        "created_date": "2024-01-25",
        "last_modified": "2024-05-10",
        "labels": ["code-review", "best-practices", "guide"]
    },
    {
        "id": "12349",
        "title": "LLM 서비스 운영 가이드",
        "space_key": "TECH",
        "content": """# LLM 서비스 운영 가이드

## 아키텍처

```
사용자 → API Gateway → LLM Service → vLLM/TGI Server
                           ↓
                      Rate Limiter
                           ↓
                       Monitoring
```

## 서버 구성

### vLLM 서버 실행
```bash
python -m vllm.entrypoints.openai.api_server \\
    --model /models/llama-3-70b \\
    --tensor-parallel-size 4 \\
    --max-model-len 8192 \\
    --port 8000
```

### 환경 변수
- `MODEL_PATH`: 모델 경로
- `MAX_BATCH_SIZE`: 최대 배치 크기
- `GPU_MEMORY_UTILIZATION`: GPU 메모리 사용률

## 모니터링

### 주요 메트릭
1. **처리량**: 초당 토큰 수
2. **지연시간**: P50, P95, P99
3. **GPU 사용률**: 메모리 및 연산
4. **큐 길이**: 대기 중인 요청 수

### 알림 설정
- GPU 메모리 > 90%: Warning
- 지연시간 P95 > 5s: Critical
- 에러율 > 1%: Critical

## 트러블슈팅

### OOM 발생 시
1. 배치 크기 감소
2. max_model_len 조정
3. GPU 메모리 확보

### 느린 응답
1. KV 캐시 상태 확인
2. 동시 요청 수 확인
3. 모델 최적화 검토

## 스케일링 전략
- 수평 확장: 서버 추가
- 수직 확장: GPU 업그레이드
- 로드 밸런싱: Round-robin 또는 Least-connections
""",
        "author": "AI Platform Team",
        "created_date": "2024-04-01",
        "last_modified": "2024-08-10",
        "labels": ["llm", "vllm", "operations", "ai"]
    },
    {
        "id": "12350",
        "title": "프로젝트 알파 - 기술 명세서",
        "space_key": "PROJ",
        "content": """# 프로젝트 알파 - 기술 명세서

## 프로젝트 개요
차세대 추천 시스템 구축

## 기술 스택
- **Backend**: Python, FastAPI
- **ML**: PyTorch, Transformers
- **Infrastructure**: Kubernetes, ArgoCD
- **Database**: PostgreSQL, Redis, Elasticsearch

## 시스템 요구사항
- 일 1억 요청 처리
- 응답 시간 < 100ms (P95)
- 가용성 99.9%

## 마일스톤
1. **Phase 1** (완료): 기반 인프라 구축
2. **Phase 2** (진행중): 추천 모델 개발
3. **Phase 3** (예정): A/B 테스트 및 배포
4. **Phase 4** (예정): 운영 안정화

## 팀 구성
- PM: 홍길동
- Tech Lead: 김개발
- Backend: 3명
- ML: 2명
- DevOps: 1명

## 위험 요소
1. 모델 학습 데이터 품질
2. 실시간 처리 성능
3. 타 시스템 연동 일정
""",
        "author": "Project Alpha Team",
        "created_date": "2024-05-15",
        "last_modified": "2024-08-05",
        "labels": ["project", "alpha", "recommendation", "spec"]
    }
]


class ConfluenceSearchTool(BaseTool):
    """
    MCP Confluence Search Tool

    Confluence 위키에서 문서를 검색합니다.
    - CQL(Confluence Query Language) 스타일 검색
    - 스페이스 필터링
    - 라벨 필터링
    """

    name = "confluence_search"
    description = """Confluence 위키에서 문서를 검색합니다.
사내 위키, 프로젝트 문서, 가이드라인 등을 검색할 수 있습니다.
스페이스나 라벨로 필터링하여 원하는 문서를 찾을 수 있습니다."""

    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="검색 쿼리 (키워드 또는 문장)",
            required=True
        ),
        ToolParameter(
            name="space_key",
            type="string",
            description="검색할 Confluence 스페이스 키 (예: TECH, PROJ, HR, GUIDE). 생략 시 전체 검색",
            required=False
        ),
        ToolParameter(
            name="labels",
            type="array",
            description="필터링할 라벨 목록",
            required=False
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="최대 반환 결과 수 (기본값: 5)",
            required=False,
            default=5
        ),
        ToolParameter(
            name="include_content",
            type="boolean",
            description="문서 전체 내용 포함 여부 (기본값: True)",
            required=False,
            default=True
        )
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pages = MOCK_CONFLUENCE_PAGES
        self.spaces = MOCK_CONFLUENCE_SPACES

    def _search_pages(
        self,
        query: str,
        space_key: Optional[str] = None,
        labels: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """페이지 검색"""
        query_lower = query.lower()
        query_words = query_lower.split()

        results = []

        for page in self.pages:
            # 스페이스 필터
            if space_key and page["space_key"] != space_key:
                continue

            # 라벨 필터
            if labels:
                page_labels = set(page.get("labels", []))
                if not any(label in page_labels for label in labels):
                    continue

            # 점수 계산
            score = 0
            searchable_text = (
                page["title"].lower() + " " +
                page["content"].lower() + " " +
                " ".join(page.get("labels", []))
            )

            for word in query_words:
                if word in searchable_text:
                    # 제목에 있으면 더 높은 점수
                    if word in page["title"].lower():
                        score += 3
                    else:
                        score += 1

            if score > 0:
                results.append({
                    **page,
                    "score": score,
                    "space_name": self.spaces.get(page["space_key"], {}).get("name", page["space_key"])
                })

        # 점수순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:max_results]

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """
        Confluence 검색 실행

        Args:
            params: {
                "query": 검색 쿼리,
                "space_key": 스페이스 키 (선택),
                "labels": 라벨 리스트 (선택),
                "max_results": 최대 결과 수 (선택),
                "include_content": 내용 포함 여부 (선택)
            }

        Returns:
            ToolResult: 검색 결과
        """
        query = params.get("query", "")
        space_key = params.get("space_key")
        labels = params.get("labels")
        max_results = params.get("max_results", 5)
        include_content = params.get("include_content", True)

        if not query:
            return ToolResult(
                success=False,
                content="",
                error="검색 쿼리가 필요합니다"
            )

        logger.info(f"[Confluence] 검색 실행: query='{query}', space={space_key}, labels={labels}")

        try:
            results = self._search_pages(query, space_key, labels, max_results)

            if not results:
                filter_info = ""
                if space_key:
                    filter_info += f" (스페이스: {space_key})"
                if labels:
                    filter_info += f" (라벨: {', '.join(labels)})"

                return ToolResult(
                    success=True,
                    content=f"'{query}'에 대한 Confluence 검색 결과가 없습니다{filter_info}.",
                    metadata={"query": query, "result_count": 0}
                )

            # 결과 포맷팅
            output_parts = [f"Confluence 검색 결과: '{query}' ({len(results)}건)\n"]

            for i, page in enumerate(results, 1):
                output_parts.append(f"\n## {i}. {page['title']}")
                output_parts.append(f"- 페이지 ID: {page['id']}")
                output_parts.append(f"- 스페이스: {page['space_name']} ({page['space_key']})")
                output_parts.append(f"- 작성자: {page['author']}")
                output_parts.append(f"- 라벨: {', '.join(page.get('labels', []))}")
                output_parts.append(f"- 최종 수정일: {page['last_modified']}")

                if include_content:
                    # 내용이 너무 길면 요약
                    content = page['content']
                    if len(content) > 2000:
                        content = content[:2000] + "\n\n... (이하 생략)"
                    output_parts.append(f"\n### 내용:\n{content}")

                output_parts.append("\n---")

            content = "\n".join(output_parts)

            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "query": query,
                    "space_key": space_key,
                    "result_count": len(results),
                    "page_ids": [page["id"] for page in results]
                }
            )

        except Exception as e:
            logger.error(f"[Confluence] 검색 오류: {e}")
            return ToolResult(
                success=False,
                content="",
                error=f"Confluence 검색 중 오류 발생: {str(e)}"
            )


class ConfluenceGetPageTool(BaseTool):
    """
    Confluence 페이지 직접 조회 도구
    """

    name = "confluence_get_page"
    description = "Confluence 페이지 ID로 특정 페이지를 직접 조회합니다."

    parameters = [
        ToolParameter(
            name="page_id",
            type="string",
            description="조회할 Confluence 페이지 ID",
            required=True
        )
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pages = {page["id"]: page for page in MOCK_CONFLUENCE_PAGES}
        self.spaces = MOCK_CONFLUENCE_SPACES

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """페이지 ID로 조회"""
        page_id = params.get("page_id", "")

        if not page_id:
            return ToolResult(
                success=False,
                content="",
                error="페이지 ID가 필요합니다"
            )

        page = self.pages.get(page_id)
        if not page:
            return ToolResult(
                success=False,
                content="",
                error=f"페이지를 찾을 수 없습니다: {page_id}"
            )

        space_name = self.spaces.get(page["space_key"], {}).get("name", page["space_key"])

        content = f"""# {page['title']}

**페이지 ID**: {page['id']}
**스페이스**: {space_name} ({page['space_key']})
**작성자**: {page['author']}
**라벨**: {', '.join(page.get('labels', []))}
**작성일**: {page['created_date']}
**수정일**: {page['last_modified']}

---

{page['content']}
"""

        return ToolResult(
            success=True,
            content=content,
            metadata={"page_id": page_id, "space_key": page["space_key"]}
        )


class ConfluenceListSpacesTool(BaseTool):
    """
    사용 가능한 Confluence 스페이스 목록 조회
    """

    name = "confluence_list_spaces"
    description = "사용 가능한 Confluence 스페이스 목록을 조회합니다."

    parameters = []

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.spaces = MOCK_CONFLUENCE_SPACES

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """스페이스 목록 조회"""
        output_parts = ["# 사용 가능한 Confluence 스페이스\n"]

        for key, space in self.spaces.items():
            output_parts.append(f"- **{key}**: {space['name']}")
            output_parts.append(f"  - {space['description']}")

        return ToolResult(
            success=True,
            content="\n".join(output_parts),
            metadata={"space_count": len(self.spaces)}
        )
