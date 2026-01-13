"""
MCP Retrieve Tool (Mock 구현)

문서 검색/조회를 위한 MCP 도구입니다.
실제 환경에서는 회사 내부 문서 저장소와 연동됩니다.

이 mock 구현은 실제 MCP 서버 연동 전 테스트 및 개발용입니다.
"""

import json
import logging
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph_agent.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# Mock 데이터: 실제 환경에서는 MCP 서버에서 가져옴
MOCK_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "시스템 아키텍처 설계 문서",
        "content": """# 시스템 아키텍처 개요

## 1. 마이크로서비스 구조
- API Gateway: Kong 기반
- 서비스 메시: Istio
- 컨테이너 오케스트레이션: Kubernetes

## 2. 데이터 레이어
- 주 데이터베이스: PostgreSQL 클러스터
- 캐시: Redis Cluster
- 메시지 큐: Apache Kafka

## 3. 모니터링
- 메트릭: Prometheus + Grafana
- 로깅: ELK Stack
- 트레이싱: Jaeger
""",
        "metadata": {
            "author": "김아키텍트",
            "created_at": "2024-01-15",
            "updated_at": "2024-06-20",
            "tags": ["architecture", "microservices", "kubernetes"],
            "department": "Platform Engineering"
        }
    },
    {
        "id": "doc-002",
        "title": "API 설계 가이드라인",
        "content": """# API 설계 가이드라인

## RESTful API 원칙
1. 자원 중심 설계
2. HTTP 메서드 적절한 사용
3. 상태 코드 표준 준수

## 버전 관리
- URL 경로 버전: /api/v1/resources
- 하위 호환성 유지 필수

## 인증/인가
- OAuth 2.0 + JWT 토큰
- API Key는 외부 연동용으로만 사용

## 에러 응답 형식
```json
{
    "error": {
        "code": "ERR_001",
        "message": "Human readable message",
        "details": {}
    }
}
```
""",
        "metadata": {
            "author": "이개발자",
            "created_at": "2024-02-10",
            "updated_at": "2024-05-15",
            "tags": ["api", "rest", "guidelines"],
            "department": "Backend Team"
        }
    },
    {
        "id": "doc-003",
        "title": "ML 모델 배포 파이프라인",
        "content": """# ML 모델 배포 파이프라인

## 개요
MLOps 파이프라인을 통한 모델 자동 배포 프로세스

## 파이프라인 단계
1. **모델 학습**: MLflow로 실험 추적
2. **모델 검증**: A/B 테스트 및 성능 검증
3. **모델 등록**: Model Registry에 등록
4. **배포**: Kubernetes + Seldon Core

## 모니터링
- 모델 드리프트 감지
- 추론 지연시간 모니터링
- 예측 분포 분석

## 롤백 전략
- Blue-Green 배포
- 자동 롤백 트리거 조건 정의
""",
        "metadata": {
            "author": "박ML엔지니어",
            "created_at": "2024-03-01",
            "updated_at": "2024-07-10",
            "tags": ["mlops", "deployment", "kubernetes", "ml"],
            "department": "AI/ML Team"
        }
    },
    {
        "id": "doc-004",
        "title": "보안 정책 및 가이드",
        "content": """# 보안 정책

## 접근 제어
- RBAC(Role-Based Access Control) 적용
- 최소 권한 원칙 준수
- MFA 필수

## 데이터 보안
- 저장 데이터 암호화 (AES-256)
- 전송 데이터 암호화 (TLS 1.3)
- PII 데이터 마스킹

## 네트워크 보안
- VPC 분리
- 방화벽 규칙 화이트리스트 방식
- DDoS 보호

## 감사 로깅
- 모든 관리 작업 로깅
- 로그 보관 기간: 1년
- SIEM 연동
""",
        "metadata": {
            "author": "최보안담당자",
            "created_at": "2024-01-05",
            "updated_at": "2024-08-01",
            "tags": ["security", "policy", "compliance"],
            "department": "Security Team"
        }
    },
    {
        "id": "doc-005",
        "title": "LLM 에이전트 개발 가이드",
        "content": """# LLM 에이전트 개발 가이드

## 개요
사내 LLM 기반 에이전트 개발을 위한 표준 가이드

## 아키텍처 패턴
1. **ReAct 패턴**: 추론-행동 반복
2. **Plan-and-Execute**: 계획 수립 후 실행
3. **Multi-Agent**: 역할별 에이전트 협업

## 프롬프트 엔지니어링
- System Prompt 표준 템플릿 사용
- Few-shot 예제 포함
- 출력 형식 명시

## 도구 설계
- MCP(Model Context Protocol) 준수
- 도구 설명 명확하게 작성
- 에러 핸들링 필수

## 평가 지표
- 정확도 (Accuracy)
- 태스크 완료율
- 응답 시간
- 할루시네이션 비율
""",
        "metadata": {
            "author": "정AI연구원",
            "created_at": "2024-04-15",
            "updated_at": "2024-08-15",
            "tags": ["llm", "agent", "ai", "development"],
            "department": "AI/ML Team"
        }
    }
]


class RetrieveTool(BaseTool):
    """
    MCP Retrieve Tool

    문서 저장소에서 관련 문서를 검색하고 조회합니다.
    - 키워드 기반 검색
    - 시맨틱 검색 (mock에서는 키워드 매칭으로 시뮬레이션)
    - 메타데이터 필터링
    """

    name = "retrieve"
    description = """문서 저장소에서 관련 문서를 검색합니다.
키워드나 질문을 기반으로 관련 문서를 찾아 내용을 반환합니다.
사내 기술 문서, 가이드라인, 정책 문서 등을 검색할 수 있습니다."""

    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="검색 쿼리 (키워드 또는 자연어 질문)",
            required=True
        ),
        ToolParameter(
            name="filters",
            type="object",
            description="검색 필터 (예: {'department': 'AI/ML Team', 'tags': ['llm']})",
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
        self.documents = MOCK_DOCUMENTS
        self._build_index()

    def _build_index(self):
        """간단한 역인덱스 구축"""
        self.keyword_index: Dict[str, List[str]] = {}

        for doc in self.documents:
            doc_id = doc["id"]
            # 제목과 내용에서 키워드 추출
            text = (doc["title"] + " " + doc["content"]).lower()
            words = set(text.replace("\n", " ").replace("#", "").split())

            for word in words:
                if len(word) > 2:  # 짧은 단어 제외
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(doc_id)

            # 태그도 인덱싱
            for tag in doc["metadata"].get("tags", []):
                if tag not in self.keyword_index:
                    self.keyword_index[tag] = []
                self.keyword_index[tag].append(doc_id)

    def _search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """문서 검색"""
        query_words = query.lower().split()

        # 각 문서의 점수 계산
        scores: Dict[str, float] = {}
        for word in query_words:
            for indexed_word, doc_ids in self.keyword_index.items():
                if word in indexed_word or indexed_word in word:
                    for doc_id in doc_ids:
                        scores[doc_id] = scores.get(doc_id, 0) + 1

        # 필터 적용
        if filters:
            filtered_scores = {}
            for doc_id, score in scores.items():
                doc = next((d for d in self.documents if d["id"] == doc_id), None)
                if doc:
                    match = True
                    for key, value in filters.items():
                        doc_value = doc["metadata"].get(key)
                        if isinstance(value, list):
                            if not doc_value or not any(v in doc_value for v in value):
                                match = False
                        elif doc_value != value:
                            match = False
                    if match:
                        filtered_scores[doc_id] = score
            scores = filtered_scores

        # 점수순 정렬
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 결과 구성
        results = []
        for doc_id in sorted_doc_ids[:max_results]:
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc:
                results.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": scores[doc_id]
                })

        return results

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """
        문서 검색 실행

        Args:
            params: {
                "query": 검색 쿼리,
                "filters": 검색 필터 (선택),
                "max_results": 최대 결과 수 (선택),
                "include_content": 내용 포함 여부 (선택)
            }

        Returns:
            ToolResult: 검색 결과
        """
        query = params.get("query", "")
        filters = params.get("filters")
        max_results = params.get("max_results", 5)
        include_content = params.get("include_content", True)

        if not query:
            return ToolResult(
                success=False,
                content="",
                error="검색 쿼리가 필요합니다"
            )

        logger.info(f"[Retrieve] 검색 실행: query='{query}', filters={filters}")

        try:
            results = self._search(query, filters, max_results)

            if not results:
                return ToolResult(
                    success=True,
                    content=f"'{query}'에 대한 검색 결과가 없습니다.",
                    metadata={"query": query, "result_count": 0}
                )

            # 결과 포맷팅
            output_parts = [f"'{query}'에 대한 검색 결과 ({len(results)}건):\n"]

            for i, doc in enumerate(results, 1):
                output_parts.append(f"\n## {i}. {doc['title']}")
                output_parts.append(f"- 문서 ID: {doc['id']}")
                output_parts.append(f"- 작성자: {doc['metadata'].get('author', 'Unknown')}")
                output_parts.append(f"- 부서: {doc['metadata'].get('department', 'Unknown')}")
                output_parts.append(f"- 태그: {', '.join(doc['metadata'].get('tags', []))}")
                output_parts.append(f"- 최종 수정일: {doc['metadata'].get('updated_at', 'Unknown')}")

                if include_content:
                    output_parts.append(f"\n### 내용:\n{doc['content']}")
                output_parts.append("\n---")

            content = "\n".join(output_parts)

            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "document_ids": [doc["id"] for doc in results]
                }
            )

        except Exception as e:
            logger.error(f"[Retrieve] 검색 오류: {e}")
            return ToolResult(
                success=False,
                content="",
                error=f"검색 중 오류 발생: {str(e)}"
            )


class RetrieveByIdTool(BaseTool):
    """
    문서 ID로 직접 조회하는 도구
    """

    name = "retrieve_by_id"
    description = "문서 ID로 특정 문서를 직접 조회합니다."

    parameters = [
        ToolParameter(
            name="document_id",
            type="string",
            description="조회할 문서 ID",
            required=True
        )
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.documents = {doc["id"]: doc for doc in MOCK_DOCUMENTS}

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """문서 ID로 조회"""
        doc_id = params.get("document_id", "")

        if not doc_id:
            return ToolResult(
                success=False,
                content="",
                error="문서 ID가 필요합니다"
            )

        doc = self.documents.get(doc_id)
        if not doc:
            return ToolResult(
                success=False,
                content="",
                error=f"문서를 찾을 수 없습니다: {doc_id}"
            )

        content = f"""# {doc['title']}

**문서 ID**: {doc['id']}
**작성자**: {doc['metadata'].get('author', 'Unknown')}
**부서**: {doc['metadata'].get('department', 'Unknown')}
**태그**: {', '.join(doc['metadata'].get('tags', []))}
**작성일**: {doc['metadata'].get('created_at', 'Unknown')}
**수정일**: {doc['metadata'].get('updated_at', 'Unknown')}

---

{doc['content']}
"""

        return ToolResult(
            success=True,
            content=content,
            metadata={"document_id": doc_id}
        )
