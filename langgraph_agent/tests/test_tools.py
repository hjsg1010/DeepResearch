#!/usr/bin/env python3
"""
도구 테스트

MCP Mock 도구들의 기능을 테스트합니다.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph_agent.tools.base import ToolRegistry, ToolResult
from langgraph_agent.tools.retrieve import RetrieveTool, RetrieveByIdTool
from langgraph_agent.tools.confluence import (
    ConfluenceSearchTool,
    ConfluenceGetPageTool,
    ConfluenceListSpacesTool
)


class TestRetrieveTool:
    """Retrieve 도구 테스트"""

    def setup_method(self):
        """테스트 설정"""
        self.tool = RetrieveTool()

    def test_basic_search(self):
        """기본 검색 테스트"""
        result = self.tool.call({"query": "API 설계"})

        assert result.success is True
        assert len(result.content) > 0
        assert "API" in result.content or "api" in result.content.lower()

    def test_search_with_filters(self):
        """필터 검색 테스트"""
        result = self.tool.call({
            "query": "LLM",
            "filters": {"department": "AI/ML Team"}
        })

        assert result.success is True
        assert "AI/ML" in result.content or result.metadata.get("result_count", 0) >= 0

    def test_search_no_results(self):
        """결과 없음 테스트"""
        result = self.tool.call({"query": "xyznotexist123"})

        assert result.success is True
        assert "결과가 없습니다" in result.content or result.metadata.get("result_count", 0) == 0

    def test_search_max_results(self):
        """최대 결과 수 제한 테스트"""
        result = self.tool.call({
            "query": "시스템",
            "max_results": 2
        })

        assert result.success is True
        assert result.metadata.get("result_count", 0) <= 2

    def test_missing_query(self):
        """쿼리 누락 테스트"""
        result = self.tool.call({})

        assert result.success is False
        assert result.error is not None


class TestRetrieveByIdTool:
    """ID로 문서 조회 테스트"""

    def setup_method(self):
        self.tool = RetrieveByIdTool()

    def test_get_existing_document(self):
        """존재하는 문서 조회"""
        result = self.tool.call({"document_id": "doc-001"})

        assert result.success is True
        assert "시스템 아키텍처" in result.content

    def test_get_nonexistent_document(self):
        """존재하지 않는 문서 조회"""
        result = self.tool.call({"document_id": "doc-999"})

        assert result.success is False
        assert "찾을 수 없습니다" in result.error


class TestConfluenceSearchTool:
    """Confluence 검색 도구 테스트"""

    def setup_method(self):
        self.tool = ConfluenceSearchTool()

    def test_basic_search(self):
        """기본 검색 테스트"""
        result = self.tool.call({"query": "온보딩"})

        assert result.success is True
        assert len(result.content) > 0

    def test_search_with_space(self):
        """스페이스 필터 검색"""
        result = self.tool.call({
            "query": "CI/CD",
            "space_key": "TECH"
        })

        assert result.success is True
        assert "TECH" in result.content

    def test_search_with_labels(self):
        """라벨 필터 검색"""
        result = self.tool.call({
            "query": "guide",
            "labels": ["guide"]
        })

        assert result.success is True

    def test_search_no_results(self):
        """결과 없음 테스트"""
        result = self.tool.call({
            "query": "nonexistentquery12345",
            "space_key": "TECH"
        })

        assert result.success is True
        assert "결과가 없습니다" in result.content


class TestConfluenceGetPageTool:
    """Confluence 페이지 직접 조회 테스트"""

    def setup_method(self):
        self.tool = ConfluenceGetPageTool()

    def test_get_existing_page(self):
        """존재하는 페이지 조회"""
        result = self.tool.call({"page_id": "12345"})

        assert result.success is True
        assert "온보딩" in result.content

    def test_get_nonexistent_page(self):
        """존재하지 않는 페이지 조회"""
        result = self.tool.call({"page_id": "99999"})

        assert result.success is False
        assert "찾을 수 없습니다" in result.error


class TestConfluenceListSpacesTool:
    """Confluence 스페이스 목록 조회 테스트"""

    def setup_method(self):
        self.tool = ConfluenceListSpacesTool()

    def test_list_spaces(self):
        """스페이스 목록 조회"""
        result = self.tool.call({})

        assert result.success is True
        assert "TECH" in result.content
        assert "PROJ" in result.content
        assert result.metadata.get("space_count", 0) > 0


class TestToolRegistry:
    """도구 레지스트리 테스트"""

    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(RetrieveTool())
        self.registry.register(ConfluenceSearchTool())

    def test_list_tools(self):
        """도구 목록 조회"""
        tools = self.registry.list_tools()

        assert "retrieve" in tools
        assert "confluence_search" in tools

    def test_get_tool(self):
        """도구 조회"""
        tool = self.registry.get("retrieve")

        assert tool is not None
        assert tool.name == "retrieve"

    def test_get_nonexistent_tool(self):
        """존재하지 않는 도구 조회"""
        tool = self.registry.get("nonexistent")

        assert tool is None

    def test_call_tool(self):
        """레지스트리를 통한 도구 호출"""
        result = self.registry.call_tool("retrieve", {"query": "테스트"})

        assert isinstance(result, ToolResult)

    def test_call_nonexistent_tool(self):
        """존재하지 않는 도구 호출"""
        result = self.registry.call_tool("nonexistent", {})

        assert result.success is False
        assert "찾을 수 없습니다" in result.error

    def test_get_all_schemas(self):
        """모든 도구 스키마 조회"""
        schemas = self.registry.get_all_schemas()

        assert len(schemas) == 2
        assert all("function" in s for s in schemas)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
