"""
도구 기본 클래스 및 레지스트리

MCP(Model Context Protocol) 스타일의 도구 인터페이스를 정의합니다.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_string(self) -> str:
        """결과를 문자열로 변환"""
        if self.success:
            return self.content
        else:
            return f"[Error] {self.error}"


@dataclass
class ToolParameter:
    """도구 파라미터 정의"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


class BaseTool(ABC):
    """
    도구 기본 클래스

    모든 도구는 이 클래스를 상속받아 구현합니다.
    MCP 프로토콜과 호환되는 인터페이스를 제공합니다.
    """

    # 서브클래스에서 오버라이드
    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: List[ToolParameter] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        도구 초기화

        Args:
            config: 도구 설정
        """
        self.config = config or {}

    @abstractmethod
    def call(self, params: Dict[str, Any]) -> ToolResult:
        """
        도구 실행 (동기)

        Args:
            params: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        pass

    async def acall(self, params: Dict[str, Any]) -> ToolResult:
        """
        도구 실행 (비동기)

        기본 구현은 동기 메서드를 호출합니다.
        비동기 처리가 필요한 도구는 이 메서드를 오버라이드하세요.

        Args:
            params: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        return self.call(params)

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """
        파라미터 유효성 검사

        Args:
            params: 검사할 파라미터

        Returns:
            에러 메시지 (유효하면 None)
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return f"필수 파라미터 누락: {param.name}"

            if param.name in params and param.enum:
                if params[param.name] not in param.enum:
                    return f"파라미터 '{param.name}'은(는) {param.enum} 중 하나여야 합니다"

        return None

    def get_schema(self) -> Dict[str, Any]:
        """
        JSON Schema 형태로 도구 스키마 반환

        Returns:
            도구 스키마 딕셔너리
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def get_prompt_description(self) -> str:
        """
        프롬프트에 포함될 도구 설명 생성

        Returns:
            도구 설명 문자열
        """
        return json.dumps(self.get_schema(), ensure_ascii=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class ToolRegistry:
    """
    도구 레지스트리

    사용 가능한 모든 도구를 관리합니다.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        도구 등록

        Args:
            tool: 등록할 도구
        """
        if tool.name in self._tools:
            logger.warning(f"도구 '{tool.name}'이(가) 이미 등록되어 있습니다. 덮어씁니다.")
        self._tools[tool.name] = tool
        logger.debug(f"도구 등록: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """
        이름으로 도구 조회

        Args:
            name: 도구 이름

        Returns:
            도구 인스턴스 (없으면 None)
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        등록된 모든 도구 이름 반환

        Returns:
            도구 이름 리스트
        """
        return list(self._tools.keys())

    def get_all(self) -> Dict[str, BaseTool]:
        """
        모든 도구 반환

        Returns:
            도구 이름-인스턴스 딕셔너리
        """
        return self._tools.copy()

    def call_tool(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """
        이름으로 도구 호출

        Args:
            name: 도구 이름
            params: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                content="",
                error=f"도구를 찾을 수 없습니다: {name}"
            )

        # 파라미터 유효성 검사
        error = tool.validate_params(params)
        if error:
            return ToolResult(
                success=False,
                content="",
                error=error
            )

        try:
            return tool.call(params)
        except Exception as e:
            logger.error(f"도구 '{name}' 실행 오류: {e}")
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )

    async def acall_tool(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """
        이름으로 도구 비동기 호출

        Args:
            name: 도구 이름
            params: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                content="",
                error=f"도구를 찾을 수 없습니다: {name}"
            )

        try:
            return await tool.acall(params)
        except Exception as e:
            logger.error(f"도구 '{name}' 비동기 실행 오류: {e}")
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        모든 도구의 스키마 반환

        Returns:
            도구 스키마 리스트
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def get_prompt_descriptions(self) -> str:
        """
        모든 도구의 프롬프트 설명 생성

        Returns:
            도구 설명 문자열
        """
        descriptions = [tool.get_prompt_description() for tool in self._tools.values()]
        return "\n".join(descriptions)


# 전역 레지스트리 인스턴스
default_registry = ToolRegistry()


def register_tool(tool_class: Type[BaseTool], config: Optional[Dict[str, Any]] = None) -> BaseTool:
    """
    데코레이터/함수로 도구 등록

    Args:
        tool_class: 도구 클래스
        config: 도구 설정

    Returns:
        등록된 도구 인스턴스
    """
    tool = tool_class(config)
    default_registry.register(tool)
    return tool
