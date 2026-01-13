"""
텍스트 파싱 유틸리티
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    텍스트에서 JSON 추출

    Args:
        text: JSON을 포함한 텍스트

    Returns:
        파싱된 JSON 객체 (없거나 실패 시 None)
    """
    # 먼저 전체 텍스트가 JSON인지 확인
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # ```json ... ``` 블록 찾기
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # { } 패턴 찾기
    brace_pattern = r"\{[^{}]*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    # 중첩된 JSON 처리
    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

    return None


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    텍스트에서 코드 블록 추출

    Args:
        text: 코드 블록을 포함한 텍스트
        language: 특정 언어만 추출 (None이면 모두)

    Returns:
        코드 블록 리스트
    """
    if language:
        pattern = rf"```{language}\s*(.*?)\s*```"
    else:
        pattern = r"```(?:\w+)?\s*(.*?)\s*```"

    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def truncate_text(
    text: str,
    max_length: int = 10000,
    suffix: str = "\n\n... (truncated)"
) -> str:
    """
    텍스트를 최대 길이로 자르기

    Args:
        text: 원본 텍스트
        max_length: 최대 문자 수
        suffix: 잘린 경우 추가할 접미사

    Returns:
        잘린 텍스트
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length - len(suffix)]
    return truncated + suffix


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    텍스트의 토큰 수 계산

    Args:
        text: 토큰 수를 계산할 텍스트
        encoding_name: tiktoken 인코딩 이름

    Returns:
        토큰 수
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except ImportError:
        # tiktoken이 없으면 대략적인 추정 (영어 기준 4자당 1토큰)
        return len(text) // 4


def truncate_to_tokens(
    text: str,
    max_tokens: int = 95000,
    encoding_name: str = "cl100k_base"
) -> str:
    """
    텍스트를 최대 토큰 수로 자르기

    Args:
        text: 원본 텍스트
        max_tokens: 최대 토큰 수
        encoding_name: tiktoken 인코딩 이름

    Returns:
        잘린 텍스트
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except ImportError:
        # tiktoken이 없으면 문자 기반으로 추정
        estimated_chars = max_tokens * 4
        return text[:estimated_chars]


def parse_xml_tag(text: str, tag: str) -> Optional[str]:
    """
    XML 태그 내용 추출

    Args:
        text: XML 태그를 포함한 텍스트
        tag: 추출할 태그 이름

    Returns:
        태그 내용 (없으면 None)
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def clean_content(text: str) -> str:
    """
    텍스트 정리 (불필요한 공백 제거 등)

    Args:
        text: 원본 텍스트

    Returns:
        정리된 텍스트
    """
    # 연속된 빈 줄을 하나로
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 줄 끝 공백 제거
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
