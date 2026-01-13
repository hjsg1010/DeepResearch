"""
PPL/엔트로피 기반 불확실성 분석

ParallelMuse 스타일의 토큰별 엔트로피 계산 및
불확실성 기반 분기점 탐지 기능을 제공합니다.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from langgraph_agent.iter_research.config import StepPPL, PartialSamplingMode

logger = logging.getLogger(__name__)


@dataclass
class TokenEntropy:
    """토큰별 엔트로피 정보"""
    token: str
    entropy: float
    logprob: float = 0.0


@dataclass
class StepUncertainty:
    """단계별 불확실성 정보"""
    step_idx: int
    think_ppl: float = -1.0
    tool_call_ppl: float = -1.0
    all_ppl: float = -1.0
    token_entropies: List[TokenEntropy] = field(default_factory=list)

    def get_ppl_by_mode(self, mode: PartialSamplingMode) -> float:
        """모드에 따른 PPL 값 반환"""
        if mode == PartialSamplingMode.ALL_PPL:
            return self.all_ppl
        elif mode == PartialSamplingMode.THINK_PPL:
            return self.think_ppl
        elif mode == PartialSamplingMode.TOOL_CALL_PPL:
            return self.tool_call_ppl
        else:
            return self.all_ppl


class EntropyCalculator:
    """
    엔트로피 계산기

    LLM 응답의 logprobs를 분석하여 토큰별 엔트로피와
    단계별 PPL(Perplexity)을 계산합니다.
    """

    def __init__(self, top_k_logprobs: int = 16):
        """
        계산기 초기화

        Args:
            top_k_logprobs: 상위 K개 logprobs 사용
        """
        self.top_k_logprobs = top_k_logprobs

    def calculate_token_entropy(
        self,
        top_logprobs: List[Dict[str, float]],
    ) -> float:
        """
        단일 토큰의 엔트로피 계산

        Args:
            top_logprobs: 상위 토큰별 logprob 딕셔너리

        Returns:
            엔트로피 값
        """
        if not top_logprobs:
            return 0.0

        # logprob 값 추출
        logprob_values = np.array(
            [lp for lp in top_logprobs[:self.top_k_logprobs]],
            dtype=np.float64
        )

        # 확률로 변환
        probs = np.exp(logprob_values)
        probs = probs / probs.sum()  # 정규화

        # 엔트로피 계산: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return float(entropy)

    def calculate_ppl_from_entropies(
        self,
        entropies: List[float],
    ) -> float:
        """
        엔트로피 리스트에서 PPL 계산

        PPL = exp(mean(entropies))

        Args:
            entropies: 엔트로피 값 리스트

        Returns:
            PPL 값
        """
        if not entropies:
            return -1.0

        mean_entropy = np.mean(entropies)
        ppl = np.exp(mean_entropy)

        return float(ppl)

    def analyze_response(
        self,
        response_text: str,
        tokens: List[str],
        top_logprobs_list: List[List[Dict[str, float]]],
    ) -> StepUncertainty:
        """
        LLM 응답 분석

        응답 텍스트와 logprobs를 분석하여 불확실성 정보를 계산합니다.

        Args:
            response_text: 응답 텍스트
            tokens: 토큰 리스트
            top_logprobs_list: 토큰별 상위 logprobs 리스트

        Returns:
            StepUncertainty: 불확실성 정보
        """
        if len(tokens) != len(top_logprobs_list):
            logger.warning("토큰 수와 logprobs 수가 일치하지 않습니다")
            return StepUncertainty(step_idx=0)

        # 토큰별 엔트로피 계산
        token_entropies = []
        for token, top_logprobs in zip(tokens, top_logprobs_list):
            logprob_values = [lp.get("logprob", 0) for lp in top_logprobs]
            entropy = self.calculate_token_entropy(logprob_values)
            token_entropies.append(TokenEntropy(
                token=token,
                entropy=entropy,
                logprob=top_logprobs[0].get("logprob", 0) if top_logprobs else 0,
            ))

        entropies = [te.entropy for te in token_entropies]

        # 전체 PPL
        all_ppl = self.calculate_ppl_from_entropies(entropies)

        # Think 영역 PPL
        think_ppl = self._calculate_region_ppl(
            tokens, entropies, "<think>", "</think>"
        )

        # Tool Call 영역 PPL
        tool_call_ppl = self._calculate_region_ppl(
            tokens, entropies, "<tool_call>", "</tool_call>"
        )

        return StepUncertainty(
            step_idx=0,
            think_ppl=think_ppl,
            tool_call_ppl=tool_call_ppl,
            all_ppl=all_ppl,
            token_entropies=token_entropies,
        )

    def _calculate_region_ppl(
        self,
        tokens: List[str],
        entropies: List[float],
        start_tag: str,
        end_tag: str,
    ) -> float:
        """
        특정 영역의 PPL 계산

        Args:
            tokens: 토큰 리스트
            entropies: 엔트로피 리스트
            start_tag: 시작 태그
            end_tag: 종료 태그

        Returns:
            해당 영역의 PPL (-1 if not found)
        """
        # 태그 위치 찾기
        text = "".join(tokens)
        start_idx = None
        end_idx = None

        # 토큰 인덱스로 변환
        current_pos = 0
        for i, token in enumerate(tokens):
            token_start = current_pos
            token_end = current_pos + len(token)

            if start_idx is None and start_tag in text[token_start:token_end + len(start_tag)]:
                start_idx = i

            if end_idx is None and end_tag in text[token_start:token_end + len(end_tag)]:
                end_idx = i

            current_pos = token_end

        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return -1.0

        # 해당 영역의 엔트로피로 PPL 계산
        region_entropies = entropies[start_idx + 1:end_idx]

        if not region_entropies:
            return -1.0

        return self.calculate_ppl_from_entropies(region_entropies)


class UncertaintyDetector:
    """
    불확실성 기반 분기점 탐지기

    ParallelMuse 스타일의 부분 롤아웃을 위해
    불확실성이 높은 지점을 탐지합니다.
    """

    def __init__(
        self,
        mode: PartialSamplingMode = PartialSamplingMode.TOOL_CALL_PPL,
        top_k: int = 2,
    ):
        """
        탐지기 초기화

        Args:
            mode: 불확실성 측정 모드
            top_k: 상위 K개 분기점 선택
        """
        self.mode = mode
        self.top_k = top_k
        self.calculator = EntropyCalculator()

    def detect_branch_points(
        self,
        rollout_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        불확실성이 높은 분기점 탐지

        Args:
            rollout_messages: 롤아웃 메시지 리스트

        Returns:
            분기점 정보 리스트 [{"step_idx": int, "ppl": float}, ...]
        """
        if self.mode == PartialSamplingMode.NONE:
            return []

        branch_points = []

        for i, msg in enumerate(rollout_messages):
            # Assistant 메시지만 분석
            if msg.get("role") != "assistant":
                continue

            # step_ppl 정보가 있으면 사용
            step_ppl = msg.get("step_ppl")
            if step_ppl:
                ppl_value = self._get_ppl_by_mode(step_ppl)
                if ppl_value > 0:
                    branch_points.append({
                        "step_idx": i,
                        "ppl": ppl_value,
                    })

        # PPL이 높은 순으로 정렬하여 상위 K개 선택
        branch_points.sort(key=lambda x: x["ppl"], reverse=True)

        return branch_points[:self.top_k]

    def detect_branch_points_mixed(
        self,
        rollout_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Mixed PPL 모드로 분기점 탐지

        Think PPL과 Tool Call PPL을 혼합하여 분기점 선택

        Args:
            rollout_messages: 롤아웃 메시지 리스트

        Returns:
            분기점 정보 리스트
        """
        tool_call_top_k = math.ceil(self.top_k / 2)
        think_top_k = math.floor(self.top_k / 2)

        tool_call_points = []
        think_points = []

        for i, msg in enumerate(rollout_messages):
            if msg.get("role") != "assistant":
                continue

            step_ppl = msg.get("step_ppl")
            if not step_ppl:
                continue

            # Tool Call PPL
            tc_ppl = step_ppl.get("tool_call_ppl", -1)
            if tc_ppl > 0:
                tool_call_points.append({"step_idx": i, "ppl": tc_ppl, "type": "tool_call"})

            # Think PPL
            think_ppl = step_ppl.get("think_ppl", -1)
            if think_ppl > 0:
                think_points.append({"step_idx": i, "ppl": think_ppl, "type": "think"})

        # 각각 정렬하여 상위 K개 선택
        tool_call_points.sort(key=lambda x: x["ppl"], reverse=True)
        think_points.sort(key=lambda x: x["ppl"], reverse=True)

        branch_points = (
            tool_call_points[:tool_call_top_k] +
            think_points[:think_top_k]
        )

        return branch_points

    def _get_ppl_by_mode(self, step_ppl: Dict[str, float]) -> float:
        """모드에 따른 PPL 값 반환"""
        if self.mode == PartialSamplingMode.ALL_PPL:
            return step_ppl.get("all_ppl", -1)
        elif self.mode == PartialSamplingMode.THINK_PPL:
            return step_ppl.get("think_ppl", -1)
        elif self.mode == PartialSamplingMode.TOOL_CALL_PPL:
            return step_ppl.get("tool_call_ppl", -1)
        else:
            return step_ppl.get("all_ppl", -1)


def compute_step_ppl_from_logprobs(
    response_logprobs: List[Dict[str, Any]],
) -> StepPPL:
    """
    logprobs에서 StepPPL 계산 (편의 함수)

    Args:
        response_logprobs: OpenAI API 형식의 logprobs

    Returns:
        StepPPL: 단계별 PPL 정보
    """
    if not response_logprobs:
        return StepPPL()

    calculator = EntropyCalculator()

    # 토큰과 top_logprobs 추출
    tokens = [item.get("token", "") for item in response_logprobs]
    top_logprobs_list = [
        [{"logprob": tlp.get("logprob", 0)} for tlp in item.get("top_logprobs", [])]
        for item in response_logprobs
    ]

    response_text = "".join(tokens)
    uncertainty = calculator.analyze_response(response_text, tokens, top_logprobs_list)

    return StepPPL(
        think_ppl=uncertainty.think_ppl,
        tool_call_ppl=uncertainty.tool_call_ppl,
        all_ppl=uncertainty.all_ppl,
    )
