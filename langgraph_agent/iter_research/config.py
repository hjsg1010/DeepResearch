"""
IterResearch 설정 모듈

IterResearch (Heavy) 모드의 설정을 관리합니다.
Test-Time Scaling 전략을 위한 다양한 파라미터를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class PartialSamplingMode(str, Enum):
    """부분 샘플링 모드"""
    NONE = "none"               # 부분 샘플링 비활성화 (전체 롤아웃만)
    ALL_PPL = "all_ppl"         # 전체 PPL 기반
    THINK_PPL = "think_ppl"     # Think 영역 PPL 기반
    TOOL_CALL_PPL = "tool_call_ppl"  # Tool Call 영역 PPL 기반
    MIXED_PPL = "mixed_ppl"     # Think + Tool Call PPL 혼합


class AggregationStrategy(str, Enum):
    """결과 집계 전략"""
    MAJORITY_VOTE = "majority_vote"   # 다수결
    BEST_CONFIDENCE = "best_confidence"  # 가장 높은 신뢰도
    FIRST_SUCCESS = "first_success"   # 첫 번째 성공
    LLM_JUDGE = "llm_judge"           # LLM을 사용한 판정


@dataclass
class IterResearchConfig:
    """
    IterResearch 설정

    Test-Time Scaling을 통해 모델의 최대 성능을 이끌어내기 위한 설정입니다.
    """

    # === 기본 롤아웃 설정 ===
    rollout_count: int = 3
    """각 질문당 실행할 롤아웃 수 (기본: 3)"""

    max_workers: int = 20
    """병렬 처리 워커 수"""

    # === 단일 롤아웃 제한 ===
    max_turns_per_rollout: int = 100
    """단일 롤아웃의 최대 턴 수"""

    max_context_length: int = 128 * 1024
    """최대 컨텍스트 길이 (토큰)"""

    max_execution_time: int = 150 * 60
    """최대 실행 시간 (초) - 기본 150분"""

    # === 부분 샘플링 설정 (ParallelMuse 스타일) ===
    partial_sampling_mode: PartialSamplingMode = PartialSamplingMode.NONE
    """부분 샘플링 모드"""

    initial_rollout_num: int = 1
    """초기 롤아웃 수 (부분 샘플링 시 기준)"""

    partial_sampling_topk: int = 2
    """부분 샘플링 시 상위 K개 불확실성 지점 선택"""

    partial_sampling_rounds: int = 1
    """부분 샘플링 라운드 수"""

    partial_sampling_times_per_pos: int = 3
    """각 분기 지점당 샘플링 횟수"""

    sampling_budget: int = 8
    """총 샘플링 예산 (초기 롤아웃 + 부분 샘플링)"""

    # === 결과 집계 설정 ===
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FIRST_SUCCESS
    """결과 집계 전략"""

    judge_model: Optional[str] = None
    """LLM Judge 사용 시 모델명"""

    # === 출력 설정 ===
    output_dir: str = "./iter_research_outputs"
    """출력 디렉토리"""

    save_intermediate: bool = True
    """중간 결과 저장 여부"""

    verbose: bool = True
    """상세 로깅 여부"""

    def validate(self) -> None:
        """설정 유효성 검사"""
        if self.rollout_count < 1:
            raise ValueError("rollout_count는 1 이상이어야 합니다")

        if self.partial_sampling_mode != PartialSamplingMode.NONE:
            expected_budget = (
                self.initial_rollout_num +
                self.initial_rollout_num *
                self.partial_sampling_topk *
                self.partial_sampling_rounds *
                self.partial_sampling_times_per_pos
            )
            if self.sampling_budget < expected_budget:
                raise ValueError(
                    f"sampling_budget({self.sampling_budget})는 최소 {expected_budget}이어야 합니다 "
                    f"(initial_rollout_num * (1 + partial_sampling_topk * rounds * times_per_pos))"
                )

    def get_output_files(self, base_name: str) -> dict:
        """
        롤아웃별 출력 파일 경로 생성

        Args:
            base_name: 기본 파일명

        Returns:
            롤아웃 인덱스별 파일 경로 딕셔너리
        """
        import os
        return {
            i: os.path.join(self.output_dir, f"{base_name}_iter{i}.jsonl")
            for i in range(1, self.rollout_count + 1)
        }


@dataclass
class RolloutResult:
    """단일 롤아웃 결과"""
    question: str
    answer: str  # Ground truth (있는 경우)
    prediction: str  # 모델 예측
    messages: List[dict]
    termination: str  # 종료 사유
    rollout_idx: int
    execution_time: float = 0.0
    total_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def is_successful(self) -> bool:
        """성공적으로 답변을 생성했는지 확인"""
        return self.termination == "answer" and bool(self.prediction.strip())

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "question": self.question,
            "answer": self.answer,
            "prediction": self.prediction,
            "messages": self.messages,
            "termination": self.termination,
            "rollout_idx": self.rollout_idx,
            "execution_time": self.execution_time,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutResult":
        """딕셔너리에서 생성"""
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            prediction=data.get("prediction", ""),
            messages=data.get("messages", []),
            termination=data.get("termination", "unknown"),
            rollout_idx=data.get("rollout_idx", 0),
            execution_time=data.get("execution_time", 0.0),
            total_tokens=data.get("total_tokens", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AggregatedResult:
    """집계된 결과"""
    question: str
    answer: str  # Ground truth
    rollout_results: List[RolloutResult]
    final_prediction: str
    aggregation_strategy: AggregationStrategy
    metrics: dict = field(default_factory=dict)

    @property
    def pass_at_k(self) -> dict:
        """Pass@K 메트릭 계산"""
        successful = [r for r in self.rollout_results if r.is_successful()]
        total = len(self.rollout_results)

        return {
            f"pass@{k}": 1 if len(successful) >= 1 else 0
            for k in range(1, total + 1)
        }

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if not self.rollout_results:
            return 0.0
        successful = sum(1 for r in self.rollout_results if r.is_successful())
        return successful / len(self.rollout_results)

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "question": self.question,
            "answer": self.answer,
            "rollout_results": [r.to_dict() for r in self.rollout_results],
            "final_prediction": self.final_prediction,
            "aggregation_strategy": self.aggregation_strategy.value,
            "metrics": self.metrics,
            "pass_at_k": self.pass_at_k,
            "success_rate": self.success_rate,
        }


@dataclass
class StepPPL:
    """단계별 PPL (Perplexity) 정보"""
    think_ppl: float = -1.0
    tool_call_ppl: float = -1.0
    all_ppl: float = -1.0

    def to_dict(self) -> dict:
        return {
            "think_ppl": self.think_ppl,
            "tool_call_ppl": self.tool_call_ppl,
            "all_ppl": self.all_ppl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepPPL":
        return cls(
            think_ppl=data.get("think_ppl", -1.0),
            tool_call_ppl=data.get("tool_call_ppl", -1.0),
            all_ppl=data.get("all_ppl", -1.0),
        )
