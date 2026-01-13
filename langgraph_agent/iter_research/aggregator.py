"""
결과 집계 및 평가 모듈

다중 롤아웃 결과를 집계하고 평가 메트릭을 계산합니다.
Pass@K, Best@1, Avg@K 등의 메트릭을 제공합니다.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langgraph_agent.iter_research.config import (
    AggregatedResult,
    AggregationStrategy,
    RolloutResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """평가 메트릭 결과"""
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    best_at_1: float = 0.0
    avg_at_3: float = 0.0

    # 라운드별 성공률
    round_pass_rates: Dict[int, float] = field(default_factory=dict)

    # 통계
    total_questions: int = 0
    total_rollouts: int = 0
    successful_rollouts: int = 0

    # 상세 정보
    avg_execution_time: float = 0.0
    avg_tokens_per_rollout: float = 0.0
    termination_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "pass_at_1": round(self.pass_at_1 * 100, 2),
            "pass_at_3": round(self.pass_at_3 * 100, 2),
            "best_at_1": round(self.best_at_1 * 100, 2),
            "avg_at_3": round(self.avg_at_3 * 100, 2),
            "round_pass_rates": {
                k: round(v * 100, 2) for k, v in self.round_pass_rates.items()
            },
            "total_questions": self.total_questions,
            "total_rollouts": self.total_rollouts,
            "successful_rollouts": self.successful_rollouts,
            "avg_execution_time": round(self.avg_execution_time, 2),
            "avg_tokens_per_rollout": round(self.avg_tokens_per_rollout, 2),
            "termination_distribution": self.termination_distribution,
        }

    def __str__(self) -> str:
        """문자열 표현"""
        return (
            f"=== 평가 결과 ===\n"
            f"Pass@1: {self.pass_at_1 * 100:.2f}%\n"
            f"Pass@3: {self.pass_at_3 * 100:.2f}%\n"
            f"Best@1: {self.best_at_1 * 100:.2f}%\n"
            f"Avg@3: {self.avg_at_3 * 100:.2f}%\n"
            f"총 질문 수: {self.total_questions}\n"
            f"총 롤아웃 수: {self.total_rollouts}\n"
            f"성공 롤아웃 수: {self.successful_rollouts}\n"
            f"평균 실행 시간: {self.avg_execution_time:.2f}s\n"
        )


class ResultAggregator:
    """
    결과 집계기

    다중 롤아웃 결과를 집계하고 최종 예측을 결정합니다.
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FIRST_SUCCESS,
        judge_func: Optional[Callable[[str, str, str], bool]] = None,
    ):
        """
        집계기 초기화

        Args:
            strategy: 집계 전략
            judge_func: 커스텀 판정 함수 (question, answer, prediction) -> bool
        """
        self.strategy = strategy
        self.judge_func = judge_func

    def aggregate(
        self,
        results: List[RolloutResult],
    ) -> AggregatedResult:
        """
        롤아웃 결과 집계

        Args:
            results: 롤아웃 결과 리스트

        Returns:
            AggregatedResult: 집계된 결과
        """
        if not results:
            raise ValueError("결과 리스트가 비어있습니다")

        question = results[0].question
        answer = results[0].answer

        # 전략에 따른 최종 예측 결정
        final_prediction = self._select_final_prediction(results)

        return AggregatedResult(
            question=question,
            answer=answer,
            rollout_results=results,
            final_prediction=final_prediction,
            aggregation_strategy=self.strategy,
            metrics=self._compute_metrics(results),
        )

    def _select_final_prediction(
        self,
        results: List[RolloutResult],
    ) -> str:
        """
        최종 예측 선택

        Args:
            results: 롤아웃 결과 리스트

        Returns:
            최종 예측 문자열
        """
        if self.strategy == AggregationStrategy.FIRST_SUCCESS:
            # 첫 번째 성공한 결과 선택
            for result in results:
                if result.is_successful():
                    return result.prediction
            # 성공한 결과가 없으면 마지막 결과
            return results[-1].prediction if results else ""

        elif self.strategy == AggregationStrategy.MAJORITY_VOTE:
            # 다수결 (같은 답변이 가장 많은 것)
            vote_counts = defaultdict(int)
            for result in results:
                if result.prediction.strip():
                    vote_counts[result.prediction.strip()] += 1

            if vote_counts:
                return max(vote_counts.items(), key=lambda x: x[1])[0]
            return results[-1].prediction if results else ""

        elif self.strategy == AggregationStrategy.BEST_CONFIDENCE:
            # 메타데이터에 confidence가 있으면 사용
            best_result = None
            best_confidence = -1

            for result in results:
                confidence = result.metadata.get("confidence", 0)
                if confidence > best_confidence and result.is_successful():
                    best_confidence = confidence
                    best_result = result

            if best_result:
                return best_result.prediction

            # confidence가 없으면 첫 번째 성공
            for result in results:
                if result.is_successful():
                    return result.prediction
            return results[-1].prediction if results else ""

        else:
            # 기본: 첫 번째 성공
            for result in results:
                if result.is_successful():
                    return result.prediction
            return results[-1].prediction if results else ""

    def _compute_metrics(
        self,
        results: List[RolloutResult],
    ) -> dict:
        """롤아웃 결과에 대한 메트릭 계산"""
        successful = [r for r in results if r.is_successful()]

        return {
            "total_rollouts": len(results),
            "successful_rollouts": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_execution_time": (
                sum(r.execution_time for r in results) / len(results)
                if results else 0
            ),
        }


class MetricsCalculator:
    """
    평가 메트릭 계산기

    Pass@K, Best@1, Avg@K 등의 메트릭을 계산합니다.
    """

    def __init__(
        self,
        judge_func: Optional[Callable[[str, str, str], bool]] = None,
    ):
        """
        계산기 초기화

        Args:
            judge_func: 정답 판정 함수 (question, ground_truth, prediction) -> bool
                       None이면 is_successful() 사용
        """
        self.judge_func = judge_func

    def calculate(
        self,
        results_by_question: Dict[str, List[RolloutResult]],
    ) -> EvaluationMetrics:
        """
        전체 메트릭 계산

        Args:
            results_by_question: 질문별 롤아웃 결과

        Returns:
            EvaluationMetrics: 평가 메트릭
        """
        metrics = EvaluationMetrics()

        if not results_by_question:
            return metrics

        metrics.total_questions = len(results_by_question)

        # 질문별 결과 분석
        all_results = []
        question_correct = {}  # 질문별 라운드별 정답 여부

        for question, results in results_by_question.items():
            question_correct[question] = {}

            for result in results:
                all_results.append(result)
                is_correct = self._is_correct(result)
                question_correct[question][result.rollout_idx] = is_correct

        metrics.total_rollouts = len(all_results)
        metrics.successful_rollouts = sum(1 for r in all_results if r.is_successful())

        # Pass@K 계산
        metrics.pass_at_1 = self._calculate_pass_at_k(question_correct, k=1)
        metrics.pass_at_3 = self._calculate_pass_at_k(question_correct, k=3)

        # Best@1 계산
        metrics.best_at_1 = self._calculate_best_at_1(question_correct)

        # Avg@3 계산
        metrics.avg_at_3 = self._calculate_avg_at_k(question_correct, k=3)

        # 라운드별 성공률
        metrics.round_pass_rates = self._calculate_round_pass_rates(question_correct)

        # 통계
        if all_results:
            metrics.avg_execution_time = (
                sum(r.execution_time for r in all_results) / len(all_results)
            )
            metrics.avg_tokens_per_rollout = (
                sum(r.total_tokens for r in all_results) / len(all_results)
            )

        # 종료 사유 분포
        for result in all_results:
            termination = result.termination
            metrics.termination_distribution[termination] = (
                metrics.termination_distribution.get(termination, 0) + 1
            )

        return metrics

    def _is_correct(self, result: RolloutResult) -> bool:
        """결과가 정답인지 판정"""
        if self.judge_func:
            return self.judge_func(
                result.question,
                result.answer,
                result.prediction,
            )
        return result.is_successful()

    def _calculate_pass_at_k(
        self,
        question_correct: Dict[str, Dict[int, bool]],
        k: int,
    ) -> float:
        """
        Pass@K 계산

        K번의 시도 중 최소 1번 성공한 질문의 비율

        Args:
            question_correct: 질문별 라운드별 정답 여부
            k: 시도 횟수

        Returns:
            Pass@K 비율 (0.0 ~ 1.0)
        """
        if not question_correct:
            return 0.0

        total_pass = 0

        for question, round_results in question_correct.items():
            # 상위 K개 라운드만 확인
            sorted_rounds = sorted(round_results.items(), key=lambda x: x[0])[:k]

            if any(is_correct for _, is_correct in sorted_rounds):
                total_pass += 1

        return total_pass / len(question_correct)

    def _calculate_best_at_1(
        self,
        question_correct: Dict[str, Dict[int, bool]],
    ) -> float:
        """
        Best@1 계산

        각 라운드 중 가장 높은 성공률

        Args:
            question_correct: 질문별 라운드별 정답 여부

        Returns:
            Best@1 비율 (0.0 ~ 1.0)
        """
        if not question_correct:
            return 0.0

        # 라운드별 정답 수 계산
        round_correct_counts = defaultdict(int)
        round_total_counts = defaultdict(int)

        for question, round_results in question_correct.items():
            for round_idx, is_correct in round_results.items():
                round_total_counts[round_idx] += 1
                if is_correct:
                    round_correct_counts[round_idx] += 1

        # 각 라운드의 성공률 계산
        round_pass_rates = {
            round_idx: round_correct_counts[round_idx] / round_total_counts[round_idx]
            for round_idx in round_total_counts
        }

        return max(round_pass_rates.values()) if round_pass_rates else 0.0

    def _calculate_avg_at_k(
        self,
        question_correct: Dict[str, Dict[int, bool]],
        k: int,
    ) -> float:
        """
        Avg@K 계산

        K개 라운드의 평균 성공률

        Args:
            question_correct: 질문별 라운드별 정답 여부
            k: 라운드 수

        Returns:
            Avg@K 비율 (0.0 ~ 1.0)
        """
        if not question_correct:
            return 0.0

        round_pass_rates = self._calculate_round_pass_rates(question_correct)

        # 상위 K개 라운드만 사용
        sorted_rates = sorted(round_pass_rates.items(), key=lambda x: x[0])[:k]

        if not sorted_rates:
            return 0.0

        return sum(rate for _, rate in sorted_rates) / len(sorted_rates)

    def _calculate_round_pass_rates(
        self,
        question_correct: Dict[str, Dict[int, bool]],
    ) -> Dict[int, float]:
        """
        라운드별 성공률 계산

        Args:
            question_correct: 질문별 라운드별 정답 여부

        Returns:
            라운드별 성공률 딕셔너리
        """
        round_correct_counts = defaultdict(int)
        round_total_counts = defaultdict(int)

        for question, round_results in question_correct.items():
            for round_idx, is_correct in round_results.items():
                round_total_counts[round_idx] += 1
                if is_correct:
                    round_correct_counts[round_idx] += 1

        return {
            round_idx: round_correct_counts[round_idx] / round_total_counts[round_idx]
            for round_idx in round_total_counts
        }


def load_results_from_files(
    file_paths: Dict[int, str],
) -> Dict[str, List[RolloutResult]]:
    """
    파일에서 결과 로드

    Args:
        file_paths: 롤아웃 인덱스별 파일 경로

    Returns:
        질문별 롤아웃 결과 딕셔너리
    """
    results_by_question = defaultdict(list)

    for rollout_idx, filepath in file_paths.items():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    result = RolloutResult.from_dict(data)
                    result.rollout_idx = rollout_idx
                    results_by_question[result.question].append(result)
        except FileNotFoundError:
            logger.warning(f"파일을 찾을 수 없습니다: {filepath}")
        except Exception as e:
            logger.error(f"파일 로드 오류: {filepath}, {e}")

    return dict(results_by_question)


def evaluate_iter_research_results(
    file_paths: Dict[int, str],
    judge_func: Optional[Callable[[str, str, str], bool]] = None,
) -> EvaluationMetrics:
    """
    IterResearch 결과 평가 (편의 함수)

    Args:
        file_paths: 롤아웃 인덱스별 결과 파일 경로
        judge_func: 정답 판정 함수

    Returns:
        EvaluationMetrics: 평가 메트릭
    """
    results_by_question = load_results_from_files(file_paths)
    calculator = MetricsCalculator(judge_func)

    return calculator.calculate(results_by_question)
