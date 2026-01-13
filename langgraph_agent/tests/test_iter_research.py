#!/usr/bin/env python3
"""
IterResearch 모듈 테스트
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph_agent.iter_research.config import (
    IterResearchConfig,
    PartialSamplingMode,
    AggregationStrategy,
    RolloutResult,
    AggregatedResult,
    StepPPL,
)
from langgraph_agent.iter_research.aggregator import (
    ResultAggregator,
    MetricsCalculator,
    EvaluationMetrics,
)
from langgraph_agent.iter_research.entropy import (
    EntropyCalculator,
    UncertaintyDetector,
)


class TestIterResearchConfig:
    """IterResearch 설정 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = IterResearchConfig()

        assert config.rollout_count == 3
        assert config.max_workers == 20
        assert config.partial_sampling_mode == PartialSamplingMode.NONE

    def test_config_validation_pass(self):
        """설정 검증 성공 테스트"""
        config = IterResearchConfig(
            rollout_count=3,
            partial_sampling_mode=PartialSamplingMode.NONE,
        )

        # 예외 없이 통과해야 함
        config.validate()

    def test_config_validation_fail(self):
        """설정 검증 실패 테스트"""
        config = IterResearchConfig(
            rollout_count=0,  # 잘못된 값
        )

        with pytest.raises(ValueError):
            config.validate()

    def test_partial_sampling_budget_validation(self):
        """부분 샘플링 예산 검증"""
        config = IterResearchConfig(
            partial_sampling_mode=PartialSamplingMode.TOOL_CALL_PPL,
            initial_rollout_num=1,
            partial_sampling_topk=2,
            partial_sampling_rounds=1,
            partial_sampling_times_per_pos=3,
            sampling_budget=5,  # 너무 작은 예산
        )

        with pytest.raises(ValueError):
            config.validate()

    def test_output_files_generation(self):
        """출력 파일 경로 생성 테스트"""
        config = IterResearchConfig(rollout_count=3)
        files = config.get_output_files("test")

        assert len(files) == 3
        assert 1 in files
        assert 2 in files
        assert 3 in files
        assert "iter1" in files[1]
        assert "iter2" in files[2]
        assert "iter3" in files[3]


class TestRolloutResult:
    """롤아웃 결과 테스트"""

    def test_successful_result(self):
        """성공적인 결과 테스트"""
        result = RolloutResult(
            question="테스트 질문",
            answer="정답",
            prediction="예측 답변",
            messages=[{"role": "assistant", "content": "답변"}],
            termination="answer",
            rollout_idx=1,
        )

        assert result.is_successful() is True

    def test_failed_result(self):
        """실패한 결과 테스트"""
        result = RolloutResult(
            question="테스트 질문",
            answer="정답",
            prediction="",
            messages=[],
            termination="error",
            rollout_idx=1,
        )

        assert result.is_successful() is False

    def test_to_dict_from_dict(self):
        """직렬화/역직렬화 테스트"""
        original = RolloutResult(
            question="테스트",
            answer="정답",
            prediction="예측",
            messages=[{"role": "user", "content": "테스트"}],
            termination="answer",
            rollout_idx=1,
            execution_time=10.5,
            total_tokens=100,
        )

        # 딕셔너리로 변환
        data = original.to_dict()

        # 다시 객체로 변환
        restored = RolloutResult.from_dict(data)

        assert restored.question == original.question
        assert restored.prediction == original.prediction
        assert restored.rollout_idx == original.rollout_idx


class TestResultAggregator:
    """결과 집계기 테스트"""

    def setup_method(self):
        """테스트 데이터 설정"""
        self.results = [
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="예측1",
                messages=[],
                termination="answer",
                rollout_idx=1,
            ),
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="예측2",
                messages=[],
                termination="answer",
                rollout_idx=2,
            ),
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="",
                messages=[],
                termination="error",
                rollout_idx=3,
            ),
        ]

    def test_first_success_strategy(self):
        """첫 번째 성공 전략 테스트"""
        aggregator = ResultAggregator(strategy=AggregationStrategy.FIRST_SUCCESS)
        result = aggregator.aggregate(self.results)

        assert result.final_prediction == "예측1"
        assert result.question == "질문"

    def test_majority_vote_strategy(self):
        """다수결 전략 테스트"""
        # 같은 예측이 2개인 경우
        results = [
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="답변A",
                messages=[],
                termination="answer",
                rollout_idx=1,
            ),
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="답변A",
                messages=[],
                termination="answer",
                rollout_idx=2,
            ),
            RolloutResult(
                question="질문",
                answer="정답",
                prediction="답변B",
                messages=[],
                termination="answer",
                rollout_idx=3,
            ),
        ]

        aggregator = ResultAggregator(strategy=AggregationStrategy.MAJORITY_VOTE)
        result = aggregator.aggregate(results)

        assert result.final_prediction == "답변A"


class TestMetricsCalculator:
    """메트릭 계산기 테스트"""

    def setup_method(self):
        """테스트 데이터 설정"""
        self.results_by_question = {
            "질문1": [
                RolloutResult(
                    question="질문1",
                    answer="정답1",
                    prediction="예측",
                    messages=[],
                    termination="answer",
                    rollout_idx=1,
                ),
                RolloutResult(
                    question="질문1",
                    answer="정답1",
                    prediction="예측",
                    messages=[],
                    termination="answer",
                    rollout_idx=2,
                ),
                RolloutResult(
                    question="질문1",
                    answer="정답1",
                    prediction="",
                    messages=[],
                    termination="error",
                    rollout_idx=3,
                ),
            ],
            "질문2": [
                RolloutResult(
                    question="질문2",
                    answer="정답2",
                    prediction="",
                    messages=[],
                    termination="error",
                    rollout_idx=1,
                ),
                RolloutResult(
                    question="질문2",
                    answer="정답2",
                    prediction="",
                    messages=[],
                    termination="error",
                    rollout_idx=2,
                ),
                RolloutResult(
                    question="질문2",
                    answer="정답2",
                    prediction="",
                    messages=[],
                    termination="error",
                    rollout_idx=3,
                ),
            ],
        }

    def test_calculate_metrics(self):
        """메트릭 계산 테스트"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(self.results_by_question)

        assert metrics.total_questions == 2
        assert metrics.total_rollouts == 6

        # 질문1은 성공, 질문2는 실패
        # Pass@3: 1/2 = 50%
        assert metrics.pass_at_3 == 0.5

    def test_pass_at_1(self):
        """Pass@1 계산 테스트"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(self.results_by_question)

        # 라운드1에서 질문1만 성공 = 50%
        assert metrics.pass_at_1 == 0.5

    def test_round_pass_rates(self):
        """라운드별 성공률 테스트"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(self.results_by_question)

        # 라운드1: 1/2 성공, 라운드2: 1/2 성공, 라운드3: 0/2 성공
        assert 1 in metrics.round_pass_rates
        assert 2 in metrics.round_pass_rates
        assert 3 in metrics.round_pass_rates


class TestEntropyCalculator:
    """엔트로피 계산기 테스트"""

    def test_calculate_entropy(self):
        """엔트로피 계산 테스트"""
        calculator = EntropyCalculator()

        # 균등 분포 (높은 엔트로피)
        uniform_logprobs = [-1.0, -1.0, -1.0, -1.0]
        entropy_uniform = calculator.calculate_token_entropy(uniform_logprobs)

        # 집중 분포 (낮은 엔트로피)
        concentrated_logprobs = [-0.1, -5.0, -5.0, -5.0]
        entropy_concentrated = calculator.calculate_token_entropy(concentrated_logprobs)

        # 균등 분포가 더 높은 엔트로피를 가져야 함
        assert entropy_uniform > entropy_concentrated

    def test_calculate_ppl(self):
        """PPL 계산 테스트"""
        calculator = EntropyCalculator()

        entropies = [1.0, 1.5, 2.0]
        ppl = calculator.calculate_ppl_from_entropies(entropies)

        # PPL = exp(mean(entropies)) = exp(1.5)
        import math
        expected_ppl = math.exp(1.5)

        assert abs(ppl - expected_ppl) < 0.01


class TestUncertaintyDetector:
    """불확실성 탐지기 테스트"""

    def test_detect_no_branch_points(self):
        """분기점 없음 테스트"""
        detector = UncertaintyDetector(
            mode=PartialSamplingMode.NONE,
            top_k=2,
        )

        messages = [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
        ]

        branch_points = detector.detect_branch_points(messages)

        assert len(branch_points) == 0

    def test_detect_branch_points_with_ppl(self):
        """PPL 기반 분기점 탐지 테스트"""
        detector = UncertaintyDetector(
            mode=PartialSamplingMode.TOOL_CALL_PPL,
            top_k=2,
        )

        messages = [
            {"role": "user", "content": "질문"},
            {
                "role": "assistant",
                "content": "<think>생각</think><tool_call>도구</tool_call>",
                "step_ppl": {"think_ppl": 1.5, "tool_call_ppl": 2.5, "all_ppl": 2.0},
            },
            {"role": "user", "content": "<tool_response>결과</tool_response>"},
            {
                "role": "assistant",
                "content": "<think>생각2</think><tool_call>도구2</tool_call>",
                "step_ppl": {"think_ppl": 1.2, "tool_call_ppl": 3.0, "all_ppl": 2.1},
            },
        ]

        branch_points = detector.detect_branch_points(messages)

        # PPL이 높은 순으로 정렬되어 있어야 함
        assert len(branch_points) <= 2
        if len(branch_points) >= 2:
            assert branch_points[0]["ppl"] >= branch_points[1]["ppl"]


class TestStepPPL:
    """StepPPL 테스트"""

    def test_to_dict_from_dict(self):
        """직렬화/역직렬화 테스트"""
        original = StepPPL(think_ppl=1.5, tool_call_ppl=2.0, all_ppl=1.8)

        data = original.to_dict()
        restored = StepPPL.from_dict(data)

        assert restored.think_ppl == original.think_ppl
        assert restored.tool_call_ppl == original.tool_call_ppl
        assert restored.all_ppl == original.all_ppl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
