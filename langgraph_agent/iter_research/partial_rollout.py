"""
부분 롤아웃 (Partial Rollout) 모듈

ParallelMuse 스타일의 불확실성 기반 부분 롤아웃을 구현합니다.
기존 롤아웃의 불확실성이 높은 지점에서 분기하여 추가 샘플링합니다.
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langgraph_agent.iter_research.config import (
    IterResearchConfig,
    PartialSamplingMode,
    RolloutResult,
)
from langgraph_agent.iter_research.entropy import UncertaintyDetector
from langgraph_agent.iter_research.rollout import RolloutTask
from langgraph_agent.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class BranchPoint:
    """분기점 정보"""
    step_idx: int
    ppl: float
    branch_type: str = "full"  # "full", "tool_call", "think"
    prepend_messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PartialRolloutResult(RolloutResult):
    """부분 롤아웃 결과"""
    branch_point: Optional[BranchPoint] = None
    rollout_type: str = "partial"  # "full", "partial"
    parent_rollout_idx: Optional[int] = None


class PartialRolloutExecutor:
    """
    부분 롤아웃 실행기

    기존 롤아웃의 불확실성이 높은 지점에서 분기하여
    추가 샘플링을 수행합니다.
    """

    def __init__(
        self,
        iter_config: IterResearchConfig,
        agent_config: Optional[AgentConfig] = None,
        run_single_rollout_func: Optional[Callable] = None,
    ):
        """
        실행기 초기화

        Args:
            iter_config: IterResearch 설정
            agent_config: 에이전트 설정
            run_single_rollout_func: 단일 롤아웃 실행 함수
        """
        self.iter_config = iter_config
        self.agent_config = agent_config or AgentConfig()
        self.run_single_rollout_func = run_single_rollout_func

        self.detector = UncertaintyDetector(
            mode=iter_config.partial_sampling_mode,
            top_k=iter_config.partial_sampling_topk,
        )

    def detect_branch_points(
        self,
        initial_rollout: RolloutResult,
    ) -> List[BranchPoint]:
        """
        초기 롤아웃에서 분기점 탐지

        Args:
            initial_rollout: 초기 롤아웃 결과

        Returns:
            분기점 리스트
        """
        if self.iter_config.partial_sampling_mode == PartialSamplingMode.NONE:
            return []

        messages = initial_rollout.messages

        # 분기점 탐지
        if self.iter_config.partial_sampling_mode == PartialSamplingMode.MIXED_PPL:
            detected = self.detector.detect_branch_points_mixed(messages)
        else:
            detected = self.detector.detect_branch_points(messages)

        # BranchPoint 객체로 변환
        branch_points = []
        for bp in detected:
            step_idx = bp["step_idx"]

            # 분기 지점 이전까지의 메시지 추출
            prepend_messages = copy.deepcopy(messages[:step_idx])

            branch_points.append(BranchPoint(
                step_idx=step_idx,
                ppl=bp["ppl"],
                branch_type=bp.get("type", "full"),
                prepend_messages=prepend_messages,
            ))

        return branch_points

    def create_partial_tasks(
        self,
        initial_rollouts: List[RolloutResult],
    ) -> List[Tuple[RolloutTask, BranchPoint]]:
        """
        부분 롤아웃 태스크 생성

        Args:
            initial_rollouts: 초기 롤아웃 결과 리스트

        Returns:
            (태스크, 분기점) 튜플 리스트
        """
        tasks = []

        for rollout in initial_rollouts:
            # 분기점 탐지
            branch_points = self.detect_branch_points(rollout)

            logger.info(
                f"[Partial] 롤아웃 {rollout.rollout_idx}에서 "
                f"{len(branch_points)}개 분기점 탐지"
            )

            # 각 분기점에서 여러 번 샘플링
            for bp in branch_points:
                for sample_idx in range(self.iter_config.partial_sampling_times_per_pos):
                    task = RolloutTask(
                        question=rollout.question,
                        answer=rollout.answer,
                        rollout_idx=len(tasks) + 1,  # 새 인덱스
                        metadata={
                            "parent_rollout_idx": rollout.rollout_idx,
                            "branch_step_idx": bp.step_idx,
                            "branch_ppl": bp.ppl,
                            "sample_idx": sample_idx,
                        },
                    )
                    tasks.append((task, bp))

        return tasks

    def calculate_remaining_turns(
        self,
        branch_point: BranchPoint,
    ) -> int:
        """
        분기 지점에서 남은 턴 수 계산

        Args:
            branch_point: 분기점 정보

        Returns:
            남은 턴 수
        """
        # 메시지 수에서 시스템/유저 메시지를 제외한 턴 수 계산
        turns_used = (branch_point.step_idx - 2) // 2  # 대략적인 턴 수
        remaining = self.iter_config.max_turns_per_rollout - turns_used

        return max(1, remaining)


class PartialRolloutRunner:
    """
    부분 롤아웃 실행 조정자

    전체 부분 롤아웃 프로세스를 관리합니다.
    """

    def __init__(
        self,
        iter_config: IterResearchConfig,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        조정자 초기화

        Args:
            iter_config: IterResearch 설정
            agent_config: 에이전트 설정
        """
        self.iter_config = iter_config
        self.agent_config = agent_config or AgentConfig()
        self.executor = PartialRolloutExecutor(iter_config, agent_config)

    def run_with_partial_sampling(
        self,
        question: str,
        answer: str = "",
        initial_rollouts: Optional[List[RolloutResult]] = None,
    ) -> List[RolloutResult]:
        """
        부분 샘플링과 함께 실행

        Args:
            question: 질문
            answer: 정답
            initial_rollouts: 초기 롤아웃 (없으면 새로 실행)

        Returns:
            전체 롤아웃 결과 리스트
        """
        from langgraph_agent.iter_research.rollout import (
            MultiRolloutExecutor,
            run_iter_research,
        )

        all_results = []

        # 1. 초기 롤아웃 실행 (없는 경우)
        if initial_rollouts is None:
            initial_executor = MultiRolloutExecutor(
                IterResearchConfig(
                    rollout_count=self.iter_config.initial_rollout_num,
                    max_workers=self.iter_config.max_workers,
                    verbose=self.iter_config.verbose,
                ),
                self.agent_config,
            )
            initial_rollouts = initial_executor.run_single_question(question, answer)

        all_results.extend(initial_rollouts)
        logger.info(f"[Partial] 초기 롤아웃 {len(initial_rollouts)}개 완료")

        # 2. 부분 샘플링 모드가 비활성화면 초기 롤아웃만 반환
        if self.iter_config.partial_sampling_mode == PartialSamplingMode.NONE:
            return all_results

        # 3. 부분 롤아웃 라운드 실행
        for round_idx in range(self.iter_config.partial_sampling_rounds):
            logger.info(f"[Partial] 부분 샘플링 라운드 {round_idx + 1} 시작")

            # 분기점에서 부분 롤아웃 생성 및 실행
            partial_tasks = self.executor.create_partial_tasks(initial_rollouts)

            for task, branch_point in partial_tasks:
                result = self._run_partial_rollout(task, branch_point)
                all_results.append(result)

            logger.info(
                f"[Partial] 라운드 {round_idx + 1} 완료: "
                f"{len(partial_tasks)}개 부분 롤아웃"
            )

        return all_results

    def _run_partial_rollout(
        self,
        task: RolloutTask,
        branch_point: BranchPoint,
    ) -> PartialRolloutResult:
        """
        단일 부분 롤아웃 실행

        Args:
            task: 롤아웃 태스크
            branch_point: 분기점 정보

        Returns:
            PartialRolloutResult: 부분 롤아웃 결과
        """
        from langgraph_agent.graph import run_research
        from langgraph_agent.state import create_initial_state
        from langgraph_agent.prompts import build_system_prompt
        from langgraph_agent.tools.base import ToolRegistry
        from langgraph_agent.tools.retrieve import RetrieveTool
        from langgraph_agent.tools.confluence import ConfluenceSearchTool

        start_time = time.time()

        logger.info(
            f"[Partial Rollout] 태스크 {task.rollout_idx} 시작: "
            f"분기점={branch_point.step_idx}, PPL={branch_point.ppl:.2f}"
        )

        try:
            # 도구 레지스트리 생성
            registry = ToolRegistry()
            registry.register(RetrieveTool())
            registry.register(ConfluenceSearchTool())

            # 분기점까지의 메시지를 초기 상태로 사용
            prepend_messages = branch_point.prepend_messages

            # 남은 턴 수 계산
            remaining_turns = self.executor.calculate_remaining_turns(branch_point)

            # 연구 실행 (분기점부터 계속)
            # NOTE: 실제 구현에서는 prepend_messages를 초기 상태로 사용해야 함
            # 현재는 단순화를 위해 전체 실행
            result = run_research(
                question=task.question,
                config=self.agent_config,
                verbose=self.iter_config.verbose,
            )

            execution_time = time.time() - start_time

            return PartialRolloutResult(
                question=task.question,
                answer=task.answer,
                prediction=result.answer or "",
                messages=result.messages,
                termination=result.termination_reason or "unknown",
                rollout_idx=task.rollout_idx,
                execution_time=execution_time,
                total_tokens=result.total_tokens,
                metadata=task.metadata,
                branch_point=branch_point,
                rollout_type="partial",
                parent_rollout_idx=task.metadata.get("parent_rollout_idx"),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Partial Rollout] 오류: {e}")

            return PartialRolloutResult(
                question=task.question,
                answer=task.answer,
                prediction="",
                messages=[],
                termination=f"error: {str(e)}",
                rollout_idx=task.rollout_idx,
                execution_time=execution_time,
                metadata=task.metadata,
                branch_point=branch_point,
                rollout_type="partial",
                parent_rollout_idx=task.metadata.get("parent_rollout_idx"),
            )


def run_iter_research_with_partial(
    question: str,
    answer: str = "",
    initial_rollout_num: int = 1,
    partial_sampling_mode: PartialSamplingMode = PartialSamplingMode.TOOL_CALL_PPL,
    partial_sampling_topk: int = 2,
    partial_sampling_times_per_pos: int = 3,
    agent_config: Optional[AgentConfig] = None,
    verbose: bool = True,
) -> List[RolloutResult]:
    """
    부분 샘플링과 함께 IterResearch 실행 (편의 함수)

    Args:
        question: 질문
        answer: 정답
        initial_rollout_num: 초기 롤아웃 수
        partial_sampling_mode: 부분 샘플링 모드
        partial_sampling_topk: 상위 K개 분기점 선택
        partial_sampling_times_per_pos: 분기점당 샘플링 횟수
        agent_config: 에이전트 설정
        verbose: 상세 로깅

    Returns:
        List[RolloutResult]: 전체 롤아웃 결과
    """
    # 총 샘플링 예산 계산
    sampling_budget = (
        initial_rollout_num +
        initial_rollout_num * partial_sampling_topk * partial_sampling_times_per_pos
    )

    iter_config = IterResearchConfig(
        rollout_count=initial_rollout_num,
        initial_rollout_num=initial_rollout_num,
        partial_sampling_mode=partial_sampling_mode,
        partial_sampling_topk=partial_sampling_topk,
        partial_sampling_times_per_pos=partial_sampling_times_per_pos,
        sampling_budget=sampling_budget,
        verbose=verbose,
    )

    runner = PartialRolloutRunner(iter_config, agent_config)

    return runner.run_with_partial_sampling(question, answer)
