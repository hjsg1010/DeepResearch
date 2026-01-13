"""
다중 롤아웃 실행기

IterResearch의 핵심인 다중 롤아웃 실행을 담당합니다.
동일 질문에 대해 여러 번 독립적으로 에이전트를 실행합니다.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from langgraph_agent.iter_research.config import (
    IterResearchConfig,
    RolloutResult,
    AggregatedResult,
    PartialSamplingMode,
)
from langgraph_agent.config import AgentConfig
from langgraph_agent.graph import run_research, ResearchResult

logger = logging.getLogger(__name__)


@dataclass
class RolloutTask:
    """롤아웃 태스크 정의"""
    question: str
    answer: str  # Ground truth (있는 경우)
    rollout_idx: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiRolloutExecutor:
    """
    다중 롤아웃 실행기

    동일 질문에 대해 여러 번 독립적으로 에이전트를 실행하고
    결과를 수집합니다.
    """

    def __init__(
        self,
        iter_config: IterResearchConfig,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        실행기 초기화

        Args:
            iter_config: IterResearch 설정
            agent_config: 에이전트 설정
        """
        self.iter_config = iter_config
        self.agent_config = agent_config or AgentConfig()

        # 설정 검증
        iter_config.validate()

        # 출력 디렉토리 생성
        os.makedirs(iter_config.output_dir, exist_ok=True)

        # 통계
        self.total_rollouts = 0
        self.successful_rollouts = 0
        self.failed_rollouts = 0

    def _execute_single_rollout(
        self,
        task: RolloutTask,
    ) -> RolloutResult:
        """
        단일 롤아웃 실행

        Args:
            task: 롤아웃 태스크

        Returns:
            RolloutResult: 롤아웃 결과
        """
        start_time = time.time()

        logger.info(f"[Rollout {task.rollout_idx}] 시작: {task.question[:50]}...")

        try:
            # LangGraph 기반 연구 실행
            result: ResearchResult = run_research(
                question=task.question,
                config=self.agent_config,
                verbose=self.iter_config.verbose,
            )

            execution_time = time.time() - start_time

            rollout_result = RolloutResult(
                question=task.question,
                answer=task.answer,
                prediction=result.answer or "",
                messages=result.messages,
                termination=result.termination_reason or "unknown",
                rollout_idx=task.rollout_idx,
                execution_time=execution_time,
                total_tokens=result.total_tokens,
                metadata={
                    **task.metadata,
                    "total_turns": result.total_turns,
                },
            )

            if rollout_result.is_successful():
                self.successful_rollouts += 1
            else:
                self.failed_rollouts += 1

            self.total_rollouts += 1

            logger.info(
                f"[Rollout {task.rollout_idx}] 완료: "
                f"종료사유={result.termination_reason}, "
                f"시간={execution_time:.1f}s"
            )

            return rollout_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Rollout {task.rollout_idx}] 오류: {e}")

            self.failed_rollouts += 1
            self.total_rollouts += 1

            return RolloutResult(
                question=task.question,
                answer=task.answer,
                prediction="",
                messages=[],
                termination=f"error: {str(e)}",
                rollout_idx=task.rollout_idx,
                execution_time=execution_time,
                metadata=task.metadata,
            )

    def run_single_question(
        self,
        question: str,
        answer: str = "",
        metadata: Optional[dict] = None,
    ) -> List[RolloutResult]:
        """
        단일 질문에 대해 다중 롤아웃 실행

        Args:
            question: 질문
            answer: 정답 (평가용)
            metadata: 추가 메타데이터

        Returns:
            List[RolloutResult]: 롤아웃 결과 리스트
        """
        metadata = metadata or {}

        # 태스크 생성
        tasks = [
            RolloutTask(
                question=question,
                answer=answer,
                rollout_idx=i,
                metadata=metadata.copy(),
            )
            for i in range(1, self.iter_config.rollout_count + 1)
        ]

        results = []

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=self.iter_config.max_workers) as executor:
            futures = {
                executor.submit(self._execute_single_rollout, task): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[Rollout {task.rollout_idx}] Future 오류: {e}")
                    results.append(
                        RolloutResult(
                            question=task.question,
                            answer=task.answer,
                            prediction="",
                            messages=[],
                            termination=f"future_error: {str(e)}",
                            rollout_idx=task.rollout_idx,
                        )
                    )

        # rollout_idx 순으로 정렬
        results.sort(key=lambda r: r.rollout_idx)

        return results

    def run_batch(
        self,
        questions: List[Dict[str, Any]],
        output_prefix: str = "batch",
    ) -> Dict[str, List[RolloutResult]]:
        """
        배치 질문에 대해 다중 롤아웃 실행

        Args:
            questions: 질문 리스트 [{"question": ..., "answer": ...}, ...]
            output_prefix: 출력 파일 접두사

        Returns:
            Dict[str, List[RolloutResult]]: 질문별 롤아웃 결과
        """
        all_results = {}
        output_files = self.iter_config.get_output_files(output_prefix)

        # 이미 처리된 질문 확인 (재시작 지원)
        processed_per_rollout = self._load_processed_questions(output_files)

        total_questions = len(questions)

        for q_idx, q_data in enumerate(questions):
            question = q_data.get("question", "")
            answer = q_data.get("answer", "")

            if not question:
                logger.warning(f"[Batch] 빈 질문 스킵: index={q_idx}")
                continue

            logger.info(f"[Batch] 질문 {q_idx + 1}/{total_questions}: {question[:50]}...")

            # 각 롤아웃별로 실행
            question_results = []

            for rollout_idx in range(1, self.iter_config.rollout_count + 1):
                # 이미 처리된 경우 스킵
                if question in processed_per_rollout.get(rollout_idx, set()):
                    logger.info(f"[Batch] 롤아웃 {rollout_idx} 이미 처리됨, 스킵")
                    continue

                # 단일 롤아웃 실행
                task = RolloutTask(
                    question=question,
                    answer=answer,
                    rollout_idx=rollout_idx,
                    metadata={"batch_index": q_idx},
                )

                result = self._execute_single_rollout(task)
                question_results.append(result)

                # 결과 저장
                self._save_result(result, output_files[rollout_idx])

            all_results[question] = question_results

        return all_results

    def _load_processed_questions(
        self,
        output_files: Dict[int, str],
    ) -> Dict[int, set]:
        """이미 처리된 질문 로드"""
        processed = {}

        for rollout_idx, filepath in output_files.items():
            processed[rollout_idx] = set()

            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed[rollout_idx].add(data["question"].strip())
                except Exception as e:
                    logger.warning(f"처리된 질문 로드 실패: {filepath}, {e}")

        return processed

    def _save_result(self, result: RolloutResult, filepath: str) -> None:
        """결과 저장"""
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"결과 저장 실패: {filepath}, {e}")

    def get_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        return {
            "total_rollouts": self.total_rollouts,
            "successful_rollouts": self.successful_rollouts,
            "failed_rollouts": self.failed_rollouts,
            "success_rate": (
                self.successful_rollouts / self.total_rollouts
                if self.total_rollouts > 0
                else 0.0
            ),
            "config": {
                "rollout_count": self.iter_config.rollout_count,
                "max_workers": self.iter_config.max_workers,
            },
        }


class AsyncMultiRolloutExecutor:
    """
    비동기 다중 롤아웃 실행기

    asyncio 기반으로 더 효율적인 병렬 처리를 제공합니다.
    """

    def __init__(
        self,
        iter_config: IterResearchConfig,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        실행기 초기화

        Args:
            iter_config: IterResearch 설정
            agent_config: 에이전트 설정
        """
        self.iter_config = iter_config
        self.agent_config = agent_config or AgentConfig()

        iter_config.validate()
        os.makedirs(iter_config.output_dir, exist_ok=True)

    async def _execute_single_rollout_async(
        self,
        semaphore: asyncio.Semaphore,
        task: RolloutTask,
    ) -> RolloutResult:
        """
        비동기 단일 롤아웃 실행

        Args:
            semaphore: 동시성 제한 세마포어
            task: 롤아웃 태스크

        Returns:
            RolloutResult: 롤아웃 결과
        """
        from langgraph_agent.graph import arun_research

        async with semaphore:
            start_time = time.time()

            logger.info(f"[Async Rollout {task.rollout_idx}] 시작")

            try:
                result = await arun_research(
                    question=task.question,
                    config=self.agent_config,
                    verbose=self.iter_config.verbose,
                )

                execution_time = time.time() - start_time

                return RolloutResult(
                    question=task.question,
                    answer=task.answer,
                    prediction=result.answer or "",
                    messages=result.messages,
                    termination=result.termination_reason or "unknown",
                    rollout_idx=task.rollout_idx,
                    execution_time=execution_time,
                    total_tokens=result.total_tokens,
                    metadata=task.metadata,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[Async Rollout {task.rollout_idx}] 오류: {e}")

                return RolloutResult(
                    question=task.question,
                    answer=task.answer,
                    prediction="",
                    messages=[],
                    termination=f"error: {str(e)}",
                    rollout_idx=task.rollout_idx,
                    execution_time=execution_time,
                    metadata=task.metadata,
                )

    async def run_single_question_async(
        self,
        question: str,
        answer: str = "",
        metadata: Optional[dict] = None,
    ) -> List[RolloutResult]:
        """
        비동기로 단일 질문에 대해 다중 롤아웃 실행

        Args:
            question: 질문
            answer: 정답
            metadata: 추가 메타데이터

        Returns:
            List[RolloutResult]: 롤아웃 결과 리스트
        """
        metadata = metadata or {}
        semaphore = asyncio.Semaphore(self.iter_config.max_workers)

        tasks = [
            RolloutTask(
                question=question,
                answer=answer,
                rollout_idx=i,
                metadata=metadata.copy(),
            )
            for i in range(1, self.iter_config.rollout_count + 1)
        ]

        coroutines = [
            self._execute_single_rollout_async(semaphore, task)
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    RolloutResult(
                        question=question,
                        answer=answer,
                        prediction="",
                        messages=[],
                        termination=f"exception: {str(result)}",
                        rollout_idx=i + 1,
                    )
                )
            else:
                processed_results.append(result)

        processed_results.sort(key=lambda r: r.rollout_idx)

        return processed_results


def run_iter_research(
    question: str,
    answer: str = "",
    rollout_count: int = 3,
    max_workers: int = 3,
    agent_config: Optional[AgentConfig] = None,
    verbose: bool = True,
) -> List[RolloutResult]:
    """
    IterResearch 실행 (편의 함수)

    Args:
        question: 질문
        answer: 정답 (평가용)
        rollout_count: 롤아웃 수
        max_workers: 병렬 워커 수
        agent_config: 에이전트 설정
        verbose: 상세 로깅

    Returns:
        List[RolloutResult]: 롤아웃 결과 리스트
    """
    iter_config = IterResearchConfig(
        rollout_count=rollout_count,
        max_workers=max_workers,
        verbose=verbose,
    )

    executor = MultiRolloutExecutor(iter_config, agent_config)

    return executor.run_single_question(question, answer)


async def arun_iter_research(
    question: str,
    answer: str = "",
    rollout_count: int = 3,
    max_workers: int = 3,
    agent_config: Optional[AgentConfig] = None,
    verbose: bool = True,
) -> List[RolloutResult]:
    """
    비동기 IterResearch 실행 (편의 함수)

    Args:
        question: 질문
        answer: 정답 (평가용)
        rollout_count: 롤아웃 수
        max_workers: 병렬 워커 수
        agent_config: 에이전트 설정
        verbose: 상세 로깅

    Returns:
        List[RolloutResult]: 롤아웃 결과 리스트
    """
    iter_config = IterResearchConfig(
        rollout_count=rollout_count,
        max_workers=max_workers,
        verbose=verbose,
    )

    executor = AsyncMultiRolloutExecutor(iter_config, agent_config)

    return await executor.run_single_question_async(question, answer)
