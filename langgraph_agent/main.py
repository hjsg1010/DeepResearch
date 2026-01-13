#!/usr/bin/env python3
"""
LangGraph DeepResearch Agent 메인 진입점

사용 예시:
    # 기본 실행
    python -m langgraph_agent.main "LLM 에이전트 개발 방법에 대해 알려줘"

    # 환경 변수로 설정
    LLM_BASE_URL=http://localhost:8000/v1 python -m langgraph_agent.main "질문"
"""

import argparse
import json
import logging
import sys
from typing import Optional

from langgraph_agent.config import AgentConfig
from langgraph_agent.graph import ResearchAgent, run_research


def setup_logging(verbose: bool = False) -> None:
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LangGraph 기반 DeepResearch Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python -m langgraph_agent.main "시스템 아키텍처에 대해 알려줘"
    python -m langgraph_agent.main --verbose "CI/CD 파이프라인 구성 방법"
    python -m langgraph_agent.main --output result.json "보안 정책 문서"
        """
    )

    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="연구할 질문"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로깅 활성화"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="결과를 저장할 JSON 파일 경로"
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="최대 턴 수 (기본값: 100)"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        help="LLM API base URL (기본값: 환경변수 또는 http://localhost:8000/v1)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM API 키 (기본값: 환경변수 또는 'your-api-key-here')"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="모델 이름 (기본값: openai/gpt-oss-120b)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드 활성화"
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 설정 구성
    config = AgentConfig.from_env()

    if args.base_url:
        config.llm.base_url = args.base_url
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.model:
        config.llm.model_name = args.model
    if args.max_turns:
        config.max_turns = args.max_turns

    config.verbose = args.verbose

    # 대화형 모드
    if args.interactive:
        run_interactive(config, logger)
        return

    # 질문이 없으면 에러
    if not args.question:
        parser.print_help()
        sys.exit(1)

    # 연구 실행
    logger.info(f"연구 시작: {args.question}")
    logger.info(f"설정: base_url={config.llm.base_url}, model={config.llm.model_name}")

    try:
        result = run_research(
            question=args.question,
            config=config,
            verbose=args.verbose
        )

        # 결과 출력
        print("\n" + "=" * 60)
        print("연구 결과")
        print("=" * 60)
        print(f"\n질문: {result.question}\n")
        print(f"답변:\n{result.answer}\n")
        print("-" * 60)
        print(f"종료 사유: {result.termination_reason}")
        print(f"총 턴 수: {result.total_turns}")
        print(f"총 토큰: {result.total_tokens}")
        print(f"실행 시간: {result.execution_time:.1f}초")

        # 파일 저장
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"결과 저장: {args.output}")

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_interactive(config: AgentConfig, logger: logging.Logger) -> None:
    """대화형 모드 실행"""
    print("=" * 60)
    print("LangGraph DeepResearch Agent - 대화형 모드")
    print("=" * 60)
    print("'exit' 또는 'quit'을 입력하면 종료됩니다.")
    print("'clear'를 입력하면 화면을 지웁니다.")
    print()

    agent = ResearchAgent(config)

    while True:
        try:
            question = input("\n질문> ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("종료합니다.")
                break

            if question.lower() == "clear":
                print("\033[2J\033[H", end="")
                continue

            if question.lower() == "stats":
                stats = agent.get_stats()
                print(f"\n통계: {json.dumps(stats, ensure_ascii=False, indent=2)}")
                continue

            # 연구 실행
            result = agent.run(question, verbose=config.verbose)

            print("\n" + "-" * 40)
            print("답변:")
            print(result.answer)
            print("-" * 40)
            print(f"(턴: {result.total_turns}, 토큰: {result.total_tokens}, 시간: {result.execution_time:.1f}s)")

        except KeyboardInterrupt:
            print("\n(Ctrl+C 감지, 종료하려면 'exit' 입력)")
            continue
        except EOFError:
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    main()
