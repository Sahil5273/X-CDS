"""CLI entrypoint for the X-CDS end-to-end smoke test."""

from __future__ import annotations

import argparse
import os
import sys

from backend.app.pipeline.service import build_default_service
from backend.app.testing.e2e_harness import DEFAULT_QUERY, SmokeWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Clinical query used for the smoke test.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against the default service with real Gemini (requires GOOGLE_API_KEY).",
    )
    return parser


def run_offline_smoke(query: str) -> int:
    with SmokeWorkspace() as workspace:
        service = workspace.build_service()
        result = service.answer(query)

    if not result.contexts:
        print("Smoke test failed: no contexts retrieved.")
        return 1
    if not result.validation_passed:
        print("Smoke test failed: citation validation did not pass.")
        print(result.validation_issues)
        return 1
    if "[1]" not in result.answer:
        print("Smoke test failed: answer missing expected citation marker [1].")
        return 1

    print("Offline smoke test passed.")
    print(f"query={result.query}")
    print(f"answer={result.answer}")
    print(f"contexts={len(result.contexts)} validation_passed={result.validation_passed}")
    return 0


def run_live_smoke(query: str) -> int:
    if not os.getenv("GOOGLE_API_KEY"):
        print("Live smoke test requires GOOGLE_API_KEY.")
        return 1

    service = build_default_service()
    result = service.answer(query)
    if not result.contexts or not result.answer.strip():
        print("Live smoke test failed: empty pipeline response.")
        return 1

    print("Live smoke test passed.")
    print(f"query={result.query}")
    print(f"validation_passed={result.validation_passed}")
    print(f"generation_attempts={result.generation_attempts}")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.live or os.getenv("RUN_LIVE_SMOKE") == "1":
        code = run_live_smoke(args.query)
    else:
        code = run_offline_smoke(args.query)
    sys.exit(code)


if __name__ == "__main__":
    main()
