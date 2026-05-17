"""
cli.py — CLI entrypoint for the RAG agent.

"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("orchestrator")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the RAG multi-agent system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The question to ask the system.",
    )
    parser.add_argument(
        "--session-id", "-s",
        default=None,
        dest="session_id",
        help="Session UUID for multi-turn conversation. Omit to start a new session.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show intent classification, sources, and session_id.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON instead of formatted text.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("orchestrator").setLevel(logging.INFO)
        logging.getLogger("retriever").setLevel(logging.INFO)
        logging.getLogger("generator").setLevel(logging.INFO)

    try:
        from orchestrator.agent import run_query
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print("Have you run: pip install -r requirements.txt?", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_query(query=args.query, session_id=args.session_id)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        print(json.dumps(result, indent=2))
        return

    print("\n" + "─" * 60)
    print(result["answer"])
    print("─" * 60)

    if args.verbose or result.get("sources"):
        if result.get("sources"):
            print(f"\nSources: {', '.join(result['sources'])}")
        if args.verbose:
            print(f"Intent:     {result.get('intent', 'unknown')}")
            print(f"Session ID: {result['session_id']}")
            if result.get("error"):
                print(f"Error:      {result['error']}")

    print() 


if __name__ == "__main__":
    main()