from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from add_pfe.integration.payloads import route_feedback_payloads  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route feedbacks from a JSON payload containing organizations and feedbacks.")
    parser.add_argument("payload", type=Path, help="JSON payload path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidates to return.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.payload.open("r", encoding="utf-8") as payload_file:
        payload = json.load(payload_file)

    result = route_feedback_payloads(
        organizations=payload.get("organizations", []),
        feedbacks=payload.get("feedbacks", []),
        top_k=int(payload.get("top_k", args.top_k) or args.top_k),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
