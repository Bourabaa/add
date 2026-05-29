from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from add_pfe.config import ProjectPaths
from matcher import HybridRequestMatcher, combine_request_text, normalize_text

PROJECT_PATHS = ProjectPaths.default()
DEFAULT_ORGANIZATIONS_PATH = PROJECT_PATHS.organizations_enriched_path
DEFAULT_FEEDBACKS_PATH = PROJECT_PATHS.feedbacks_eval_ready_path
DEFAULT_REPORT_PATH = PROJECT_PATHS.lexical_report_path


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the request matcher on feedback rows that have a strong organization label."
    )
    parser.add_argument(
        "--organizations",
        type=Path,
        default=DEFAULT_ORGANIZATIONS_PATH,
        help=f"Enriched organizations CSV. Defaults to {DEFAULT_ORGANIZATIONS_PATH}.",
    )
    parser.add_argument(
        "--feedbacks",
        type=Path,
        default=DEFAULT_FEEDBACKS_PATH,
        help=f"Feedback CSV. Defaults to {DEFAULT_FEEDBACKS_PATH}.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k cutoff used for the evaluation.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"CSV file where per-row evaluation details are written. Defaults to {DEFAULT_REPORT_PATH}.",
    )
    return parser.parse_args()


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "feedback_url",
                "title",
                "true_org_slug",
                "true_org_name",
                "predicted_top1_slug",
                "predicted_top1_name",
                "predicted_top1_confidence",
                "top_k_slugs",
                "top1_correct",
                "topk_correct",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_console_text(value: str) -> str:
    text = normalize_text(value)
    try:
        text.encode("cp1252")
        return text
    except UnicodeEncodeError:
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )


def main() -> int:
    args = parse_args()
    matcher = HybridRequestMatcher.from_csv(args.organizations)
    feedback_rows = load_csv_rows(args.feedbacks)

    labeled_rows = [
        row
        for row in feedback_rows
        if normalize_text(row.get("evaluation_label_strength") or row.get("label_strength")) == "strong"
        and normalize_text(row.get("evaluation_org_slug") or row.get("primary_org_slug"))
    ]

    if not labeled_rows:
        print("No strong-labeled feedback rows found. Scrape more pages or add labeled examples first.")
        return 0

    report_rows: list[dict[str, Any]] = []
    top1_correct = 0
    topk_correct = 0

    for row in labeled_rows:
        request_text = combine_request_text(row.get("title", ""), row.get("description", ""))
        results = matcher.match(request_text=request_text, top_k=args.top_k)
        predicted_top1 = results[0] if results else {}
        predicted_slugs = [result["organization_slug"] for result in results]
        true_slug = normalize_text(row.get("evaluation_org_slug") or row.get("primary_org_slug"))
        true_name = normalize_text(row.get("evaluation_org_name") or row.get("primary_org_name"))
        is_top1_correct = predicted_top1.get("organization_slug") == true_slug
        is_topk_correct = true_slug in predicted_slugs

        top1_correct += int(is_top1_correct)
        topk_correct += int(is_topk_correct)

        report_rows.append(
            {
                "feedback_url": row.get("feedback_url", ""),
                "title": row.get("title", ""),
                "true_org_slug": true_slug,
                "true_org_name": true_name,
                "predicted_top1_slug": predicted_top1.get("organization_slug", ""),
                "predicted_top1_name": predicted_top1.get("organization", ""),
                "predicted_top1_confidence": predicted_top1.get("confidence", ""),
                "top_k_slugs": "|".join(predicted_slugs),
                "top1_correct": int(is_top1_correct),
                "topk_correct": int(is_topk_correct),
            }
        )

    total = len(labeled_rows)
    top1_accuracy = (top1_correct / total) * 100
    topk_accuracy = (topk_correct / total) * 100

    write_report(report_rows, args.report)

    print(f"Strong-labeled rows: {total}")
    print(f"Top-1 accuracy: {top1_correct}/{total} = {top1_accuracy:.2f}%")
    print(f"Top-{args.top_k} accuracy: {topk_correct}/{total} = {topk_accuracy:.2f}%")
    print(f"Detailed report: {args.report}")

    print("")
    print("Sample predictions:")
    for row in report_rows[: min(5, len(report_rows))]:
        status = "OK" if row["top1_correct"] else "MISS"
        print(
            f"- {status} | true={row['true_org_slug']} | "
            f"pred={row['predicted_top1_slug']} | {safe_console_text(row['title'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
