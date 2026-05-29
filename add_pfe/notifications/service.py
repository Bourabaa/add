from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from add_pfe.config import DEFAULT_TOP_K, ProjectPaths
from add_pfe.lexical.matcher import normalize_text
from add_pfe.notifications.repository import DEFAULT_DB_PATH, connect_db, upsert_notification
from add_pfe.routing.service import RoutingDecisionThresholds, RoutingService
from add_pfe.semantic.embedding_matcher import DEFAULT_EMBEDDING_WEIGHT


PROJECT_PATHS = ProjectPaths.default()
DEFAULT_FEEDBACKS_PATH = PROJECT_PATHS.feedbacks_eval_ready_path
DEFAULT_ORGANIZATIONS_PATH = PROJECT_PATHS.organizations_enriched_path


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_draft_response(title: str, results: list[dict[str, object]], decision: dict[str, object]) -> str:
    if not results:
        return "Aucune organisation pertinente n'a pu etre identifiee automatiquement. Une verification manuelle est necessaire."

    top1 = results[0]
    if decision["decision"] == "auto_route":
        return (
            f"L'organisation la plus pertinente pour traiter la demande '{title}' semble etre "
            f"{top1['organization']} avec une confiance de {top1['confidence']:.2f}%. "
            "Une validation finale peut etre effectuee avant publication sur le portail."
        )

    suggestions = ", ".join(f"{item['organization']} ({item['confidence']:.2f}%)" for item in results[:3])
    return (
        f"Cette demande doit etre verifiee manuellement. Suggestions d'organisations: {suggestions}. "
        "Merci de confirmer ou corriger l'organisation avant de publier une reponse."
    )


@dataclass
class NotificationProcessingSummary:
    processed: int = 0
    inserted: int = 0
    updated: int = 0
    pending_auto_route: int = 0
    pending_manual_review: int = 0


class NotificationPipelineService:
    def __init__(self, routing_service: RoutingService, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.routing_service = routing_service
        self.db_path = db_path

    @classmethod
    def from_project_defaults(
        cls,
        paths: ProjectPaths | None = None,
        db_path: Path | None = None,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        top_k: int = DEFAULT_TOP_K,
        thresholds: RoutingDecisionThresholds | None = None,
    ) -> "NotificationPipelineService":
        resolved_paths = paths or ProjectPaths.default()
        routing_service = RoutingService.from_project_defaults(
            paths=resolved_paths,
            embedding_weight=embedding_weight,
            default_top_k=top_k,
            thresholds=thresholds,
        )
        return cls(routing_service=routing_service, db_path=db_path or resolved_paths.notifications_db_path)

    def build_notification_payload(self, row: dict[str, str], top_k: int | None = None) -> dict[str, object] | None:
        title = row.get("title", "")
        description = row.get("description", "")
        routing_result = self.routing_service.route_feedback(title=title, description=description, top_k=top_k)
        request_text = routing_result["request_text"]
        if not normalize_text(request_text):
            return None

        results = routing_result["results"]
        decision = routing_result["decision"]
        top1 = results[0] if results else {}
        review_status = "pending_auto_route" if decision["decision"] == "auto_route" else "pending_review"

        return {
            "feedback_url": row.get("feedback_url", ""),
            "title": title,
            "description": description,
            "predicted_org_slug": top1.get("organization_slug", ""),
            "predicted_org_name": top1.get("organization", ""),
            "top1_confidence": decision.get("top1_confidence", 0.0),
            "top2_confidence": decision.get("top2_confidence", 0.0),
            "top1_margin": decision.get("margin", 0.0),
            "routing_decision": decision.get("decision", "manual_review"),
            "routing_reasons": decision.get("reasons", []),
            "top_candidates": results,
            "draft_response": build_draft_response(title, results, decision),
            "review_status": review_status,
        }

    def process_feedback_rows(self, feedback_rows: list[dict[str, str]], top_k: int | None = None) -> NotificationProcessingSummary:
        summary = NotificationProcessingSummary()
        connection = connect_db(self.db_path)
        try:
            for row in feedback_rows:
                payload = self.build_notification_payload(row, top_k=top_k)
                if payload is None:
                    continue

                _, operation = upsert_notification(connection, payload)
                summary.processed += 1
                if payload["routing_decision"] == "auto_route":
                    summary.pending_auto_route += 1
                else:
                    summary.pending_manual_review += 1

                if operation == "inserted":
                    summary.inserted += 1
                else:
                    summary.updated += 1
        finally:
            connection.close()

        return summary

    def process_feedback_file(self, feedbacks_path: Path, top_k: int | None = None, limit: int = 0) -> NotificationProcessingSummary:
        rows = load_csv_rows(feedbacks_path)
        if limit > 0:
            rows = rows[:limit]
        return self.process_feedback_rows(rows, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh admin notifications from processed feedback requests.")
    parser.add_argument(
        "--feedbacks",
        type=Path,
        default=DEFAULT_FEEDBACKS_PATH,
        help=f"Processed feedback CSV. Defaults to {DEFAULT_FEEDBACKS_PATH}.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--organizations",
        type=Path,
        default=DEFAULT_ORGANIZATIONS_PATH,
        help=f"Enriched organizations CSV. Defaults to {DEFAULT_ORGANIZATIONS_PATH}.",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=PROJECT_PATHS.embeddings_cache_path,
        help=f"Embeddings cache path. Defaults to {PROJECT_PATHS.embeddings_cache_path}.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=PROJECT_PATHS.local_e5_model_dir,
        help=f"Local ONNX model directory. Defaults to {PROJECT_PATHS.local_e5_model_dir}.",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=DEFAULT_EMBEDDING_WEIGHT,
        help=f"Embeddings blend weight. Defaults to {DEFAULT_EMBEDDING_WEIGHT}.",
    )
    parser.add_argument(
        "--auto-route-min-confidence",
        type=float,
        default=RoutingDecisionThresholds().min_confidence,
        help=f"Minimum top-1 confidence for auto-route. Defaults to {RoutingDecisionThresholds().min_confidence}.",
    )
    parser.add_argument(
        "--auto-route-min-margin",
        type=float,
        default=RoutingDecisionThresholds().min_margin,
        help=f"Minimum top1-top2 margin for auto-route. Defaults to {RoutingDecisionThresholds().min_margin}.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of candidates to store per request.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on processed rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routing_service = RoutingService.from_paths(
        organizations_path=args.organizations,
        embeddings_cache_path=args.embeddings_cache,
        model_cache_dir=args.model_cache_dir,
        embedding_weight=args.embedding_weight,
        default_top_k=args.top_k,
        thresholds=RoutingDecisionThresholds(
            min_confidence=args.auto_route_min_confidence,
            min_margin=args.auto_route_min_margin,
        ),
    )
    pipeline = NotificationPipelineService(routing_service=routing_service, db_path=args.db)
    summary = pipeline.process_feedback_file(args.feedbacks, top_k=args.top_k, limit=args.limit)

    print(f"Notifications processed: {summary.processed}")
    print(f"Inserted: {summary.inserted}")
    print(f"Updated: {summary.updated}")
    print(f"Pending auto-route: {summary.pending_auto_route}")
    print(f"Pending manual review: {summary.pending_manual_review}")
    print(f"Database: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
