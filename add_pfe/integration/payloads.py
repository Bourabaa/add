from __future__ import annotations

from pathlib import Path
from typing import Any

from add_pfe.config import DEFAULT_TOP_K, ProjectPaths
from add_pfe.lexical.matcher import normalize_text
from add_pfe.routing.service import RoutingDecisionThresholds, RoutingService
from add_pfe.semantic.embedding_matcher import DEFAULT_EMBEDDING_WEIGHT


PROJECT_PATHS = ProjectPaths.default()


def normalize_organization_payload(payload: dict[str, Any]) -> dict[str, Any]:
    slug = normalize_text(payload.get("organization_slug") or payload.get("slug") or payload.get("name"))
    organization = normalize_text(payload.get("organization") or payload.get("title") or payload.get("display_name") or slug)
    aliases = normalize_pipe_or_list(payload.get("aliases"))
    domains = normalize_pipe_or_list(payload.get("domains"))
    keywords_fr = normalize_pipe_or_list(payload.get("keywords_fr") or payload.get("keywords"))
    keywords_en = normalize_pipe_or_list(payload.get("keywords_en"))
    keywords_ar = normalize_pipe_or_list(payload.get("keywords_ar"))
    profile_text = normalize_text(payload.get("profile_text"))

    if not profile_text:
        profile_text = " ".join(
            item
            for item in [
                organization,
                normalize_text(payload.get("full_name")),
                aliases.replace("|", " "),
                domains.replace("|", " "),
                keywords_fr.replace("|", " "),
                keywords_en.replace("|", " "),
                keywords_ar.replace("|", " "),
            ]
            if item
        )

    return {
        "organization": organization,
        "organization_slug": slug,
        "full_name": normalize_text(payload.get("full_name") or organization),
        "category": normalize_text(payload.get("category")),
        "aliases": aliases,
        "domains": domains,
        "keywords_fr": keywords_fr,
        "keywords_en": keywords_en,
        "keywords_ar": keywords_ar,
        "profile_text": profile_text,
        "package_count": int(payload.get("package_count") or 0),
    }


def normalize_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    feedback_id = normalize_text(payload.get("feedback_id") or payload.get("id") or payload.get("feedback_url"))
    title = normalize_text(payload.get("title"))
    description = normalize_text(payload.get("description") or payload.get("body") or payload.get("text"))
    return {
        "feedback_id": feedback_id,
        "feedback_url": normalize_text(payload.get("feedback_url") or payload.get("url") or feedback_id),
        "title": title,
        "description": description,
    }


def normalize_pipe_or_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(normalize_text(item) for item in value if normalize_text(item))
    return "|".join(normalize_text(item) for item in str(value).split("|") if normalize_text(item))


def route_feedback_payloads(
    organizations: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    embeddings_cache_path: Path | None = None,
    model_cache_dir: Path | None = None,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    thresholds: RoutingDecisionThresholds | None = None,
) -> dict[str, Any]:
    normalized_organizations = [normalize_organization_payload(item) for item in organizations]
    normalized_feedbacks = [normalize_feedback_payload(item) for item in feedbacks]
    if not normalized_organizations:
        raise ValueError("The organizations list is required.")

    routing_service = RoutingService.from_organization_rows(
        organization_rows=normalized_organizations,
        embeddings_cache_path=embeddings_cache_path or PROJECT_PATHS.embeddings_cache_path,
        model_cache_dir=model_cache_dir or PROJECT_PATHS.local_e5_model_dir,
        embedding_weight=embedding_weight,
        default_top_k=top_k,
        thresholds=thresholds,
    )

    routed_feedbacks: list[dict[str, Any]] = []
    for feedback in normalized_feedbacks:
        routing_result = routing_service.route_feedback(
            title=feedback["title"],
            description=feedback["description"],
            top_k=top_k,
        )
        decision = routing_result["decision"]
        routed_feedbacks.append(
            {
                "feedback_id": feedback["feedback_id"],
                "feedback_url": feedback["feedback_url"],
                "title": feedback["title"],
                "decision": decision["decision"],
                "top1_confidence": decision["top1_confidence"],
                "top2_confidence": decision["top2_confidence"],
                "margin": decision["margin"],
                "reasons": decision["reasons"],
                "timings_ms": routing_result["timings_ms"],
                "candidates": routing_result["results"],
            }
        )

    return {
        "processed": len(routed_feedbacks),
        "results": routed_feedbacks,
    }
