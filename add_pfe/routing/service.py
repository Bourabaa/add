from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from add_pfe.config import DEFAULT_TOP_K, ProjectPaths
from add_pfe.lexical.matcher import combine_request_text, normalize_text
from add_pfe.semantic.embedding_matcher import (
    DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE,
    DEFAULT_AUTO_ROUTE_MIN_MARGIN,
    DEFAULT_EMBEDDING_WEIGHT,
    EmbeddingRequestMatcher,
    classify_route_decision,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecisionThresholds:
    min_confidence: float = DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE
    min_margin: float = DEFAULT_AUTO_ROUTE_MIN_MARGIN


class RoutingService:
    def __init__(
        self,
        matcher: EmbeddingRequestMatcher,
        default_top_k: int = DEFAULT_TOP_K,
        thresholds: RoutingDecisionThresholds | None = None,
    ) -> None:
        self.matcher = matcher
        self.default_top_k = default_top_k
        self.thresholds = thresholds or RoutingDecisionThresholds()

    @classmethod
    def from_paths(
        cls,
        organizations_path: Path,
        embeddings_cache_path: Path,
        model_cache_dir: Path,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        default_top_k: int = DEFAULT_TOP_K,
        thresholds: RoutingDecisionThresholds | None = None,
    ) -> "RoutingService":
        matcher = EmbeddingRequestMatcher.from_csv(
            csv_path=organizations_path,
            embeddings_cache_path=embeddings_cache_path,
            model_cache_dir=model_cache_dir,
            embedding_weight=embedding_weight,
        )
        return cls(matcher=matcher, default_top_k=default_top_k, thresholds=thresholds)

    @classmethod
    def from_organization_rows(
        cls,
        organization_rows: list[dict[str, Any]],
        embeddings_cache_path: Path,
        model_cache_dir: Path,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        default_top_k: int = DEFAULT_TOP_K,
        thresholds: RoutingDecisionThresholds | None = None,
        source: str = "json payload",
    ) -> "RoutingService":
        matcher = EmbeddingRequestMatcher.from_rows(
            rows=organization_rows,
            embeddings_cache_path=embeddings_cache_path,
            model_cache_dir=model_cache_dir,
            embedding_weight=embedding_weight,
            source=source,
        )
        return cls(matcher=matcher, default_top_k=default_top_k, thresholds=thresholds)

    @classmethod
    def from_project_defaults(
        cls,
        paths: ProjectPaths | None = None,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        default_top_k: int = DEFAULT_TOP_K,
        thresholds: RoutingDecisionThresholds | None = None,
    ) -> "RoutingService":
        resolved_paths = paths or ProjectPaths.default()
        return cls.from_paths(
            organizations_path=resolved_paths.organizations_enriched_path,
            embeddings_cache_path=resolved_paths.embeddings_cache_path,
            model_cache_dir=resolved_paths.local_e5_model_dir,
            embedding_weight=embedding_weight,
            default_top_k=default_top_k,
            thresholds=thresholds,
        )

    def route_text(self, request_text: str, top_k: int | None = None) -> dict[str, Any]:
        total_start = time.perf_counter()
        normalized_request = normalize_text(request_text)
        if not normalized_request:
            return {
                "request_text": "",
                "decision": {
                    "decision": "manual_review",
                    "top1_confidence": 0.0,
                    "top2_confidence": 0.0,
                    "margin": 0.0,
                    "reasons": ["empty_request"],
                },
                "results": [],
                "timings_ms": {
                    "embedding_inference_ms": 0.0,
                    "lexical_matching_ms": 0.0,
                    "hybrid_scoring_ms": 0.0,
                    "total_routing_ms": 0.0,
                },
            }

        resolved_top_k = top_k or self.default_top_k
        results, timings = self.matcher.match_with_timings(request_text=normalized_request, top_k=resolved_top_k)
        decision = classify_route_decision(
            results=results,
            min_confidence=self.thresholds.min_confidence,
            min_margin=self.thresholds.min_margin,
        )
        timings["total_routing_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        LOGGER.info(
            "routing_perf chars=%s embedding_inference_ms=%.2f lexical_matching_ms=%.2f hybrid_scoring_ms=%.2f total_routing_ms=%.2f",
            len(normalized_request),
            timings["embedding_inference_ms"],
            timings["lexical_matching_ms"],
            timings["hybrid_scoring_ms"],
            timings["total_routing_ms"],
        )
        return {
            "request_text": normalized_request,
            "decision": decision,
            "results": results,
            "timings_ms": timings,
        }

    def route_feedback(self, title: str, description: str, top_k: int | None = None) -> dict[str, Any]:
        return self.route_text(combine_request_text(title, description), top_k=top_k)
