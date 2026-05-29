from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from add_pfe.config import ProjectPaths
from add_pfe.lexical.matcher import (
    HybridRequestMatcher,
    ascii_fold_text,
    combine_request_text,
    normalize_text,
    safe_console_text,
    split_pipe_values,
    unique_preserve_order,
)


PROJECT_PATHS = ProjectPaths.default()
DEFAULT_ORGANIZATIONS_PATH = PROJECT_PATHS.organizations_enriched_path
DEFAULT_EMBEDDINGS_CACHE_PATH = PROJECT_PATHS.embeddings_cache_path
DEFAULT_MODEL_CACHE_DIR = PROJECT_PATHS.local_e5_model_dir
DEFAULT_MODEL_REPO = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_SUBFOLDER = "onnx"
DEFAULT_MODEL_FILENAME = "model_O4.onnx"
DEFAULT_MAX_LENGTH = 384
DEFAULT_BATCH_SIZE = 16
DEFAULT_EMBEDDING_WEIGHT = 0.35
DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE = 40.0
DEFAULT_AUTO_ROUTE_MIN_MARGIN = 3.0
LOGGER = logging.getLogger(__name__)


@dataclass
class EmbeddingOrganizationProfile:
    organization: str
    organization_slug: str
    full_name: str
    category: str
    aliases: list[str]
    domains: list[str]
    keywords: list[str]
    profile_text: str
    package_count: int
    embedding: np.ndarray
    normalized_aliases: list[str]
    normalized_keywords: list[str]


class MultilingualE5OnnxEncoder:
    def __init__(
        self,
        model_cache_dir: Path,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_subfolder: str = DEFAULT_MODEL_SUBFOLDER,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_cache_dir = model_cache_dir
        self.model_repo = model_repo
        self.model_subfolder = model_subfolder
        self.model_filename = model_filename
        self.max_length = max_length

        self.model_path, self.tokenizer_path = self._ensure_model_files()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(length=max_length, pad_id=pad_id or 0, pad_token="[PAD]")
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_names = [tensor.name for tensor in self.session.get_inputs()]

    def _ensure_model_files(self) -> tuple[Path, Path]:
        model_path = self.model_cache_dir / self.model_filename
        tokenizer_path = self.model_cache_dir / "tokenizer.json"
        missing_files = [str(path) for path in (model_path, tokenizer_path) if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                "Local model artifacts are required for offline execution. "
                f"Missing: {', '.join(missing_files)}. "
                "Place model_O4.onnx and tokenizer.json under models/multilingual-e5-small-onnx."
            )
        return model_path, tokenizer_path

    def encode(self, texts: list[str], is_query: bool, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        prefix = "query" if is_query else "passage"
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = [f"{prefix}: {normalize_text(text)}" for text in texts[start : start + batch_size]]
            encodings = self.tokenizer.encode_batch(batch_texts)
            input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            attention_mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            token_type_ids = np.asarray([encoding.type_ids for encoding in encodings], dtype=np.int64)

            feeds: dict[str, np.ndarray] = {}
            for input_name in self.input_names:
                lowered_name = input_name.lower()
                if "input_ids" in lowered_name:
                    feeds[input_name] = input_ids
                elif "attention_mask" in lowered_name:
                    feeds[input_name] = attention_mask
                elif "token_type_ids" in lowered_name:
                    feeds[input_name] = token_type_ids

            outputs = self.session.run(None, feeds)
            raw_output = np.asarray(outputs[0])
            if raw_output.ndim == 3:
                pooled_output = mean_pool(raw_output, attention_mask)
            elif raw_output.ndim == 2:
                pooled_output = raw_output
            else:
                raise RuntimeError(f"Unexpected ONNX output shape: {raw_output.shape}")

            all_embeddings.append(l2_normalize(pooled_output).astype(np.float32))

        return np.vstack(all_embeddings)


class EmbeddingRequestMatcher:
    def __init__(
        self,
        profiles: list[EmbeddingOrganizationProfile],
        lexical_matcher: HybridRequestMatcher,
        encoder: MultilingualE5OnnxEncoder,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    ) -> None:
        self.profiles = profiles
        self.lexical_matcher = lexical_matcher
        self.encoder = encoder
        self.embedding_weight = embedding_weight
        self.lexical_weight = 1.0 - embedding_weight

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        embeddings_cache_path: Path = DEFAULT_EMBEDDINGS_CACHE_PATH,
        model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    ) -> "EmbeddingRequestMatcher":
        rows = load_csv_rows(csv_path)
        return cls.from_rows(
            rows=rows,
            embeddings_cache_path=embeddings_cache_path,
            model_cache_dir=model_cache_dir,
            model_repo=model_repo,
            model_filename=model_filename,
            max_length=max_length,
            batch_size=batch_size,
            embedding_weight=embedding_weight,
            source=str(csv_path),
        )

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        embeddings_cache_path: Path = DEFAULT_EMBEDDINGS_CACHE_PATH,
        model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        source: str = "json payload",
    ) -> "EmbeddingRequestMatcher":
        if not rows:
            raise ValueError(f"No organization profiles found in {source}")

        encoder = MultilingualE5OnnxEncoder(
            model_cache_dir=model_cache_dir,
            model_repo=model_repo,
            model_filename=model_filename,
            max_length=max_length,
        )
        signature = build_rows_signature(rows=rows, model_repo=model_repo, model_filename=model_filename, max_length=max_length)
        cached_embeddings = load_embeddings_cache(embeddings_cache_path=embeddings_cache_path, signature=signature)
        if cached_embeddings is None:
            cached_embeddings = encoder.encode(
                texts=[row.get("profile_text", "") for row in rows],
                is_query=False,
                batch_size=batch_size,
            )
            save_embeddings_cache(embeddings_cache_path=embeddings_cache_path, signature=signature, embeddings=cached_embeddings)

        profiles: list[EmbeddingOrganizationProfile] = []
        for row, embedding in zip(rows, cached_embeddings):
            aliases = unique_preserve_order(split_pipe_values(row.get("aliases", "")))
            domains = unique_preserve_order(split_pipe_values(row.get("domains", "")))
            keywords = unique_preserve_order(
                split_pipe_values(row.get("keywords_fr", ""))
                + split_pipe_values(row.get("keywords_en", ""))
                + split_pipe_values(row.get("keywords_ar", ""))
            )
            profiles.append(
                EmbeddingOrganizationProfile(
                    organization=normalize_text(row.get("organization")),
                    organization_slug=normalize_text(row.get("organization_slug")),
                    full_name=normalize_text(row.get("full_name")),
                    category=normalize_text(row.get("category")),
                    aliases=aliases,
                    domains=domains,
                    keywords=keywords,
                    profile_text=normalize_text(row.get("profile_text")),
                    package_count=int(row.get("package_count") or 0),
                    embedding=np.asarray(embedding, dtype=np.float32),
                    normalized_aliases=unique_preserve_order(
                        [ascii_fold_text(alias) for alias in aliases if ascii_fold_text(alias)]
                    ),
                    normalized_keywords=unique_preserve_order(
                        [ascii_fold_text(item) for item in keywords + domains if ascii_fold_text(item)]
                    ),
                )
            )

        lexical_matcher = HybridRequestMatcher.from_rows(rows, source=source)
        return cls(
            profiles=profiles,
            lexical_matcher=lexical_matcher,
            encoder=encoder,
            embedding_weight=embedding_weight,
        )

    def match(self, request_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        results, _ = self.match_with_timings(request_text=request_text, top_k=top_k)
        return results

    def match_with_timings(self, request_text: str, top_k: int = 3) -> tuple[list[dict[str, Any]], dict[str, float]]:
        request_text = normalize_text(request_text)
        if not request_text:
            return [], {
                "embedding_inference_ms": 0.0,
                "lexical_matching_ms": 0.0,
                "hybrid_scoring_ms": 0.0,
            }

        embedding_start = time.perf_counter()
        query_embedding = self.encoder.encode([request_text], is_query=True)[0]
        embedding_inference_ms = elapsed_ms(embedding_start)

        lexical_start = time.perf_counter()
        lexical_results = self.lexical_matcher.match(request_text=request_text, top_k=len(self.profiles))
        lexical_by_slug = {item["organization_slug"]: item for item in lexical_results}
        lexical_matching_ms = elapsed_ms(lexical_start)

        hybrid_start = time.perf_counter()
        results: list[dict[str, Any]] = []
        for profile in self.profiles:
            embedding_score = max(0.0, float(np.dot(query_embedding, profile.embedding)))
            lexical_result = lexical_by_slug.get(profile.organization_slug, {})
            lexical_score = float(lexical_result.get("score", 0.0))
            final_score = min(1.0, self.embedding_weight * embedding_score + self.lexical_weight * lexical_score)
            results.append(
                {
                    "organization": profile.organization,
                    "organization_slug": profile.organization_slug,
                    "full_name": profile.full_name,
                    "category": profile.category,
                    "score": round(final_score, 6),
                    "confidence": round(final_score * 100, 2),
                    "embedding_score": round(embedding_score, 6),
                    "lexical_score": round(lexical_score, 6),
                    "lexical_semantic_score": round(float(lexical_result.get("semantic_score", 0.0)), 6),
                    "alias_bonus": round(float(lexical_result.get("alias_bonus", 0.0)), 6),
                    "keyword_bonus": round(float(lexical_result.get("keyword_bonus", 0.0)), 6),
                    "keyword_hits": lexical_result.get("keyword_hits", []),
                    "package_count": profile.package_count,
                }
            )

        results.sort(key=lambda item: (-item["score"], -item["package_count"], item["organization"]))
        hybrid_scoring_ms = elapsed_ms(hybrid_start)
        timings = {
            "embedding_inference_ms": embedding_inference_ms,
            "lexical_matching_ms": lexical_matching_ms,
            "hybrid_scoring_ms": hybrid_scoring_ms,
        }
        LOGGER.info(
            "matcher_perf embedding_inference_ms=%.2f lexical_matching_ms=%.2f hybrid_scoring_ms=%.2f",
            embedding_inference_ms,
            lexical_matching_ms,
            hybrid_scoring_ms,
        )
        return results[: max(1, top_k)], timings


def classify_route_decision(
    results: list[dict[str, Any]],
    min_confidence: float = DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_AUTO_ROUTE_MIN_MARGIN,
) -> dict[str, Any]:
    if not results:
        return {
            "decision": "manual_review",
            "top1_confidence": 0.0,
            "top2_confidence": 0.0,
            "margin": 0.0,
            "reasons": ["no_candidates"],
        }

    top1_confidence = float(results[0].get("confidence", 0.0))
    top2_confidence = float(results[1].get("confidence", 0.0)) if len(results) > 1 else 0.0
    margin = top1_confidence - top2_confidence

    reasons: list[str] = []
    if top1_confidence < min_confidence:
        reasons.append("low_confidence")
    if margin < min_margin:
        reasons.append("low_margin")

    return {
        "decision": "auto_route" if not reasons else "manual_review",
        "top1_confidence": round(top1_confidence, 2),
        "top2_confidence": round(top2_confidence, 2),
        "margin": round(margin, 2),
        "reasons": reasons,
    }


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., None]
    masked_hidden_state = last_hidden_state * mask
    summed_hidden_state = masked_hidden_state.sum(axis=1)
    token_counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed_hidden_state / token_counts


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_rows_signature(rows: list[dict[str, str]], model_repo: str, model_filename: str, max_length: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model_repo}|{model_filename}|{max_length}".encode("utf-8"))
    for row in rows:
        digest.update(normalize_text(row.get("organization_slug")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalize_text(row.get("profile_text")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_embeddings_cache(embeddings_cache_path: Path, signature: str) -> np.ndarray | None:
    if not embeddings_cache_path.exists():
        return None

    with embeddings_cache_path.open("rb") as cache_file:
        payload = pickle.load(cache_file)

    if payload.get("signature") != signature:
        return None
    return np.asarray(payload.get("embeddings"), dtype=np.float32)


def save_embeddings_cache(embeddings_cache_path: Path, signature: str, embeddings: np.ndarray) -> None:
    embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with embeddings_cache_path.open("wb") as cache_file:
        pickle.dump({"signature": signature, "embeddings": embeddings}, cache_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank organizations with a multilingual ONNX embeddings matcher blended with lexical reranking."
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
        default=DEFAULT_EMBEDDINGS_CACHE_PATH,
        help=f"Cache file for precomputed organization embeddings. Defaults to {DEFAULT_EMBEDDINGS_CACHE_PATH}.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
        help=f"Directory used to cache ONNX model files. Defaults to {DEFAULT_MODEL_CACHE_DIR}.",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=DEFAULT_EMBEDDING_WEIGHT,
        help=f"Weight assigned to embeddings in the final score. Defaults to {DEFAULT_EMBEDDING_WEIGHT}.",
    )
    parser.add_argument(
        "--auto-route-min-confidence",
        type=float,
        default=DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE,
        help=(
            "Minimum top-1 confidence required to auto-route a request. "
            f"Defaults to {DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE}."
        ),
    )
    parser.add_argument(
        "--auto-route-min-margin",
        type=float,
        default=DEFAULT_AUTO_ROUTE_MIN_MARGIN,
        help=(
            "Minimum top1-top2 confidence margin required to auto-route a request. "
            f"Defaults to {DEFAULT_AUTO_ROUTE_MIN_MARGIN}."
        ),
    )
    parser.add_argument("--request", help="Single request text to classify.")
    parser.add_argument("--title", default="", help="Optional title to combine with the description.")
    parser.add_argument("--description", default="", help="Optional description to combine with the title.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a readable text summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_text = normalize_text(args.request or combine_request_text(args.title, args.description))
    if not request_text:
        raise SystemExit("Provide --request or --title/--description.")

    matcher = EmbeddingRequestMatcher.from_csv(
        csv_path=args.organizations,
        embeddings_cache_path=args.embeddings_cache,
        model_cache_dir=args.model_cache_dir,
        embedding_weight=args.embedding_weight,
    )
    results = matcher.match(request_text=request_text, top_k=args.top_k)
    decision = classify_route_decision(
        results=results,
        min_confidence=args.auto_route_min_confidence,
        min_margin=args.auto_route_min_margin,
    )

    if args.json:
        print(json.dumps({"request": request_text, "decision": decision, "results": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"Request: {safe_console_text(request_text)}")
    print("")
    print(
        f"Decision: {decision['decision']} "
        f"(top1={decision['top1_confidence']:.2f}%, "
        f"top2={decision['top2_confidence']:.2f}%, "
        f"margin={decision['margin']:.2f} pts)"
    )
    print(
        f"Thresholds: confidence>={args.auto_route_min_confidence:.2f}% "
        f"and margin>={args.auto_route_min_margin:.2f} pts"
    )
    if decision["reasons"]:
        print(f"Reasons: {', '.join(decision['reasons'])}")
    print("")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {safe_console_text(result['organization'])} "
            f"[slug={result['organization_slug']}] "
            f"- {result['confidence']:.2f}%"
        )
        print(
            f"   category={result['category']} "
            f"embedding={result['embedding_score']:.3f} "
            f"lexical={result['lexical_score']:.3f} "
            f"alias_bonus={result['alias_bonus']:.3f} "
            f"keyword_bonus={result['keyword_bonus']:.3f}"
        )
        if result["keyword_hits"]:
            print(f"   keyword_hits={', '.join(result['keyword_hits'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
