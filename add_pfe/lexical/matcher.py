from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from add_pfe.config import ProjectPaths


DEFAULT_ORGANIZATIONS_PATH = ProjectPaths.default().organizations_enriched_path
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def casefold_text(value: Any) -> str:
    return normalize_text(value).casefold()


def ascii_fold_text(value: Any) -> str:
    return unicodedata.normalize("NFKD", casefold_text(value)).encode("ascii", "ignore").decode("ascii")


def safe_console_text(value: Any) -> str:
    text = normalize_text(value)
    try:
        text.encode("cp1252")
        return text
    except UnicodeEncodeError:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def split_pipe_values(value: str) -> list[str]:
    return [normalize_text(item) for item in str(value or "").split("|") if normalize_text(item)]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = casefold_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def cosine_similarity(left: dict[str, float], left_norm: float, right: dict[str, float], right_norm: float) -> float:
    if not left or not right or left_norm == 0 or right_norm == 0:
        return 0.0

    if len(left) > len(right):
        left, right = right, left

    dot_product = sum(weight * right.get(feature, 0.0) for feature, weight in left.items())
    if dot_product == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def char_ngrams(text: str, lengths: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    compact_text = f" {casefold_text(text)} "
    compact_text = re.sub(r"\s+", " ", compact_text)
    grams: list[str] = []
    for length in lengths:
        if len(compact_text) < length:
            continue
        grams.extend(compact_text[index : index + length] for index in range(len(compact_text) - length + 1))
    return grams


def token_features(text: str) -> Counter[str]:
    features: Counter[str] = Counter()
    folded = casefold_text(text)
    ascii_folded = ascii_fold_text(text)

    for token in WORD_PATTERN.findall(folded):
        if len(token) >= 2:
            features[f"w:{token}"] += 1

    if ascii_folded and ascii_folded != folded:
        for token in WORD_PATTERN.findall(ascii_folded):
            if len(token) >= 2:
                features[f"a:{token}"] += 1

    for gram in char_ngrams(text):
        features[f"c:{gram}"] += 1

    return features


def weighted_vector(features: Counter[str], idf_map: dict[str, float]) -> tuple[dict[str, float], float]:
    vector: dict[str, float] = {}
    squared_norm = 0.0

    for feature, frequency in features.items():
        if feature not in idf_map or frequency <= 0:
            continue
        weight = (1.0 + math.log(frequency)) * idf_map[feature]
        vector[feature] = weight
        squared_norm += weight * weight

    return vector, math.sqrt(squared_norm)


@dataclass
class OrganizationProfile:
    organization: str
    organization_slug: str
    full_name: str
    category: str
    aliases: list[str]
    domains: list[str]
    keywords: list[str]
    profile_text: str
    package_count: int
    vector: dict[str, float]
    norm: float
    normalized_aliases: list[str]
    normalized_keywords: list[str]


class HybridRequestMatcher:
    def __init__(self, profiles: list[OrganizationProfile], idf_map: dict[str, float]) -> None:
        self.profiles = profiles
        self.idf_map = idf_map

    @classmethod
    def from_csv(cls, csv_path: Path) -> "HybridRequestMatcher":
        rows = load_csv_rows(csv_path)
        return cls.from_rows(rows, source=str(csv_path))

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], source: str = "json payload") -> "HybridRequestMatcher":
        if not rows:
            raise ValueError(f"No organization profiles found in {source}")

        document_features = [token_features(row.get("profile_text", "")) for row in rows]
        idf_map = build_idf_map(document_features)

        profiles: list[OrganizationProfile] = []
        for row, features in zip(rows, document_features):
            aliases = unique_preserve_order(split_pipe_values(row.get("aliases", "")))
            domains = unique_preserve_order(split_pipe_values(row.get("domains", "")))
            keywords = unique_preserve_order(
                split_pipe_values(row.get("keywords_fr", ""))
                + split_pipe_values(row.get("keywords_en", ""))
                + split_pipe_values(row.get("keywords_ar", ""))
            )
            vector, norm = weighted_vector(features, idf_map)
            profiles.append(
                OrganizationProfile(
                    organization=normalize_text(row.get("organization")),
                    organization_slug=normalize_text(row.get("organization_slug")),
                    full_name=normalize_text(row.get("full_name")),
                    category=normalize_text(row.get("category")),
                    aliases=aliases,
                    domains=domains,
                    keywords=keywords,
                    profile_text=normalize_text(row.get("profile_text")),
                    package_count=int(row.get("package_count") or 0),
                    vector=vector,
                    norm=norm,
                    normalized_aliases=unique_preserve_order(
                        [ascii_fold_text(alias) for alias in aliases if ascii_fold_text(alias)]
                    ),
                    normalized_keywords=unique_preserve_order(
                        [ascii_fold_text(item) for item in keywords + domains if ascii_fold_text(item)]
                    ),
                )
            )

        return cls(profiles=profiles, idf_map=idf_map)

    def match(self, request_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        request_text = normalize_text(request_text)
        if not request_text:
            return []

        request_features = token_features(request_text)
        request_vector, request_norm = weighted_vector(request_features, self.idf_map)
        request_ascii = ascii_fold_text(request_text)
        request_tokens = set(WORD_PATTERN.findall(request_ascii))

        results: list[dict[str, Any]] = []
        for profile in self.profiles:
            semantic_score = cosine_similarity(request_vector, request_norm, profile.vector, profile.norm)
            alias_bonus = compute_alias_bonus(request_ascii, request_tokens, profile.normalized_aliases)
            keyword_bonus, keyword_hits = compute_keyword_bonus(request_ascii, request_tokens, profile.normalized_keywords)
            final_score = min(1.0, semantic_score + alias_bonus + keyword_bonus)
            results.append(
                {
                    "organization": profile.organization,
                    "organization_slug": profile.organization_slug,
                    "full_name": profile.full_name,
                    "category": profile.category,
                    "score": round(final_score, 6),
                    "confidence": round(final_score * 100, 2),
                    "semantic_score": round(semantic_score, 6),
                    "alias_bonus": round(alias_bonus, 6),
                    "keyword_bonus": round(keyword_bonus, 6),
                    "keyword_hits": keyword_hits,
                    "package_count": profile.package_count,
                }
            )

        results.sort(key=lambda item: (-item["score"], -item["package_count"], item["organization"]))
        return results[: max(1, top_k)]


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_idf_map(document_features: list[Counter[str]]) -> dict[str, float]:
    document_count = len(document_features)
    frequencies: Counter[str] = Counter()
    for features in document_features:
        frequencies.update(features.keys())

    return {
        feature: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for feature, frequency in frequencies.items()
    }


def compute_alias_bonus(request_ascii: str, request_tokens: set[str], aliases: list[str]) -> float:
    bonus = 0.0
    for alias in aliases:
        if not alias:
            continue
        if len(alias) <= 4:
            if alias in request_tokens:
                bonus = max(bonus, 0.16)
        elif alias in request_ascii:
            bonus = max(bonus, 0.18)
    return bonus


def compute_keyword_bonus(request_ascii: str, request_tokens: set[str], keywords: list[str]) -> tuple[float, list[str]]:
    hits: list[str] = []
    for keyword in keywords:
        if not keyword or len(hits) >= 4:
            continue
        if len(keyword) <= 4:
            if keyword in request_tokens:
                hits.append(keyword)
        elif keyword in request_ascii:
            hits.append(keyword)

    bonus = min(0.12, 0.03 * len(hits))
    return bonus, hits


def combine_request_text(title: str, description: str) -> str:
    title = normalize_text(title)
    description = normalize_text(description)
    if title and description:
        return f"{title}. {description}"
    return title or description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank the most relevant organizations for a request using the enriched organization profiles."
    )
    parser.add_argument(
        "--organizations",
        type=Path,
        default=DEFAULT_ORGANIZATIONS_PATH,
        help=f"Enriched organizations CSV. Defaults to {DEFAULT_ORGANIZATIONS_PATH}.",
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

    matcher = HybridRequestMatcher.from_csv(args.organizations)
    results = matcher.match(request_text=request_text, top_k=args.top_k)

    if args.json:
        print(json.dumps({"request": request_text, "results": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"Request: {safe_console_text(request_text)}")
    print("")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {safe_console_text(result['organization'])} "
            f"[slug={result['organization_slug']}] "
            f"- {result['confidence']:.2f}%"
        )
        print(
            f"   category={result['category']} "
            f"semantic={result['semantic_score']:.3f} "
            f"alias_bonus={result['alias_bonus']:.3f} "
            f"keyword_bonus={result['keyword_bonus']:.3f}"
        )
        if result["keyword_hits"]:
            print(f"   keyword_hits={', '.join(result['keyword_hits'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
