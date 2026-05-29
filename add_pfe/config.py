from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        return cls(root_dir=ROOT_DIR)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    @property
    def templates_dir(self) -> Path:
        return self.root_dir / "templates"

    @property
    def organizations_raw_path(self) -> Path:
        return self.raw_data_dir / "organisations.csv"

    @property
    def feedbacks_raw_path(self) -> Path:
        return self.raw_data_dir / "feedbacks.csv"

    @property
    def organizations_enriched_path(self) -> Path:
        return self.processed_data_dir / "organisations_enriched.csv"

    @property
    def feedbacks_labeled_path(self) -> Path:
        return self.processed_data_dir / "feedbacks_labeled.csv"

    @property
    def feedbacks_eval_ready_path(self) -> Path:
        return self.processed_data_dir / "feedbacks_eval_ready.csv"

    @property
    def notifications_db_path(self) -> Path:
        return self.processed_data_dir / "notifications.db"

    @property
    def embeddings_cache_path(self) -> Path:
        return self.cache_dir / "org_profile_embeddings_e5.pkl"

    @property
    def model_cache_dir(self) -> Path:
        return self.cache_dir / "models" / "multilingual-e5-small-onnx"

    @property
    def local_e5_model_dir(self) -> Path:
        return self.models_dir / "multilingual-e5-small-onnx"

    @property
    def embedding_report_path(self) -> Path:
        return self.reports_dir / "embedding_matcher_evaluation.csv"

    @property
    def lexical_report_path(self) -> Path:
        return self.reports_dir / "matcher_evaluation.csv"


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    debug: bool


def load_dashboard_config() -> DashboardConfig:
    return DashboardConfig(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )


STATUS_FILTERS = ("all", "pending_review", "pending_auto_route", "approved", "rejected", "published")
REVIEW_STATUSES = STATUS_FILTERS[1:]
DEFAULT_TOP_K = 3
DEFAULT_POLL_INTERVAL_SECONDS = 300
DEFAULT_MAX_FEEDBACK_PAGES = 12
