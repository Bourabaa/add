from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from add_pfe.semantic.embedding_matcher import (  # noqa: E402,F401
    DEFAULT_AUTO_ROUTE_MIN_CONFIDENCE,
    DEFAULT_AUTO_ROUTE_MIN_MARGIN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_EMBEDDINGS_CACHE_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_SUBFOLDER,
    DEFAULT_ORGANIZATIONS_PATH,
    EmbeddingOrganizationProfile,
    EmbeddingRequestMatcher,
    MultilingualE5OnnxEncoder,
    classify_route_decision,
    l2_normalize,
    load_csv_rows,
    main,
    mean_pool,
)


if __name__ == "__main__":
    raise SystemExit(main())

