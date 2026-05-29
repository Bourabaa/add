from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from add_pfe.lexical.matcher import (  # noqa: E402,F401
    DEFAULT_ORGANIZATIONS_PATH,
    HybridRequestMatcher,
    OrganizationProfile,
    ascii_fold_text,
    casefold_text,
    combine_request_text,
    compute_alias_bonus,
    compute_keyword_bonus,
    cosine_similarity,
    main,
    normalize_text,
    safe_console_text,
    split_pipe_values,
    unique_preserve_order,
)


if __name__ == "__main__":
    raise SystemExit(main())

