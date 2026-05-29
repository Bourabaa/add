from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from add_pfe.notifications.service import (  # noqa: E402,F401
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FEEDBACKS_PATH,
    DEFAULT_ORGANIZATIONS_PATH,
    NotificationPipelineService,
    NotificationProcessingSummary,
    build_draft_response,
    load_csv_rows,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
