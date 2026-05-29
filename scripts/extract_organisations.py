from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://data.gov.ma/data/api/3/action/organization_list"
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "raw" / "organisations.csv"
OUTPUT_COLUMNS = ("organization", "organization_slug", "description", "package_count")
DEFAULT_PAGE_SIZE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export organizations from data.gov.ma into a reusable CSV file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the CSV export. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for the CKAN API request.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Number of organizations to request per page.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def fetch_page(timeout: int, page_size: int, offset: int) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "all_fields": "true",
            "include_extras": "true",
            "limit": str(page_size),
            "offset": str(offset),
            "sort": "title asc",
        }
    )
    request = Request(
        url=f"{API_URL}?{query}",
        headers={"User": "PFE-Project/1.0"},
    )

    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    if not payload.get("success"):
        raise RuntimeError("The CKAN API responded with success=false.")

    result = payload.get("result", [])
    if not isinstance(result, list):
        raise RuntimeError("Unexpected organization payload format from CKAN API.")

    return result


def fetch_organizations(timeout: int, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    offset = 0

    while True:
        records = fetch_page(timeout=timeout, page_size=page_size, offset=offset)
        if not records:
            break

        for record in records:
            name = normalize_text(record.get("display_name") or record.get("title") or record.get("name"))
            slug = normalize_text(record.get("name"))
            if not name or name in seen_names:
                continue

            seen_names.add(name)
            rows.append(
                {
                    "organization": name,
                    "organization_slug": slug,
                    "description": normalize_text(record.get("description")),
                    "package_count": int(record.get("package_count") or 0),
                }
            )

        if len(records) < page_size:
            break

        offset += page_size

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        rows = fetch_organizations(timeout=args.timeout, page_size=args.page_size)
        write_csv(rows, args.output)
    except HTTPError as error:
        print(f"HTTP error while fetching organizations: {error.code} {error.reason}")
        return 1
    except URLError as error:
        print(f"Network error while fetching organizations: {error.reason}")
        return 1
    except Exception as error:
        print(f"Unexpected error while exporting organizations: {error}")
        return 1

    print(f"Exported {len(rows)} organizations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
