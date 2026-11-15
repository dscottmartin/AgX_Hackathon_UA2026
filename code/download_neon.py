
"""Download NEON multi-height weather observations.

Run from the repository root:
    python code/download_neon.py
"""

import argparse
import os
from pathlib import Path

import neonutilities as nu
from dotenv import load_dotenv


# Resolve paths relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "neon"

PRODUCTS = {
    "temperature": "DP1.00002.001",
    "wind": "DP1.00001.001",
}


def download_product(
    name: str,
    product_id: str,
    site: str,
    month: str,
    token: str,
) -> None:
    """Download one NEON data product and save its tables."""

    output_dir = DATA_DIR / site / month / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {name} ({product_id})")
    print(f"Site: {site} | Month: {month}")

    data = nu.load_by_product(
        dpid=product_id,
        site=site,
        startdate=month,
        enddate=month,
        package="basic",
        timeindex="30",
        check_size=False,
        include_provisional=True,
        token=token,
    )

    for table_name, df in data.items():
        if not hasattr(df, "to_csv"):
            continue

        output_file = output_dir / f"{table_name}.csv"
        df.to_csv(output_file, index=False)

        print(
            f"Saved: {output_file.name} "
            f"| Rows: {len(df)}"
        )

        # Print columns to help with preprocessing.
        print(f"Columns: {list(df.columns)}")

    print(f"Finished downloading {name}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NEON weather data"
    )

    parser.add_argument("--site", default="SOAP")
    parser.add_argument("--month", default="2025-06")

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("NEON_TOKEN")
    if not token:
        raise RuntimeError(
            "NEON_TOKEN not found. Add it to the .env file."
        )

    failed = []
    for name, product_id in PRODUCTS.items():
        try:
            download_product(
                name=name,
                product_id=product_id,
                site=args.site,
                month=args.month,
                token=token,
            )
        except Exception as exc:
            print(f"Failed to download {name}: {exc}")
            failed.append(name)

    print("\nDownload process finished.")
    if failed:
        raise SystemExit(f"Failed products: {', '.join(failed)}")


if __name__ == "__main__":
    main()
