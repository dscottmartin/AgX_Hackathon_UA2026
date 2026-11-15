
"""Download hourly weather data from California CIMIS stations.

Example:
    python code/download_cimis.py \
        --station 39 \
        --start 2026-11-01 \
        --end 2026-11-07
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from python_cimis import CimisClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cimis"

DATA_ITEMS = [
    "hly-air-tmp",
    "hly-wind-spd",
    "hly-wind-dir",
    "hly-rel-hum",
    "hly-sol-rad",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download CIMIS hourly weather observations"
    )

    parser.add_argument(
        "--station",
        type=int,
        default=39,
        help="CIMIS station number (default: 39, Parlier)",
    )

    parser.add_argument(
        "--start",
        default="2026-11-01",
        help="Start date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        default="2026-11-07",
        help="End date: YYYY-MM-DD",
    )

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("CIMIS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "CIMIS_API_KEY is missing from .env"
        )

    output_dir = DATA_DIR / f"station_{args.station}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"hourly_{args.start}_{args.end}.csv"
    )

    client = CimisClient(app_key=api_key)

    print(f"Station: {args.station}")
    print(f"Period: {args.start} to {args.end}")

    weather_data = client.get_hourly_data(
        targets=[args.station],
        start_date=args.start,
        end_date=args.end,
        data_items=DATA_ITEMS,
        unit_of_measure="Metric",
    )

    records = weather_data.get_all_records()

    if not records:
        raise RuntimeError(
            "No weather records returned for this request"
        )

    client.export_to_csv(
        weather_data,
        filename=output_file,
    )

    print(f"Downloaded {len(records)} records")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
