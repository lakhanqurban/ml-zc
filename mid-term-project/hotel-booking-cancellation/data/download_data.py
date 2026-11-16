"""Download the raw hotel bookings dataset.

The dataset is the clean TidyTuesday version of "Hotel booking demand datasets"
(Antonio, Almeida & Nunes, 2019). It is mirrored on GitHub so that the project
runs without any Kaggle credentials.

Usage
-----
    python data/download_data.py            # download to data/raw/hotel_bookings.csv
    python data/download_data.py --force    # re-download even if the file exists
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
TARGET_FILE = RAW_DIR / "hotel_bookings.csv"

EXPECTED_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
EXPECTED_ROWS = 119_390

# Ordered by preference. The first two are byte-identical mirrors.
MIRRORS = [
    (
        "https://raw.githubusercontent.com/datacamp/"
        "Machine-Learning-With-XGboost-live-training/master/data/hotel_bookings.csv"
    ),
    (
        "https://raw.githubusercontent.com/salves94/"
        "hotel-exploratory-data-analysis/master/hotel_bookings.csv"
    ),
    (
        "https://raw.githubusercontent.com/mpolinowski/"
        "hotel-booking-dataset/master/datasets/hotel_bookings.csv"
    ),
]

MANUAL_INSTRUCTIONS = """
Could not download the dataset automatically.

Download it manually and place it at:
    data/raw/hotel_bookings.csv

Sources (any of these works):
  * Kaggle  : https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
  * UCI     : https://doi.org/10.1016/j.dib.2019.100807
  * GitHub  : https://github.com/datacamp/Machine-Learning-With-XGboost-live-training

The file must contain 119,390 rows and have SHA-256
{expected}
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not args.force:
        digest = sha256_of(TARGET_FILE)
        size_mb = TARGET_FILE.stat().st_size / 1e6
        print(f"Already present: {TARGET_FILE} ({size_mb:.1f} MB, sha256={digest[:16]}...)")
        if digest != EXPECTED_SHA256:
            print(
                "WARNING: checksum differs from the expected release. "
                "Delete the file and re-run with --force if this is unexpected."
            )
        return

    errors = []
    for url in MIRRORS:
        print(f"downloading {url}")
        try:
            tmp = TARGET_FILE.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response:
                with tmp.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp.replace(TARGET_FILE)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            TARGET_FILE.with_suffix(".part").unlink(missing_ok=True)
            continue

        digest = sha256_of(TARGET_FILE)
        rows = sum(1 for _ in TARGET_FILE.open("rb")) - 1
        size_mb = TARGET_FILE.stat().st_size / 1e6
        print(f"saved {TARGET_FILE} ({size_mb:.1f} MB, {rows:,} rows)")
        print(f"sha256 {digest}")

        if digest == EXPECTED_SHA256:
            print("checksum OK")
            return
        if rows == EXPECTED_ROWS:
            print(
                "checksum differs but the row count matches the published release "
                "(line-ending normalisation is the usual cause)."
            )
            return
        print("unexpected checksum and row count, trying the next mirror")

    print("\n".join(errors))
    raise SystemExit(MANUAL_INSTRUCTIONS.format(expected=EXPECTED_SHA256))


if __name__ == "__main__":
    main()
