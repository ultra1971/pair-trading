"""
parse_data.py
=============
Converts raw CRSP data (tetsing_data.csv) into per-ticker CSV files that
qstrader's YahooDailyCsvBarPriceHandler can consume.

Usage
-----
    python parse_data.py                          # use defaults from config.yaml
    python parse_data.py --input my_data.csv      # override input path
    python parse_data.py --input my_data.csv --output-dir data/

Output format per ticker
------------------------
    data/<PERMNO>.csv
    Date,Open,High,Low,Close,Adj Close,Volume
    2016/01/04,price,price,price,price,price,volume
    ...
"""

import csv
import os
import sys
import logging
import argparse

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

CSV_HEADER = "Date,Open,High,Low,Close,Adj Close,Volume\n"

# Canonical active pairs from config — deduplicated, ordered.
DEFAULT_PAIRS = [
    "43350", "82651",
    "44644", "90458",
    "24969", "24985",
    "42585", "83621",
    "60186", "81095",
    "16548", "81577",
]


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    default_input = cfg.get("data", {}).get("testing_data_path", "tetsing_data.csv")
    default_outdir = cfg.get("data", {}).get("parsed_data_dir", "data/")

    parser = argparse.ArgumentParser(description="Parse raw CRSP CSV into per-ticker files.")
    parser.add_argument(
        "--input", default=default_input,
        help=f"Path to raw CRSP CSV file (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir", default=default_outdir,
        help=f"Directory to write per-ticker CSVs (default: {default_outdir})",
    )
    return parser.parse_args()


def build_permno_set(cfg: dict) -> set[str]:
    """Derive the set of PERMNOs to extract from config active_pairs."""
    pairs = cfg.get("pair_selection", {}).get("active_pairs", None)
    if pairs:
        return {p for pair in pairs for p in pair}
    return set(DEFAULT_PAIRS)


def format_date(raw: str) -> str:
    """Convert YYYYMMDD → YYYY/MM/DD."""
    if len(raw) != 8:
        raise ValueError(f"Unexpected date format: {raw!r}")
    return f"{raw[:4]}/{raw[4:6]}/{raw[6:8]}"


def main() -> None:
    args = parse_args()
    cfg = load_config()
    permno_set = build_permno_set(cfg)

    input_path = args.input
    output_dir = args.output_dir

    if not os.path.exists(input_path):
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    open_files: dict[str, object] = {}
    rows_written = 0
    rows_skipped = 0

    try:
        with open(input_path, newline="") as src:
            reader = csv.reader(src)
            for line_num, row in enumerate(reader, start=1):
                if len(row) < 13:
                    rows_skipped += 1
                    continue

                permno = row[1].strip()
                if permno not in permno_set:
                    continue

                try:
                    date_str = format_date(row[2].strip())
                    price = row[6].strip()
                    volume = row[7].strip()
                except (ValueError, IndexError) as exc:
                    log.warning("Line %d skipped (%s): %s", line_num, exc, row)
                    rows_skipped += 1
                    continue

                if permno not in open_files:
                    dest_path = os.path.join(output_dir, f"{permno}.csv")
                    fh = open(dest_path, "w")
                    fh.write(CSV_HEADER)
                    open_files[permno] = fh
                    log.info("Created %s", dest_path)

                data_line = f"{date_str},{price},{price},{price},{price},{price},{volume}\n"
                open_files[permno].write(data_line)
                rows_written += 1

    finally:
        for fh in open_files.values():
            fh.close()

    log.info(
        "Done. %d rows written, %d skipped. Files: %s",
        rows_written, rows_skipped, list(open_files.keys()),
    )


if __name__ == "__main__":
    main()
