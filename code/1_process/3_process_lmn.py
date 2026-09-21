"""Audit LMN station observations and estimate daily municipality pollution.

Run: python code/1_process/3_process_lmn.py
Supporting downloads are retained in LMN/raw/metadata; refresh is opt-in.
"""

import argparse
import json
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "utilities"))
from lmn import audit_sources, download_metadata, run
from parquet_pipeline import concatenate_parquet_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config/config.json")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Reuse extracted files from an already completed audit.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    download_metadata(Path(config["LMN_raw"]) / "metadata", args.refresh_metadata)
    if args.audit_only:
        audit_sources(config)
    elif not args.download_only:
        run(config, skip_audit=args.skip_audit)

    if args.audit_only or args.download_only:
        return
    output_root = Path(config["LMN_processed"])
    concatenate_parquet_files(output_root / "municipality_daily", output_root / "municipality_daily_all.parquet",
                              output_float32=config.get("output_float32", True))
    concatenate_parquet_files(output_root / "municipality_daily_all_stations",
                             output_root / "municipality_daily_all_stations_all.parquet",
                              output_float32=config.get("output_float32", True))


if __name__ == "__main__":
    main()
