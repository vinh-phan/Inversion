"""Audit LMN station observations and estimate daily municipality pollution.

Use Run Cell in VS Code: load settings/functions, then run the final cell.
Supporting downloads are retained in LMN/raw/metadata; refresh is opt-in.
"""

# %% Settings and functions
import json
import sys
from pathlib import Path


# --------------------------- Settings ---------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parents[1]
CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
REFRESH_METADATA = False  # True: refresh the supporting metadata downloads.
DOWNLOAD_ONLY = False  # True: download metadata without processing observations.
AUDIT_ONLY = False  # True: audit/extract sources without municipality processing.
SKIP_AUDIT = True  # True: reuse extracted files from an already completed audit.
sys.path.insert(0, str(SCRIPT_FOLDER / "utilities"))
from lmn import audit_sources, download_metadata, run
from parquet_pipeline import concatenate_parquet_files

with CONFIG_PATH.open() as f:
    config = json.load(f)

# --------------------------- Main -------------------------------------
def main():
    download_metadata(Path(config["LMN_raw"]) / "metadata", REFRESH_METADATA)
    if AUDIT_ONLY:
        audit_sources(config)
    elif not DOWNLOAD_ONLY:
        run(config, skip_audit=SKIP_AUDIT)

    if AUDIT_ONLY or DOWNLOAD_ONLY:
        return
    output_root = Path(config["LMN_processed"])
    concatenate_parquet_files(output_root / "municipality_daily", output_root / "municipality_daily_all.parquet",
                              output_float32=config.get("output_float32", True), 
                              exclude_columns_containing=("_n",)) #
    concatenate_parquet_files(output_root / "municipality_daily_all_stations",
                             output_root / "municipality_daily_all_stations_all.parquet",
                              output_float32=config.get("output_float32", True), 
                              exclude_columns_containing=("_n",))

# 1% of PM10 set to missing because there were less than 75% of 24 hours available for coputing the daily average. 
# PM2p5 also missing because of the same reason (in addition to being no data before 2014)

# %% Run LMN processing / concatenation
if __name__ == "__main__":
    main()
