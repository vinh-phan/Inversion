"""Aggregate spatially interpolated ERA5 to Dutch municipality-local-days.

Run: python3 code/1_process/2_aggregation.py
Daily files retain the spatial pipeline's wide columns and monthly layout.
"""

import json
import sys
from pathlib import Path

# --------------------------- Settings ---------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parents[1]
# Mirror 1_spatial_interpolation.py; paths remain private to each user.
sys.path.append(str(SCRIPT_FOLDER / "utilities"))
from aggregate import aggregate_files

CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
with CONFIG_PATH.open() as f:
    config = json.load(f)


# --------------------------- Main -------------------------------------
def main():
    aggregate_files(
        input_dir=config["ERA5_municipality_hourly"],
        output_dir=config["ERA5_municipality_daily"],
        file_pattern="municipality_*.parquet",
        output_prefix="municipality",
        dawn_max_minutes=config.get("dawn_max_minutes", 60),
    )


if __name__ == "__main__":
    main()
