"""Aggregate ERA5 weather and EAC4 pollution data to Dutch municipalities.

Run: python3 code/1_process/1_spatial_interpolation.py

Created with Codex
"""

import json
import sys
from pathlib import Path

# --------------------------- Settings ---------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parents[1]
# Find shared utilities regardless of the directory used to run this script.
sys.path.append(str(SCRIPT_FOLDER / "utilities"))
from spatial_interpolation import process_files

CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
with CONFIG_PATH.open() as f:
    config = json.load(f)


# --------------------------- Main -------------------------------------
def main():
    # Each dataset uses its own input pattern and output filename prefix.
    datasets = [
        ("ERA5", "ERA5_extracted_*.parquet", "municipality"),
        ("EAC4", "EAC4.parquet", "EAC4_municipality"),
    ]
    for dataset, file_pattern, output_prefix in datasets:
        print(f"Processing {dataset}", flush=True)
        process_files(
            input_dir=config[f"{dataset}_extracted"],
            output_dir=Path(config[f"{dataset}_processed"]) / "municipality_hourly",
            boundaries_path=config["municipality_boundaries"],
            file_pattern=file_pattern,
            output_prefix=output_prefix,
            include_area=True,
            output_float32=config.get("output_float32", True),
        )


if __name__ == "__main__":
    main()
