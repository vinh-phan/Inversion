"""Aggregate spatially interpolated ERA5 and EAC4 to Dutch municipality-local-days.

Use Run Cell in VS Code: load settings/functions, then run the final cell.
Daily files retain the spatial pipeline's wide columns and monthly layout.

Created with Codex
"""

# %% Settings and functions
import json
import sys
from pathlib import Path

# --------------------------- Settings ---------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parents[1]
CONCAT_ONLY = False  # True: reuse existing daily files; only concatenate.
SKIP_CONCAT = False  # True: aggregate only; step 4 can use monthly files directly.
# Mirror 1_spatial_interpolation.py; paths remain private to each user.
sys.path.append(str(SCRIPT_FOLDER / "utilities"))
from aggregate import aggregate_files
from parquet_pipeline import concatenate_parquet_files


CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
with CONFIG_PATH.open() as f:
    config = json.load(f)

# --------------------------- Main -------------------------------------
def main():
    if CONCAT_ONLY and SKIP_CONCAT:
        raise ValueError("CONCAT_ONLY and SKIP_CONCAT cannot both be True.")
    # Mirror the spatial pipeline's dataset prefixes and monthly file layout.

    # no inversion/dawn processing is triggered when t_dif columns are absent.
    datasets = [
        ("ERA5", "municipality_*.parquet", "municipality"),
        ("EAC4", "EAC4_municipality_*.parquet", "EAC4_municipality"),
    ]
    for dataset, file_pattern, output_prefix in datasets:
        print(f"Processing {dataset}", flush=True)
        if not CONCAT_ONLY:
            aggregate_files(
                input_dir=config[f"{dataset}_municipality_hourly"],
                output_dir=config[f"{dataset}_municipality_daily"],
                file_pattern=file_pattern,
                output_prefix=output_prefix,
                dawn_max_minutes=config.get("dawn_max_minutes", 60),
                output_significant_figures=config.get("output_significant_figures"),
                output_float32=config.get("output_float32", True),
            )

        if SKIP_CONCAT:
            continue
        #hourly_dir = Path(config[f"{dataset}_municipality_hourly"])
        #hourly_out = Path(config[f"{dataset}_processed"]) / f"{dataset}_municipality_hourly_all.parquet"
        # Exclude UTC-named columns from combined outputs only. Monthly files
        # retain UTC timestamps and inversion variants for aggregation/reuse.
        #concatenate_parquet_files(hourly_dir, hourly_out, file_pattern=file_pattern,
        #                          output_float32=config.get("output_float32", True),
        #                          exclude_columns_containing=("utc",))

        daily_dir = Path(config[f"{dataset}_municipality_daily"])
        daily_out = Path(config[f"{dataset}_processed"]) / f"{dataset}_municipality_daily_all.parquet"
        concatenate_parquet_files(daily_dir, daily_out, file_pattern=file_pattern,
                                  output_float32=config.get("output_float32", True),
                                  exclude_columns_containing=("utc", "grid_area_coverage"))
        # Concatenation automatically takes out n_obs column and columns ending with _n (indicating the number of observations available for daily aggregation). Then depending on the user, will 
        # also remove any columns containing certain patterns in exclude_columns_containing argument. 
        # No dawn observations for 2026-01-01 (so 343 missing obs for the concatenated daily files - 343 gemeente)

# %% Run aggregation / concatenation
if __name__ == "__main__":
    main()
