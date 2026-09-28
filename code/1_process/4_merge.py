"""
This file merges ERA5, EAC4, and LMN data together 
"""

# %% Settings and functions
import json
from pathlib import Path
import polars as pl 

# --------------------------- Settings ---------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parents[1]
CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
with CONFIG_PATH.open() as f:
    config = json.load(f)

ERA5_dir = Path(config["ERA5_processed"])
EAC4_dir = Path(config["EAC4_processed"])
LMN_dir = Path(config["LMN_processed"])
output_dir = Path(config["merged"])

df_ERA5_daily = pl.scan_parquet(ERA5_dir / "ERA5_municipality_daily_all.parquet")
df_EAC4_daily = pl.scan_parquet(EAC4_dir / "EAC4_municipality_daily_all.parquet")
df_LMN_daily = pl.scan_parquet(LMN_dir / "municipality_daily_all.parquet")

# %% Merge files 
# --------------------- Add data source to air pollution columns ----------------
df_EAC4_daily = df_EAC4_daily.rename(
    lambda c: f"EAC4_{c}" if c.startswith("pm") else c
)

df_LMN_daily = df_LMN_daily.rename(
    lambda c: f"LMN_{c}" if c.startswith("pm") else c
)


def add_fixed_effects(df):
    """Add fixed-effect groups used downstream."""
    return df.with_columns(
        [
            pl.col("gemeentecode").cast(pl.String).alias("municipality"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.weekday().alias("dow"),
            pl.col("date").dt.day().alias("dom"),
        ]
    ).with_columns(
        [
            pl.concat_str([pl.col("year"), pl.col("month")], separator="-").alias(
                "year_month"
            ),
            pl.concat_str(
                [pl.col("municipality"), pl.col("year"), pl.col("month")],
                separator="-",
            ).alias("municipality_year_month"),
            pl.concat_str(
                [pl.col("municipality"), pl.col("month")], separator="-"
            ).alias("municipality_month"),
            pl.concat_str(
                [pl.col("municipality"), pl.col("year")], separator="-"
            ).alias("municipality_year"),
            pl.concat_str(
                [pl.col("year"), pl.col("month"), pl.col("dom")], separator="-"
            ).alias("year_month_dom"),
        ]
    )


# ---------------------- Daily files merge --------------------------
# Merge 
(
    df_ERA5_daily
    .join(
        df_EAC4_daily,
        on=["gemeentecode", "date", "gemeentenaam", "centroid_lat", "centroid_lon"],
        how="full",
        coalesce=True,
    )
    .join(
        df_LMN_daily,
        on=["gemeentecode", "date", "gemeentenaam", "centroid_lat", "centroid_lon"],
        how="full",
        coalesce=True,
    )
    .pipe(add_fixed_effects)
    .sink_parquet(output_dir / "df_daily.parquet")
)
 