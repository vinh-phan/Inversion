"""
Temporal aggregation of municipality weather; legacy pollution helpers below.

Created with Codex. 
"""

from datetime import datetime
from pathlib import Path
import math
import re
import warnings
from zoneinfo import ZoneInfo

import polars as pl


LOCAL_TIMEZONE = "Europe/Amsterdam"
METADATA = ["gemeentenaam", "centroid_lat", "centroid_lon", "grid_area_coverage"]
INVERSION_VARIANTS = [
    "03_local", "03_utc", "06_local", "06_utc", "dawn", "all", "6h_local", "6h_utc"
]


def prepare_time(df):
    """Retain UTC instants and derive local dates *after* IANA conversion.

    Naive input `time` is explicitly interpreted as UTC, as in the extraction
    pipeline. Hourly files use aware local `time` plus `time_utc`. 
    Keeping both aware timestamps avoids ambiguous autumn wall-clock times:
    the repeated 02:00 hours remain distinct, while 03:00 occurs only once.
    Polars uses the timezone database, including historical CET/CEST changes;
    no fixed offsets or hand-written DST calendar are used here.
    Works with both DataFrame and LazyFrame.
    """
    schema = df.collect_schema()
    if "time" not in schema or not isinstance(schema["time"], pl.Datetime):
        raise ValueError("Expected Datetime 'time': naive UTC or timezone-aware observations.")
    time = pl.col("time")
    utc = (time.dt.replace_time_zone("UTC") if schema["time"].time_zone is None
           else time.dt.convert_time_zone("UTC"))
    return (df.with_columns(utc.alias("time_utc"))
            .with_columns(pl.col("time_utc").dt.convert_time_zone(LOCAL_TIMEZONE).alias("time_local"))
            .with_columns(pl.col("time_local").dt.date().alias("date")))


def _variable_name(column):
    """Recognize `<variable>_<method>`.

    Methods are deliberately unrestricted (including names with underscores),
    so new spatial methods use exactly the same temporal aggregation rules.
    Unsuffixed variables are also accepted by the generic daily helper.
    """
    pattern = (r"^(t_dif(?:_\d+_(?:\d+|2m))?(?:_indicator|_truncated)?"
               r"|wd_\d+_\d+_indicator|[A-Za-z][A-Za-z0-9]*(?:_squared|_cubed)?)"
               r"(?:_(.+))?$")
    match = re.fullmatch(pattern, _output_base(column))
    return match.group(1) if match else column


def _output_base(column):
    # Remove the spatial 'mean' suffix from legacy hourly files only. New
    # hourly names already end with the method; daily names add a temporal rule.
    return column[:-5] if column.endswith("_mean") else column


def weather_expressions(columns, precipitation_variables=("tp",)):
    """Mean available weather values, with separate precipitation sum/mean.

    Precipitation is assumed to contain interval amounts, as supplied by the
    existing ERA5 extraction. We neither deaccumulate nor infer missing amounts.
    An entirely missing series has a missing sum, not a misleading zero.
    """
    expressions = []
    for column in columns:
        variable, base = _variable_name(column), _output_base(column)
        if variable.startswith("t_dif"):
            continue
        if variable == "wd" or variable.startswith("wd_"):
            continue
        values = pl.col(column)
        expressions.append(values.mean().alias(f"{base}_mean"))
        if variable in precipitation_variables:
            expressions.append(pl.when(values.count() > 0).then(values.sum()).alias(f"{base}_sum"))
        expressions.append(values.count().alias(f"{base}_n"))
    return expressions


def add_wind_direction(df):
    """Derive meteorological wind direction from each method's daily mean U/V.

    Retain the project's radians convention: 0 = from north, pi/2 = from east,
    increasing clockwise. This is the direction of the mean wind vector, not
    the mean of hourly directions. Each component retains its own available-
    observation mean/count. If either mean is absent/nonfinite or the resultant
    speed is <= 1e-12 m/s (including numerical cancellation), direction and all
    bins are null. Four half-open sectors cover [0, 2*pi) without overlap.
    The familiar _mean suffix is retained for compatibility with daily columns;
    the new sector columns contain binary Int8 dummies, never hourly shares.
    """
    directions = []
    bins = []
    for column in df.columns:
        if not re.fullmatch(r"u10(?:_.+)?_mean", column):
            continue
        suffix = column[3:]  # _<method>_mean, or _mean for unsuffixed input.
        v_column = f"v10{suffix}"
        if v_column not in df.columns:
            raise ValueError(f"Wind direction requires {v_column} alongside {column}.")
        u, v = pl.col(column), pl.col(v_column)
        direction = f"wd{suffix}"
        valid = u.is_finite() & v.is_finite() & ((u**2 + v**2).sqrt() > 1e-12)
        directions.append(pl.when(valid)
                          .then((3 * math.pi / 2 - pl.arctan2(v, u)) % (2 * math.pi))
                          .alias(direction))
        for start in [0, 90, 180, 270]:
            bins.append(((pl.col(direction) >= math.radians(start))
                         & (pl.col(direction) < math.radians(start + 90)))
                        .cast(pl.Int8).alias(f"wd_{start}_{start + 90}_indicator{suffix}"))
    return df.with_columns(directions).with_columns(bins)


def compact_daily(df, significant_figures=None, float32=True):
    """Reduce output size only AFTER all means, thresholds, and bins are computed.

    Significant figures preserve small precipitation amounts and their powers;
    fixed decimal places could turn them into zeros. Six figures bound decimal
    rounding error to about 5e-6 relative (0.0005%). Float32 adds approximately
    6e-8 relative error for normal values and halves floating-point storage.
    Retain Float64 for columns outside Float32's normal range to avoid overflow
    or loss of relative precision in subnormal values. Metadata, diagnostics,
    integers, and wind angles are untouched; keeping angles at full precision
    preserves their [0, 2*pi) range and agreement with exact sector boundaries.
    Set significant_figures=None and float32=False for full-precision output.
    """
    if significant_figures is not None and (isinstance(significant_figures, bool)
            or not isinstance(significant_figures, int) or not 1 <= significant_figures <= 15):
        raise ValueError("output_significant_figures must be null or an integer from 1 to 15.")
    if not isinstance(float32, bool):
        raise ValueError("output_float32 must be true or false.")
    columns = [name for name, dtype in df.schema.items()
               if dtype.is_float() and name not in METADATA + ["dawn_offset_minutes"]
               and not name.startswith("wd_")]
    if significant_figures is not None:
        df = df.with_columns([pl.col(name).round_sig_figs(significant_figures) for name in columns])
    if float32 and columns:
        # Check ranges together rather than scanning the frame once per column.
        ranges = df.select([
            expression for name in columns for expression in (
                pl.col(name).abs().filter(pl.col(name).is_finite() & (pl.col(name) != 0))
                .min().alias(f"{name}_min"),
                pl.col(name).abs().filter(pl.col(name).is_finite()).max().alias(f"{name}_max"),
            )
        ]).row(0, named=True)
        safe = [name for name in columns
                if (ranges[f"{name}_min"] is None or ranges[f"{name}_min"] >= 1.1754943508222875e-38)
                and (ranges[f"{name}_max"] is None or ranges[f"{name}_max"] <= 3.4028234663852886e38)]
        df = df.with_columns([pl.col(name).cast(pl.Float32) for name in safe])
    return df


def time_selection(variant):
    """Select exact clock times; never round an unavailable hour into existence.

    Exact minute/second checks also permit sub-hourly input. Six-hour selection
    is circular clock membership (00/06/12/18), not an elapsed six-hour resample:
    DST changes the UTC spacing of local selections. UTC selections are still
    grouped by the corresponding Dutch local date.
    """
    if variant == "all":
        return pl.lit(True)
    if variant == "dawn":
        return pl.col("time_utc") == pl.col("dawn_observation_utc")
    clock = pl.col("time_local" if variant.endswith("local") else "time_utc")
    hours = [0, 6, 12, 18] if variant.startswith("6h") else [int(variant[:2])]
    return (clock.dt.hour().is_in(hours) & (clock.dt.minute() == 0)
            & (clock.dt.second() == 0) & (clock.dt.nanosecond() == 0))


def inversion_expressions(columns):
    """Preserve every existing difference, truncated strength, and indicator.

    No inversion is recomputed from an interpolated temperature difference.
    Spatially averaged indicators can be fractional shares: exact-time/dawn
    variants retain those shares. Multi-observation variants average available
    shares. Strength includes all nonmissing values,
    including zeros (and negative values in the existing raw differences).
    Empty/all-null selections remain null, distinguished from observed zero;
    per-variable counts make that distinction auditable.
    """
    expressions = []
    for column in columns:
        if not _variable_name(column).startswith("t_dif"):
            continue
        indicator = _variable_name(column).endswith("_indicator")
        base = _output_base(column)
        for variant in INVERSION_VARIANTS:
            values = pl.col(column).filter(time_selection(variant))
            if variant in {"all", "6h_local", "6h_utc"}:
                value = values.mean()
            else:
                value = values.first()
            expressions.extend([
                value.alias(f"{base}_{variant}"),
                values.count().alias(f"{base}_{variant}_n"),
            ])
    return expressions


def match_dawn(df, keys, max_minutes=60):
    """Calculate civil dawn at each municipality centroid and local date.

    Astral's standard civil depression is 6 degrees, at sea-level elevation.
    Choose the closest actual timestamp on the SAME local date; ties choose
    the earlier UTC instant. Distances are elapsed UTC minutes, never naive
    wall-clock differences across a DST jump. Matching is timestamp-based:
    a missing value at the chosen time is not replaced by a more distant one.
    Retain the nearest distance even when it exceeds the configured tolerance,
    but clear the chosen timestamp so all dawn measures become missing.
    """
    from astral import Observer
    from astral.sun import dawn

    if not math.isfinite(max_minutes) or max_minutes < 0:
        raise ValueError("dawn_max_minutes must be finite and nonnegative.")
    coordinates = ["centroid_lat", "centroid_lon"]
    if not set(coordinates).issubset(df.columns):
        raise ValueError("Civil dawn requires centroid_lat and centroid_lon (WGS84).")
    days = df.select(keys + coordinates).unique()
    if days.select(keys).is_duplicated().any():
        raise ValueError("Municipality centroid changes within a local day.")
    dawns = []
    for row in days.iter_rows(named=True):
        lat, lon = row["centroid_lat"], row["centroid_lon"]
        if (lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon)
                or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            raise ValueError("Invalid municipality centroid latitude/longitude.")
        try:
            dawns.append(dawn(Observer(lat, lon), date=row["date"], depression=6,
                              tzinfo=LOCAL_TIMEZONE))
        except ValueError:
            # Useful for reuse outside the Netherlands, where polar days may
            # have no civil dawn. Other dates/municipalities remain usable.
            dawns.append(None)
    if any(value is None for value in dawns):
        warnings.warn("Civil dawn does not occur for some municipality-days.")
    days = days.select(keys).with_columns(
        pl.Series("dawn_local", dawns, dtype=pl.Datetime("us", LOCAL_TIMEZONE))
    )
    nearest = (df.select(keys + ["time_utc"]).join(days, on=keys)
               .with_columns(((pl.col("time_utc") - pl.col("dawn_local").dt.convert_time_zone("UTC"))
                              .dt.total_microseconds().abs() / 60_000_000).alias("dawn_offset_minutes"))
               .sort(keys + ["dawn_offset_minutes", "time_utc"], nulls_last=True)
               .group_by(keys, maintain_order=True).agg(
                   pl.col("dawn_local").first(),
                   pl.col("dawn_offset_minutes").first(),
                   pl.col("time_utc").first().alias("dawn_observation_utc"),
               ).with_columns(
                   pl.when(pl.col("dawn_offset_minutes") <= max_minutes)
                   .then(pl.col("dawn_observation_utc")).alias("dawn_observation_utc")
               ))
    return nearest


def aggregate_daily(df, group_columns=("gemeentecode",), dawn_max_minutes=60,
                    precipitation_variables=("tp",)):
    """Aggregate any observation frequency to municipality × Dutch local date.

    All numeric measurement columns and all existing spatial suffixes are
    processed together. Extra identifiers (e.g. a long-format method column)
    can be supplied in group_columns. No complete-day requirement is imposed.
    NaN becomes null so counts/means consistently use observed values only.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect(engine="streaming")
    if df.is_empty():
        raise ValueError("No observations to aggregate.")
    if not set(group_columns).issubset(df.columns):
        raise ValueError(f"Missing grouping columns: {group_columns}")
    columns = [name for name, dtype in df.schema.items()
               if dtype.is_numeric() and name not in set(group_columns) | set(METADATA)]
    df = prepare_time(df).with_columns([
        pl.col(name).fill_nan(None) for name in columns if df.schema[name].is_float()
    ])
    keys = list(group_columns) + ["date"]
    if any(df[name].null_count() for name in list(group_columns) + ["time_utc"]):
        raise ValueError("Null municipality identifier or timestamp.")
    if df.select(list(group_columns) + ["time_utc"]).is_duplicated().any():
        raise ValueError("Duplicate municipality timestamp; check overlapping input files.")
    if df.filter(time_selection("03_local")).select(keys).is_duplicated().any():
        raise ValueError("Duplicate 03:00 local observation for a municipality-date.")
    metadata = [name for name in METADATA if name in df.columns and name not in group_columns]
    if df.select(list(group_columns) + metadata).unique().select(list(group_columns)).is_duplicated().any():
        raise ValueError("Municipality metadata changes within the input batch.")
    if any(_variable_name(name).startswith("t_dif") for name in columns):
        dawns = match_dawn(df, keys, dawn_max_minutes)
        df = df.join(dawns, on=keys)
        metadata += ["dawn_local", "dawn_offset_minutes", "dawn_observation_utc"]
    expressions = [pl.len().alias("n_obs")]
    expressions += [pl.col(name).first() for name in metadata]
    expressions += weather_expressions(columns, precipitation_variables)
    expressions += inversion_expressions(columns)
    return add_wind_direction(df.group_by(keys).agg(expressions).sort(keys))


def aggregate_files(input_dir, output_dir, file_pattern="municipality_*.parquet",
                    output_prefix="municipality", dawn_max_minutes=60,
                    output_significant_figures=None, output_float32=True):
    """Keep the existing wide, monthly Parquet layout, partitioned by LOCAL month.

    Scan all file timestamps to discover local months, then push each month's
    UTC interval into the scan. This includes adjacent UTC-file observations at
    midnight/month/year boundaries without loading the whole archive at once.
    Partial first/last days are retained and reported through n_obs.
    """
    files = sorted(p for p in Path(input_dir).glob(file_pattern)
                   if p.is_file() and not p.name.startswith(".")
                   and not p.name.endswith("_all.parquet"))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {input_dir}")
    if Path(input_dir).resolve() == Path(output_dir).resolve():
        raise ValueError("Daily output directory must differ from hourly input directory.")
    source = pl.scan_parquet(files)
    prepared = prepare_time(source)
    source_timezone = source.collect_schema()["time"].time_zone
    months = (prepared.select(pl.col("time_local").dt.date().dt.truncate("1mo").alias("month"))
              .unique().sort("month").collect())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for month in months["month"]:
        if month is None:
            raise ValueError("Null input timestamp.")
        start = datetime(month.year, month.month, 1, tzinfo=ZoneInfo(LOCAL_TIMEZONE))
        end = datetime(month.year + (month.month == 12), month.month % 12 + 1, 1,
                       tzinfo=ZoneInfo(LOCAL_TIMEZONE))
        # Filter the original timestamp so Parquet row-group statistics can
        # prune files. Match the source column's timezone: legacy naive UTC
        # files and new aware local files must select exactly the same instants.
        bounds = [value.astimezone(ZoneInfo(source_timezone or "UTC"))
                  for value in (start, end)]
        if source_timezone is None:
            bounds = [value.replace(tzinfo=None) for value in bounds]
        frame = source.filter((pl.col("time") >= bounds[0]) & (pl.col("time") < bounds[1])).collect()
        output = aggregate_daily(frame, dawn_max_minutes=dawn_max_minutes)
        target = output_dir / f"{output_prefix}_{month:%Y_%m}.parquet"
        output = compact_daily(output, output_significant_figures, output_float32)
        output.write_parquet(target, compression="zstd")
        print(f"{target.name}: {output.height} municipality-days; "
              f"observations/day {output['n_obs'].min()}–{output['n_obs'].max()}; "
              f"{output.width} columns", flush=True)


#####################################################
################## Aggregating the dataset
def agg_data(df, type):
    if type == "weather":
        return aggregate_daily(df)
    elif type == "pollution_EAC4":
        # Change units of variables 
        df = df.with_columns(
            (pl.col("pm1") * 1000000000).alias("pm1"),
            (pl.col("pm2p5") * 1000000000).alias("pm2p5"),
            (pl.col("pm10") * 1000000000).alias("pm10") 
        )

        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.day().alias("day"),
            pl.col("time").dt.weekday().alias("weekday"),
            pl.col("time").dt.quarter().alias("quarter"),   
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.year().alias("year")
        )

        #========= Aggregate data to the daily level =========
        # Only the PM variables for now - have to check whether it makes sense to look at the other variables, which are in "Total Column")
        print("Aggregating to the daily level...") 
        variables = ['pm1', 'pm2p5', 'pm10' ]
        
        agg_exprs_daily = [pl.col(var).mean().alias(f"{var}_mean") for var in variables]

        df_agg = df.group_by(["lon", "lat", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)

        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("lon", "lat", "year", "month", "day") 
        
        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=7)
                .over("lon", "lat")
                .alias(f"{var}_7day_mean")
            )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("lon", "lat", "year", "month", "day") 

        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=30)
                .over("lon", "lat")
                .alias(f"{var}_30day_mean")
            )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)
                
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        # Return the data 
        df_agg = df_agg.collect(engine = "streaming")

        print("Datasets aggregated.")
        print(f"Unique geographical locations (lon and lat): {df_agg.select(["lon", "lat"]).unique().height}") 
        
        return df_agg
    
    elif type == "pollution_NO2":
        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.day().alias("day"),
            pl.col("time").dt.weekday().alias("weekday"),
            pl.col("time").dt.quarter().alias("quarter"),   
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.year().alias("year")
        )
        
        #========= Aggregate data to the daily level =========
        print("Aggregating to the daily level...") 

        agg_exprs_daily = [pl.col("no2").mean().alias("no2_mean")]

        df_agg = df.group_by(["site_name", "lat", "lon", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)
     
        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("site_name", "year", "month", "day") 
        
        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("no2_mean")
                .rolling_mean(window_size=7)
                .over("site_name")
                .alias(f"no2_7day_mean"))

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("site_name", "year", "month", "day") 

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("no2_mean")
                .rolling_mean(window_size=30)
                .over("site_name")
                .alias("no2_30day_mean"))
            
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        print("Datasets aggregated.")
        return df_agg
        
    elif type == "DEFRA_PM2.5":
        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("date").dt.hour().alias("hour"),
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.quarter().alias("quarter"),   
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year")
        )
        
        #========= Aggregate data to the daily level =========
        print("Aggregating to the daily level...") 

        agg_exprs_daily = [pl.col("pollution_value").mean().alias("DEFRA_pm2p5_mean")]

        df_agg = df.group_by(["site_name", "latitude", "longitude", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)
     
        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("site_name", "year", "month", "day") 
        
        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("DEFRA_pm2p5_mean")
                .rolling_mean(window_size=7)
                .over("site_name")
                .alias(f"DEFRA_pm2p5_7day_mean"))

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("site_name", "year", "month", "day") 

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("DEFRA_pm2p5_mean")
                .rolling_mean(window_size=30)
                .over("site_name")
                .alias("DEFRA_pm2p5_30day_mean"))
            
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        print("Datasets aggregated.")
        return df_agg
