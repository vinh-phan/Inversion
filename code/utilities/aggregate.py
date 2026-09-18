"""Temporal aggregation of municipality weather; legacy pollution helpers below."""

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
    pipeline. Aware input is converted to UTC without changing the instant.
    Keeping both aware timestamps avoids ambiguous autumn wall-clock times:
    the repeated 02:00 hours remain distinct, while 03:00 occurs only once.
    Polars uses the timezone database, including historical CET/CEST changes;
    no fixed offsets or hand-written DST calendar are used here.
    Works with both DataFrame and LazyFrame.
    """
    schema = df.collect_schema()
    if "time" not in schema or not isinstance(schema["time"], pl.Datetime):
        raise ValueError("Expected a Datetime 'time' column containing UTC observations.")
    time = pl.col("time")
    utc = (time.dt.replace_time_zone("UTC") if schema["time"].time_zone is None
           else time.dt.convert_time_zone("UTC"))
    return (df.with_columns(utc.alias("time_utc"))
            .with_columns(pl.col("time_utc").dt.convert_time_zone(LOCAL_TIMEZONE).alias("time_local"))
            .with_columns(pl.col("time_local").dt.date().alias("date")))


def _variable_name(column):
    """Separate existing `<variable>_<spatial method>_mean` column names.

    Methods are deliberately unrestricted (including names with underscores),
    so new spatial methods use exactly the same temporal aggregation rules.
    Unsuffixed variables are also accepted by the generic daily helper.
    """
    pattern = (r"^(t_dif(?:_\d+_\d+)?(?:_indicator|_truncated)?"
               r"|wd_\d+_\d+_indicator|[A-Za-z][A-Za-z0-9]*(?:_squared|_cubed)?)"
               r"(?:_(.+)_mean)?$")
    match = re.fullmatch(pattern, column)
    return match.group(1) if match else column


def _output_base(column):
    # The terminal 'mean' describes spatial averaging in hourly files. Daily
    # files retain the method and replace that suffix with the temporal rule.
    return column[:-5] if column.endswith("_mean") else column


def weather_expressions(columns, precipitation_variables=("tp",)):
    """Mean available weather values, with separate precipitation sum/mean.

    Precipitation is assumed to contain interval amounts, as supplied by the
    existing ERA5 extraction. We neither deaccumulate nor infer missing amounts.
    An entirely missing series has a missing sum, not a misleading zero.
    Wind direction is stored in radians in this project: average unit vectors
    around the circle and mark cancelling vectors missing. Its polynomial
    transforms have no circular interpretation and are intentionally omitted.
    U/V and existing wind-sector indicators still receive arithmetic means.
    """
    expressions = []
    for column in columns:
        variable, base = _variable_name(column), _output_base(column)
        if variable.startswith("t_dif"):
            continue
        if variable in {"wd_squared", "wd_cubed"}:
            continue
        values = pl.col(column)
        if variable == "wd":
            sine, cosine = values.sin().mean(), values.cos().mean()
            expressions.append(
                pl.when((sine**2 + cosine**2).sqrt() > 1e-12)
                .then(pl.arctan2(sine, cosine) % (2 * math.pi))
                .alias(f"{base}_mean")
            )
        else:
            expressions.append(values.mean().alias(f"{base}_mean"))
        if variable in precipitation_variables:
            expressions.append(pl.when(values.count() > 0).then(values.sum()).alias(f"{base}_sum"))
        expressions.append(values.count().alias(f"{base}_n"))
    return expressions


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
    variants retain those shares. Multi-observation variants report whether
    *any* available share is positive. Strength includes all nonmissing values,
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
                value = (pl.when(values.count() > 0).then((values > 0).any().cast(pl.Int8))
                         if indicator else values.mean())
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
    return df.group_by(keys).agg(expressions).sort(keys)


def aggregate_files(input_dir, output_dir, file_pattern="municipality_*.parquet",
                    output_prefix="municipality", dawn_max_minutes=60):
    """Keep the existing wide, monthly Parquet layout, partitioned by LOCAL month.

    Scan all file timestamps to discover local months, then push each month's
    UTC interval into the scan. This includes adjacent UTC-file observations at
    midnight/month/year boundaries without loading the whole archive at once.
    Partial first/last days are retained and reported through n_obs.
    """
    files = sorted(Path(input_dir).glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {input_dir}")
    if Path(input_dir).resolve() == Path(output_dir).resolve():
        raise ValueError("Daily output directory must differ from hourly input directory.")
    source = pl.scan_parquet(files)
    prepared = prepare_time(source)
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
        # prune files; bounds represent local midnights expressed in UTC.
        bounds = [value.astimezone(ZoneInfo("UTC")) for value in (start, end)]
        if source.collect_schema()["time"].time_zone is None:
            bounds = [value.replace(tzinfo=None) for value in bounds]
        frame = source.filter((pl.col("time") >= bounds[0]) & (pl.col("time") < bounds[1])).collect()
        output = aggregate_daily(frame, dawn_max_minutes=dawn_max_minutes)
        target = output_dir / f"{output_prefix}_{month:%Y_%m}.parquet"
        output.write_parquet(target)
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
