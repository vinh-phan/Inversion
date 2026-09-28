"""LMN point-station processing, independent of the regular-grid pipeline.

Concentrations stay in µg/m³. Hourly donor availability is pollutant-specific.
Negative and extreme finite measurements are retained, with audit flags.
"""

import hashlib
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import certifi
import polars as pl
import pyarrow.parquet as pq

from spatial_interpolation import read_municipalities

LOCAL = ZoneInfo("Europe/Amsterdam")
UTC = timezone.utc
METHODS = ("idw", "idw_60km", "nearest")
POLLUTANTS = ("pm10", "pm2p5")
BASE_URL = "https://data.rivm.nl/data/luchtmeetnet/"
METADATA_FILES = (
    "luchtmeetnet_meetlocaties.csv", "luchtmeetnet_meetreeksen.csv",
    "luchtmeetnet_opst_meetreeksen.csv", "luchtmeetnet_meetopstellingen.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_metadata(folder, refresh=False):
    """Keep original bytes and provenance; never silently refresh a cache."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "downloads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in (*METADATA_FILES, "readme.pdf"):
        path = folder / name
        url = BASE_URL + ("" if name.endswith(".pdf") else "Metadata/") + name
        if path.exists() and not refresh:
            if name not in manifest or manifest[name]["sha256"] != sha256(path):
                raise ValueError(f"Unverifiable metadata cache: {path}. Use --refresh-metadata.")
            continue
        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
            content = response.read()
        if not content or (name.endswith(".pdf") and not content.startswith(b"%PDF")):
            raise ValueError(f"Unexpected download content: {url}")
        if name.endswith(".csv") and b"meet" not in content[:4000]:
            raise ValueError(f"Unexpected CSV download: {url}")
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(content)
        temp.replace(path)
        manifest[name] = {"url": url, "retrieved_at": datetime.now(UTC).isoformat(),
                          "sha256": sha256(path), "bytes": len(content)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_parquet(frame, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta.update({str(k).encode(): str(v).encode() for k, v in metadata.items()})
    table = table.replace_schema_metadata(meta)
    temp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(path)


def read_csv(path):
    return pl.read_csv(path, separator=";", comment_prefix="#", infer_schema=False)


def parse_time(column):
    return pl.col(column).str.to_datetime(
        format="%Y-%m-%dT%H:%M:%S%#z", time_zone="UTC", strict=False)


def deduplicate(frame):
    """Discard byte-equivalent observations only; never resolve disagreements."""
    unique = frame.unique(maintain_order=True)
    keys = ["meetlocatie_id", "component", "begindatumtijd", "einddatumtijd"]
    if unique.select(keys).is_duplicated().any():
        raise ValueError("Conflicting duplicate station/pollutant/interval observations.")
    return unique, len(frame) - len(unique)


def load_metadata(folder):
    sites = read_csv(folder / METADATA_FILES[0]).with_columns(
        pl.col("breedtegraad").cast(pl.Float64).alias("lat"),
        pl.col("lengtegraad").cast(pl.Float64).alias("lon"),
        parse_time("meetlocatie_begindatumtijd").alias("site_start"),
        parse_time("meetlocatie_einddatumtijd").alias("site_end"),
    ).select("meetlocatie_id", "lat", "lon", "meetlocatie_naam", "site_start", "site_end")
    series = read_csv(folder / METADATA_FILES[1]).filter(
        pl.col("component").is_in(["PM10", "PM2.5"]) & (pl.col("matrix") == "lucht")
    ).with_columns(
        parse_time("meetreeks_begindatumtijd").alias("series_start"),
        parse_time("meetreeks_einddatumtijd").alias("series_end"),
    ).select("meetreeks_id", "meetlocatie_id", "component", "matrix", "stationstype",
             "gebiedstype", "series_start", "series_end")
    if sites["meetlocatie_id"].is_duplicated().any():
        raise ValueError("Station coordinates have multiple records; resolve before processing.")
    if series["meetreeks_id"].is_duplicated().any():
        raise ValueError("Series classifications have multiple records; resolve before processing.")
    return sites, series


def audit_sources(config):
    raw = Path(config["LMN_raw"])
    extracted = Path(config["LMN_extracted"])
    quality = Path(config["LMN_processed"]) / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    sites, series = load_metadata(raw / "metadata")
    summaries, station_summaries, inventory = [], [], []
    files = sorted(raw.glob("[0-9][0-9][0-9][0-9]_PM*.csv"))
    if not files:
        raise FileNotFoundError(f"No LMN measurements in {raw}")
    for path in files:
        print(f"Auditing {path.name}", flush=True)
        source = read_csv(path)
        original_rows = len(source)
        source, duplicates = deduplicate(source)
        year = int(path.name[:4])
        expected_component = "PM10" if "PM10" in path.name else "PM2.5"
        frame = source.with_columns(
            parse_time("begindatumtijd").alias("time"),
            parse_time("einddatumtijd").alias("end"),
            pl.col("waarde").cast(pl.Float64, strict=False).alias("value"),
        ).join(sites, on="meetlocatie_id", how="left", validate="m:1").join(
            series, on=["meetreeks_id", "meetlocatie_id", "component", "matrix"],
            how="left", validate="m:1")
        duration = (pl.col("end") - pl.col("time")).dt.total_seconds()
        frame = frame.with_columns(
            (pl.col("value").is_not_null() & pl.col("value").is_finite()).alias("valid_value"),
            (pl.col("time").is_not_null() & pl.col("end").is_not_null()
             & (((pl.col("meetduur") == "uur") & (duration == 3600))
                | ((pl.col("meetduur") == "dag") & (duration == 86400)))
             & (pl.col("time").dt.minute() == 0)
             & (pl.col("time").dt.second() == 0)).fill_null(False).alias("valid_time"),
            (pl.col("lat").is_between(-90, 90) & pl.col("lon").is_between(-180, 180)
             & pl.col("stationstype").is_in(["achtergrond", "industrie", "verkeer"])
             & (pl.col("time") >= pl.col("site_start"))
             & (pl.col("time") >= pl.col("series_start"))
             & (pl.col("site_end").is_null() | (pl.col("end") <= pl.col("site_end")))
             & (pl.col("series_end").is_null() | (pl.col("end") <= pl.col("series_end")))
             ).fill_null(False).alias("valid_metadata"),
            ((pl.col("component") == expected_component) & (pl.col("matrix") == "lucht")
             & (pl.col("eenheid") == "µg/m³")).fill_null(False).alias("valid_identity"),
        )
        frame = frame.with_columns(pl.all_horizontal(
            "valid_value", "valid_time", "valid_metadata", "valid_identity").alias("usable"))
        finite = frame.filter(pl.col("valid_value"))["value"]
        threshold = finite.quantile(0.999) if len(finite) else None
        frame = frame.with_columns(
            (pl.col("value") < 0).fill_null(False).alias("negative"),
            (pl.col("value") > threshold).fill_null(False).alias("upper_tail"),
        )
        anomalies = frame.filter(~pl.col("usable") | pl.col("negative") | pl.col("upper_tail")
                                 | pl.col("opm_code").is_not_null())
        write_parquet(anomalies, quality / "flagged_observations" / f"{path.stem}.parquet",
                      units="µg/m³", upper_tail_threshold=threshold)
        summary = {"file": path.name, "year": year, "pollutant": expected_component,
                   "input_rows": original_rows, "identical_duplicates_removed": duplicates,
                   "rows_after_deduplication": len(frame), "stations": frame["meetlocatie_id"].n_unique(),
                   "invalid_values": frame.filter(~pl.col("valid_value")).height,
                   "invalid_times": frame.filter(~pl.col("valid_time")).height,
                   "invalid_metadata": frame.filter(~pl.col("valid_metadata")).height,
                   "invalid_identity": frame.filter(~pl.col("valid_identity")).height,
                   "usable_rows": frame.filter(pl.col("usable")).height,
                   "negative_rows": frame["negative"].sum(), "upper_tail_rows": frame["upper_tail"].sum(),
                   "minimum": finite.min(), "maximum": finite.max(), "mean": finite.mean(),
                   "p999": threshold, "durations": ",".join(sorted(frame["meetduur"].unique().to_list()))}
        summaries.append(summary)
        inventory.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
        # Gaps within the observed span do not label pre-opening/post-closure hours as missing.
        for (station, frequency), group in frame.filter(pl.col("valid_time")).group_by(
                "meetlocatie_id", "meetduur"):
            step = 3600 if frequency == "uur" else 86400
            start, end = group["time"].min(), group["end"].max()
            expected = int((end - start).total_seconds() / step)
            days = group.with_columns(pl.col("time").dt.convert_time_zone("Europe/Amsterdam")
                                      .dt.date().alias("date")).group_by("date").len()
            station_summaries.append({"file": path.name, "year": year, "pollutant": expected_component,
                "station": station, "frequency": frequency, "observations": len(group),
                "usable_observations": group["usable"].sum(), "start_utc": start.isoformat(),
                "end_utc": end.isoformat(), "absent_intervals_within_observed_span": expected-len(group),
                "observed_days": len(days), "whole_days_absent_within_observed_span":
                (end.astimezone(LOCAL).date()-start.astimezone(LOCAL).date()).days +
                (end.astimezone(LOCAL).time() != datetime.min.time())-len(days),
                "station_type": group["stationstype"][0], "lat": group["lat"][0], "lon": group["lon"][0]})
        hourly = frame.filter(pl.col("usable") & (pl.col("meetduur") == "uur")).select(
            "time", "meetlocatie_id", "value", "lat", "lon", "stationstype")
        write_parquet(hourly, extracted / f"{path.stem}.parquet", units="µg/m³", source_sha256=inventory[-1]["sha256"])
    pl.DataFrame(summaries).write_csv(quality / "source_audit.csv")
    pl.DataFrame(station_summaries).write_csv(quality / "station_year_audit.csv")
    (quality / "source_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return sites


def distances_km(target_lat, target_lon, source_lat, source_lon):
    lat1, lon1 = np.radians(target_lat)[:, None], np.radians(target_lon)[:, None]
    lat2, lon2 = np.radians(source_lat)[None, :], np.radians(source_lon)[None, :]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def interpolate(values, distances, method):
    """Return hourly estimates, donor counts and farthest-donor distances.

    Group identical availability masks to avoid rebuilding weights each hour.
    Stations at the exact target location share equal weight and override others.
    """
    values = np.asarray(values, dtype=float)
    distances = np.asarray(distances, dtype=float)
    hours, stations = values.shape
    municipalities = distances.shape[0]
    if distances.shape[1] != stations or method not in METHODS:
        raise ValueError("Invalid distance dimensions or interpolation method")
    result = np.full((hours, municipalities), np.nan)
    counts = np.zeros((hours, municipalities), dtype=np.uint16)
    farthest = np.full_like(result, np.nan)
    if not stations:
        return result, counts, farthest
    masks, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for i, mask in enumerate(masks):
        available = np.flatnonzero(mask)
        if not len(available):
            continue
        rows = np.flatnonzero(membership == i)
        d = distances[:, available]
        selected = np.zeros_like(d, dtype=bool)
        if method == "idw_60km":
            selected = d <= 60
        else:
            k = 1 if method == "nearest" else min(4, len(available))
            order = np.argsort(d, axis=1, kind="stable")[:, :k]
            np.put_along_axis(selected, order, True, axis=1)
        exact = d < 1e-9
        has_exact = exact.any(axis=1)
        selected[has_exact] = exact[has_exact]
        weights = np.where(selected, 1/np.maximum(d, 1e-9)**2, 0)
        denom = weights.sum(axis=1, keepdims=True)
        np.divide(weights, denom, out=weights, where=denom > 0)
        estimates = values[np.ix_(rows, available)] @ weights.T
        estimates[:, denom[:, 0] == 0] = np.nan
        result[rows] = estimates
        counts[rows] = selected.sum(axis=1)
        maximum = np.max(np.where(selected, d, -np.inf), axis=1)
        maximum[~np.isfinite(maximum)] = np.nan
        farthest[rows] = maximum
    return result, counts, farthest


def month_hours(year, month):
    start = datetime(year, month, 1, tzinfo=LOCAL)
    end = datetime(year+(month == 12), month % 12+1, 1, tzinfo=LOCAL)
    return pl.datetime_range(start.astimezone(UTC), end.astimezone(UTC), interval="1h",
                             closed="left", eager=True).alias("time")


def daily_mean(values, day_ids, expected_hours):
    sums = np.zeros((len(expected_hours), values.shape[1]))
    counts = np.zeros_like(sums, dtype=np.uint32)
    valid = np.isfinite(values)
    np.add.at(sums, day_ids, np.where(valid, values, 0))
    np.add.at(counts, day_ids, valid.astype(np.uint32))
    means = np.full_like(sums, np.nan)
    enough = counts >= np.ceil(np.asarray(expected_hours)[:, None]*0.75)
    np.divide(sums, counts, out=means, where=enough & (counts > 0))
    return means, counts


def process_month(year, month, sources, sites, metadata, distances, output):
    times = month_hours(year, month)
    local_dates = times.dt.convert_time_zone("Europe/Amsterdam").dt.date()
    dates = local_dates.unique().sort()
    day_ids = np.searchsorted(dates.to_numpy(), local_dates.to_numpy())
    expected = np.bincount(day_ids)
    base = pl.DataFrame({"date": dates, "n_obs": expected.astype(np.uint32)}).join(metadata, how="cross")
    support_base = base.select("date", "gemeentecode")
    time_table = pl.DataFrame({"time": times}).with_row_index("hour")
    last = times[-1]
    station_table = sites.select("meetlocatie_id").with_row_index("station")
    columns = {selection: [] for selection in ("background", "all_stations")}
    diagnostics = []
    for pollutant in POLLUTANTS:
        tag = "PM10" if pollutant == "pm10" else "PM25"
        paths = [p for p in sources if tag in p.name and abs(int(p.name[:4])-year) <= 1]
        values_by_selection = {}
        if paths:
            frame = pl.scan_parquet(paths).filter((pl.col("time") >= times[0]) &
                       (pl.col("time") <= last)).collect().join(time_table, on="time", how="inner").join(
                           station_table, on="meetlocatie_id", how="left", validate="m:1")
            if frame.select("time", "meetlocatie_id").is_duplicated().any():
                raise ValueError("Duplicate station-hour across source files")
        else:
            frame = None
        for selection in columns:
            values = np.full((len(times), len(sites)), np.nan)
            if frame is not None and len(frame):
                f = frame.filter(pl.col("stationstype") == "achtergrond") if selection == "background" else frame
                values[f["hour"].to_numpy(), f["station"].to_numpy()] = f["value"].to_numpy()
            values_by_selection[selection] = values
            for method in METHODS:
                estimates, donors, farthest = interpolate(values, distances, method)
                means, counts = daily_mean(estimates, day_ids, expected)
                name = f"{pollutant}_{method}"
                columns[selection].extend([
                    pl.Series(f"{name}_mean", means.ravel(), dtype=pl.Float32, nan_to_null=True),
                    pl.Series(f"{name}_n", counts.ravel(), dtype=pl.UInt32),
                ])
                # Support is summarized over hours with an estimate, even if daily coverage fails.
                donor_mean = np.full_like(means, np.nan)
                far_mean = np.full_like(means, np.nan)
                far_max = np.full_like(means, np.nan)
                donor_min = np.zeros_like(means, dtype=np.uint16)
                for day in range(len(dates)):
                    use = day_ids == day
                    valid = np.isfinite(estimates[use])
                    n = valid.sum(axis=0)
                    np.divide(np.where(valid, donors[use], 0).sum(axis=0), n,
                              out=donor_mean[day], where=n > 0)
                    np.divide(np.nansum(farthest[use], axis=0), n, out=far_mean[day], where=n > 0)
                    maximum = np.max(np.where(valid, farthest[use], -np.inf), axis=0)
                    far_max[day] = np.where(n > 0, maximum, np.nan)
                    donor_min[day] = np.where(n > 0, np.min(np.where(valid, donors[use], 65535), axis=0), 0)
                diagnostics.append(support_base.with_columns(
                    pl.lit(selection).alias("station_selection"), pl.lit(pollutant).alias("pollutant"),
                    pl.lit(method).alias("method"), pl.Series("valid_hours", counts.ravel()),
                    pl.Series("mean_donors", donor_mean.ravel(), nan_to_null=True),
                    pl.Series("min_donors", donor_min.ravel()),
                    pl.Series("mean_farthest_donor_km", far_mean.ravel(), nan_to_null=True),
                    pl.Series("max_farthest_donor_km", far_max.ravel(), nan_to_null=True),
                ))
        # Observed station-day comparisons use the same civil-day completeness rule.
        station_means, station_counts = daily_mean(values_by_selection["all_stations"], day_ids, expected)
        if pollutant == "pm10":
            pm10_daily, pm10_counts = station_means, station_counts
        else:
            paired = np.isfinite(pm10_daily) & np.isfinite(station_means)
            rows, cols = np.where(paired & (station_means > pm10_daily))
            pairs = pl.DataFrame({"date": dates.to_numpy()[rows],
                 "station": sites["meetlocatie_id"].to_numpy()[cols],
                 "pm10": pm10_daily[rows, cols], "pm2p5": station_means[rows, cols],
                 "pm10_n": pm10_counts[rows, cols], "pm2p5_n": station_counts[rows, cols]})
            write_parquet(pairs, output / "quality/pm25_above_pm10" / f"{year}_{month:02d}.parquet",
                          units="µg/m³", paired_station_days=int(paired.sum()))
    for selection, cols in columns.items():
        folder = "municipality_daily" if selection == "background" else "municipality_daily_all_stations"
        result = base.with_columns(cols).select("gemeentecode", "date", "n_obs", "gemeentenaam",
                 "centroid_lat", "centroid_lon", *[c.name for c in cols]).sort("gemeentecode", "date")
        write_parquet(result, output / folder / f"LMN_municipality_{year}_{month:02d}.parquet",
                      units="µg/m³", timezone="Europe/Amsterdam", station_selection=selection,
                      coverage_threshold="0.75", n_obs="expected civil-day hours; per-variable _n counts estimated hours",
                      spatial_methods="idw: up to 4 available; idw_60km: all available within 60 km; nearest: available",
                      boundaries="fixed 2025 municipalities")
    write_parquet(pl.concat(diagnostics), output / "quality/spatial_support" / f"{year}_{month:02d}.parquet")


def run(config, skip_audit=False):
    output = Path(config["LMN_processed"])
    if not skip_audit:
        audit_sources(config)
    sites, _ = load_metadata(Path(config["LMN_raw"]) / "metadata")
    sources = sorted(Path(config["LMN_extracted"]).glob("[0-9][0-9][0-9][0-9]_PM*.parquet"))
    if not sources:
        raise FileNotFoundError("No audited LMN inputs; run without --skip-audit")
    used = pl.scan_parquet(sources).select("meetlocatie_id").unique().collect()
    sites = sites.join(used, on="meetlocatie_id", how="inner").sort("meetlocatie_id")
    municipalities = read_municipalities(config["municipality_boundaries"])
    centroids = municipalities.geometry.centroid.to_crs("EPSG:4326")
    metadata = pl.DataFrame({"gemeentecode": municipalities["gemeentecode"].to_list(),
                            "gemeentenaam": municipalities["gemeentenaam"].to_list(),
                            "centroid_lat": centroids.y.to_numpy(), "centroid_lon": centroids.x.to_numpy()})
    distances = distances_km(metadata["centroid_lat"].to_numpy(), metadata["centroid_lon"].to_numpy(),
                            sites["lat"].to_numpy(), sites["lon"].to_numpy())
    for year in range(2000, 2026):
        for month in range(1, 13):
            print(f"Municipalities {year}-{month:02d}", flush=True)
            process_month(year, month, sources, sites, metadata, distances, output)
    (output / "README.md").write_text(README, encoding="utf-8")


README = """# LMN daily municipality estimates

Concentrations are **µg/m³** for LMN and for EAC4 after the extraction step.
Municipality boundaries are fixed at 2025. Dates are Dutch
civil days (Europe/Amsterdam), assigned using measurement interval start times.
All municipalities and dates in 2000–2025 are included. PM2.5 before 2014 is null:
the 2008–2013 daily samples cannot be aligned exactly to civil days and are audited
but excluded. Negative and extreme finite observations are retained, not clipped.

`municipality_daily/` uses background stations; `municipality_daily_all_stations/`
also includes traffic/industrial sites. Each file has one row per municipality/day.
`pm10/pm2p5_<method>_mean` stores the daily mean and `_n` its valid estimated hours.
`n_obs` is the expected civil-day hours (23/24/25), including when no data exist.
Daily means require ceil(0.75 * expected hours); counts remain available on failure.

Methods: `idw` = up to four nearest available stations, inverse-square weights;
`idw_60km` = all available stations within 60 km, inverse-square weights;
`nearest` = nearest available station. Donors change with each pollutant/hour.
Exact matches override other donors (coincident matches share weight). The first
and third methods have no radius limit, so distant extrapolation is possible.
There is no time imputation, PM1 estimate, or grid-cell area-weighting analogue.

Quality: source_audit.csv records counts and distributions; station_year_audit.csv
records gaps within observed spans (not a claim about planned sampling schedules).
Early daily sampling gaps can be scheduled; pre-opening/post-closure periods are
not counted. Flagged-observation files retain negatives, values above each source
file's 99.9th percentile, remarks, and invalid rows. Upper-tail flags are review
indicators, not evidence of error. Metadata date mismatches are excluded and logged.
Spatial-support files record hourly support summarized per municipality/day/method;
PM2.5 > PM10 files flag paired station-day discrepancies without altering values.

Original downloaded supporting files and download provenance live in
`../raw/metadata/`. Reruns reuse them; --refresh-metadata explicitly replaces them.
Run code/1_process/3_process_lmn.py with the private config to rebuild outputs.
--skip-audit reuses audited extracted inputs and should only be used if unchanged.
"""
