"""Bounded-memory Parquet concatenation and shared analysis-column rules."""
from pathlib import Path
import re
import uuid

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from aggregate import METADATA

METHODS = ("area", "idw", "idw_60km", "nearest")
METHOD_PATTERN = re.compile(r"_(idw_60km|idw|nearest|area)(?=_|$)")


def observation_count(name):
    return name == "n_obs" or name.endswith("_n")


def column_method(name):
    match = METHOD_PATTERN.search(name)
    return match.group(1) if match else None


def parquet_files(folder, pattern="*.parquet"):
    return sorted(p for p in Path(folder).glob(pattern)
                  if p.is_file() and not p.name.startswith(".")
                  and not p.name.endswith("_all.parquet"))


def temporary_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def concatenate_parquet_files(input_dir, output_file, file_pattern="*.parquet",
                              *, output_float32=True, batch_size=8192,
                              exclude_columns_containing=()):
    """Project counts away, compact floats, and write batches; return a Path.

    Preflight numeric ranges with streaming reductions so every batch uses the
    same schema. As in daily aggregation, keep metadata, wind directions and
    values outside Float32's normal range at their original precision.
    Never materialize the combined data. Publish only a completed output file.
    Optionally exclude column names containing any supplied substring before
    range checks and batch reads; source files remain unchanged.
    """
    output_file = Path(output_file)
    files = [p for p in parquet_files(input_dir, file_pattern)
             if p.resolve() != output_file.resolve()]
    if not files:
        raise FileNotFoundError(f"No source Parquet files in {input_dir}: {file_pattern}")
    schemas = []
    for path in files:
        schema = pl.read_parquet_schema(path)
        names = [n for n in schema if not observation_count(n)
                 and not any(part in n for part in exclude_columns_containing)]
        candidates = [n for n in names if schema[n] == pl.Float64
                      and n not in METADATA + ["dawn_offset_minutes"]
                      and not n.startswith("wd_")]
        cast = set()
        if output_float32 and candidates:
            stats = pl.scan_parquet(path).select([
                expr for n in candidates for expr in (
                    pl.col(n).abs().filter(pl.col(n).is_finite() & (pl.col(n) != 0)).min().alias(n + "__min"),
                    pl.col(n).abs().filter(pl.col(n).is_finite()).max().alias(n + "__max"),
                )]).collect(engine="streaming").row(0, named=True)
            for n in candidates:
                lo, hi = stats[n + "__min"], stats[n + "__max"]
                if (lo is None or lo >= 1.1754943508222875e-38) and (hi is None or hi <= 3.4028234663852886e38):
                    cast.add(n)
        arrow = pq.read_schema(path)
        schemas.append(pa.schema([pa.field(n, pa.float32() if n in cast else arrow.field(n).type)
                                  for n in names]))
    if any(set(s.names) != set(schemas[0].names) for s in schemas):
        raise ValueError("Source columns differ; regenerate consistent monthly files before concatenating.")
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    temp = temporary_output(output_file)
    rows = 0
    try:
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for path in files:
                with pq.ParquetFile(path) as source:
                    for batch in source.iter_batches(batch_size=batch_size, columns=schema.names, use_threads=False):
                        table = pa.Table.from_batches([batch]).select(schema.names).cast(schema)
                        writer.write_table(table, row_group_size=batch_size)
                        rows += table.num_rows
                print(f"Appended {path.name}", flush=True)
        temp.replace(output_file)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Saved {output_file}: {rows} rows; {len(schema)} columns", flush=True)
    return output_file
