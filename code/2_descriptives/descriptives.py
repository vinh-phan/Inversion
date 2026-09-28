"""Raw municipality-day measurement descriptives; see DESCRIPTIVES.md.

No models, residualization, or causal estimates. Only small column batches and
aggregated plotting tables leave lazy scans. Pair statistics use pairwise-complete
observations and exact, average-tie Spearman ranks on that overlap.
"""

# %% Settings and imports
import argparse
import csv
import hashlib
import itertools
import inspect
import json
import math
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import polars as pl

SCRIPT_FOLDER = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_FOLDER / "config" / "config.json"
VERSION = 1
SECTIONS = ["01_inversion_univariate", "02_pollution_univariate",
            "03_inversion_pollution_bivariate", "04_summary_comparisons"]
METHODS = {"area": "zonal_mean", "nearest": "NN", "idw": "IDW_4_nearest",
           "idw_60km": "IDW_60km"}
TIMES = {"03_local": "3AM_local", "06_local": "6AM_local", "dawn": "dawn_local",
         "all": "all_hours", "6h_local": "four_hours_per_day"}
CONSTRUCTIONS = ["975-1000", "950-1000", "925-1000",
                 "pressure-level-above-surface - 2m", "975 - 2m", "950 - 2m", "925 - 2m"]
LABELS = {"indicator_share": "Inversion share", "strength": "Inversion strength (K)",
          "3AM_local": "03:00 local", "6AM_local": "06:00 local", "dawn_local": "Civil dawn",
          "all_hours": "All hours", "four_hours_per_day": "Every six hours (local)",
          "zonal_mean": "Zonal mean", "NN": "Nearest neighbour",
          "IDW_4_nearest": "IDW: 4 nearest", "IDW_60km": "IDW: 60 km",
          "pressure-level-above-surface - 2m": "Surface-selected level − 2 m"}
ORDERS = {"construction": CONSTRUCTIONS, "operationalization": ["indicator_share", "strength"],
          "time": list(TIMES.values()), "spatial_method": list(METHODS.values()),
          "source": ["EAC4", "LMN"], "pollutant": ["PM10", "PM2.5"]}
COLORS = ["#426c8c", "#ae7655", "#67927e", "#8b799e", "#b49a51", "#6c929d", "#777777"]
META = ["variable_name", "family", "construction", "operationalization", "time",
        "spatial_method", "source", "pollutant", "units", "scale"]


def label(value):
    return LABELS.get(value, str(value))


def ordered(values, dimension):
    return sorted(set(values), key=lambda x: (ORDERS.get(dimension, []).index(x)
                  if x in ORDERS.get(dimension, []) else 999, str(x)))


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def save_table(table, path):
    """Atomic cache writes make interrupted calculations safely resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    table.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


# %% Catalog: patterns verified against df_daily.parquet and extraction code

def build_variable_catalog(schema):
    records, excluded = [], []
    inv = re.compile(r"^(?:ERA5_)?t_dif(?:_(975|950|925)_(1000|2m))?"
                     r"(?:_(indicator|truncated))?_(area|nearest|idw_60km|idw)_"
                     r"(03_local|06_local|dawn|all|6h_local|03_utc|06_utc|6h_utc)$")
    pollution = re.compile(r"^(EAC4|LMN)_(pm10|pm2p5|pm1)_(area|nearest|idw_60km|idw)_mean$")
    for name, dtype in schema.items():
        row = dict.fromkeys(META)
        row.update(variable_name=name, scale=1.0)
        m, p = inv.fullmatch(name), pollution.fullmatch(name)
        reason = None
        if m:
            upper, lower, op, method, timing = m.groups()
            if op is None:
                reason = "Raw signed temperature difference; outside requested operationalizations"
            elif timing not in TIMES:
                reason = "UTC timing; outside requested local-time definitions"
            else:
                construction = (f"{upper}-1000" if lower == "1000" else f"{upper} - 2m"
                                if lower == "2m" else CONSTRUCTIONS[3])
                row.update(family="inversion", construction=construction,
                           operationalization="indicator_share" if op == "indicator" else "strength",
                           time=TIMES[timing], spatial_method=METHODS[method],
                           units="share" if op == "indicator" else "K")
        elif p:
            source, pollutant, method = p.groups()
            if pollutant == "pm1":
                reason = "PM1 is outside requested pollutants"
            elif source == "LMN" and method == "area":
                reason = "UNPARSED: LMN zonal mean is not a verified project definition"
            else:
                row.update(family="pollution", source=source,
                           pollutant="PM10" if pollutant == "pm10" else "PM2.5",
                           spatial_method=METHODS[method], units="µg/m³")
        elif name.endswith("_n") or name == "n_obs":
            continue
        elif re.match(r"^(?:ERA5_)?t_dif|^(?:EAC4|LMN).*pm", name, re.I):
            reason = "UNPARSED: environmental column does not match verified naming conventions"
        else:
            continue  # Identifiers, weather controls, and dawn diagnostics are out of scope.
        if reason:
            excluded.append({"variable_name": name, "reason": reason})
        else:
            if not dtype.is_numeric():
                raise ValueError(f"Environmental measurement is not numeric: {name} ({dtype})")
            records.append(row)
    catalog = frame(records)
    if not records:
        raise ValueError("No requested environmental variables were found")
    dims = [c for c in META if c not in {"variable_name", "scale", "units"}]
    if catalog.select(dims).is_duplicated().any():
        raise ValueError("Ambiguous duplicate environmental definitions in catalog")
    return catalog, frame(excluded)


def print_catalog(catalog, excluded):
    print("\nVariable catalog (before calculations)", flush=True)
    for family in ["inversion", "pollution"]:
        subset = catalog.filter(pl.col("family") == family)
        print(f"  {family}: {subset.height}")
        dimensions = ["construction", "operationalization", "time", "spatial_method"] if family == "inversion" else ["source", "pollutant", "spatial_method"]
        for dim in dimensions:
            counts = subset.group_by(dim).len().sort(dim).to_dicts()
            print(f"    {dim}: " + "; ".join(f"{r[dim]}={r['len']}" for r in counts))
    if excluded.height:
        print(f"  Explicitly excluded/unparsed columns: {excluded.height} (see excluded_columns.csv)")
        for row in excluded.filter(pl.col("reason").str.starts_with("UNPARSED")).to_dicts():
            print(f"  WARNING {row['variable_name']}: {row['reason']}")
    present = {(r['source'], r['pollutant'], r['spatial_method']) for r in catalog.to_dicts() if r['family'] == 'pollution'}
    for source, pollutant, method in itertools.product(["EAC4", "LMN"], ["PM10", "PM2.5"], METHODS.values()):
        if source == "LMN" and method == "zonal_mean":
            continue
        if (source, pollutant, method) not in present:
            print(f"  Unavailable: {source} {pollutant}, {label(method)}")
    print("  Pollution concentrations are in µg/m³ for both EAC4 and LMN. No negative values removed.", flush=True)


class Analysis:
    def __init__(self, source, catalog, root, settings):
        self.source, self.catalog, self.root, self.settings = source, catalog, root, settings
        self.metadata = {r['variable_name']: r for r in catalog.to_dicts()}
        self.cache = root / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def project(self, names, date=False):
        # Nonfinite measurements cannot define moments/ranks. Counts excluded for
        # this mathematical reason are recorded; finite unusual values are retained.
        expressions = []
        for name in dict.fromkeys(names):
            x = pl.col(name).cast(pl.Float64)
            expressions.append(pl.when(x.is_finite()).then(x).otherwise(None).alias(name))
        if date:
            expressions.append(pl.col("date").cast(pl.Date).dt.month().alias("month"))
        return self.source.select(expressions)

    def summarize(self):
        destination = self.cache / "univariate.parquet"
        if destination.exists():
            return pl.read_parquet(destination)
        batches = list(chunks(list(self.metadata), self.settings['batch_size']))
        summaries, distributions, seasons = [], [], []
        for index, names in enumerate(batches):
            paths = [self.cache / f"univariate_{index:03d}_{kind}.parquet"
                     for kind in ["summary", "ecdf", "monthly"]]
            if not all(p.exists() for p in paths):
                # Exact quantiles require values, but at most batch_size columns
                # are materialized. No municipality-day long table is created.
                data = self.project(names, date=True).collect(engine="streaming")
                batch_summary, batch_ecdf, batch_monthly = [], [], []
                for name in names:
                    meta = self.metadata[name]
                    x = pl.col(name)
                    conditions = [False, True] if meta['operationalization'] == 'strength' else [False]
                    for positive in conditions:
                        q = data.lazy().filter(x > 0) if positive else data.lazy()
                        stats = q.select(
                            x.count().alias("n"), x.mean().alias("mean"), x.std().alias("sd"),
                            x.median().alias("median"), x.min().alias("minimum"), x.max().alias("maximum"),
                            *[x.quantile(p, interpolation="linear").alias(f"p{int(p*100)}") for p in [.1, .25, .75, .9, .95]],
                            (x == 0).mean().alias("share_zero"), (x > 0).mean().alias("share_positive"),
                            (x == 1).mean().alias("share_one"), x.drop_nulls().n_unique().alias("n_unique"),
                            x.is_null().sum().alias("n_null_or_nonfinite")
                        ).collect().row(0, named=True)
                        batch_summary.append({**meta, "positive_only": positive, **stats})
                    values = data[name].drop_nulls().sort()
                    n = len(values)
                    # A compact quantile-grid ECDF, with exact ranks at ties. It
                    # includes both endpoints; the zero/one mass is not smoothed.
                    if n:
                        indices = np.unique(np.rint(np.linspace(0, n-1, 201)).astype(int))
                        points = np.unique(values.to_numpy()[indices])
                        ranks = np.searchsorted(values.to_numpy(), points, side="right")
                        batch_ecdf.extend({"variable_name": name, "value": float(v), "cdf": float(r/n), "n": n}
                                          for v, r in zip(points, ranks))
                    monthly = data.lazy().group_by("month").agg(x.mean().alias("mean"), x.count().alias("n")).collect()
                    batch_monthly.extend({"variable_name": name, **r} for r in monthly.to_dicts())
                for rows, path in zip([batch_summary, batch_ecdf, batch_monthly], paths):
                    save_table(frame(rows), path)
            for dest, path in zip([summaries, distributions, seasons], paths):
                dest.append(pl.read_parquet(path))
            if index % 5 == 0 or index == len(batches)-1:
                print(f"  Univariate batches {index+1}/{len(batches)}", flush=True)
        result = pl.concat(summaries, how="diagonal_relaxed")
        for tables, name in [(distributions, "ecdf"), (seasons, "monthly")]:
            table = pl.concat(tables, how="diagonal_relaxed").join(self.catalog, on="variable_name")
            save_table(table, self.cache / f"{name}.parquet")
        unusual = result.filter(~pl.col("positive_only") & (
            ((pl.col("operationalization") == "indicator_share") & ((pl.col("minimum") < 0) | (pl.col("maximum") > 1))) |
            ((pl.col("operationalization") == "strength") & (pl.col("minimum") < 0))))
        if unusual.height:
            print(f"WARNING: {unusual.height} inversion measures outside expected range; values retained. See range_flags.parquet.")
            save_table(unusual, self.cache / "range_flags.parquet")
        save_table(result, destination)
        return result

    def pairs(self, tasks):
        """Reuse each projected anchor/partner batch; checkpoint exact results.

        Ranks are computed AFTER pairwise deletion. Global ranks with different
        missingness masks would yield the wrong Spearman correlation.
        """
        groups = {}
        for task in tasks:
            groups.setdefault(task['a'], []).append(task)
        batches = [(a, batch) for a, group in groups.items()
                   for batch in chunks(group, self.settings['batch_size'] - 1)]
        result = []
        start = time.monotonic()
        for i, (a, batch) in enumerate(batches):
            key = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()[:20]
            path = self.cache / "pairs" / f"{key}.parquet"
            if not path.exists():
                data = self.project([a] + [t['b'] for t in batch]).collect(engine="streaming")
                rows = []
                for task in batch:
                    b = task['b']
                    paired = data.lazy().select(pl.col(a).alias("x"), pl.col(b).alias("y")).drop_nulls()
                    x, y = pl.col("x"), pl.col("y")
                    stats = paired.select(
                        pl.len().alias("overlap_n"), pl.corr(x, y).alias("pearson"),
                        pl.corr(x, y, method="spearman").alias("spearman"),
                        x.mean().alias("a_mean"), y.mean().alias("b_mean"),
                        x.std().alias("a_sd"), y.std().alias("b_sd"),
                        (x-y).mean().alias("mean_difference"), (x-y).median().alias("median_difference"),
                        (x-y).abs().mean().alias("mae"), ((x-y)**2).mean().sqrt().alias("rmse")
                    ).collect().row(0, named=True)
                    comparable = task['kind'] not in {'association', 'operationalization'}
                    if not comparable:
                        for metric in ['mean_difference', 'median_difference', 'mae', 'rmse']:
                            stats[metric] = None
                    rows.append({**task, **stats, "difference_direction": "A minus B" if comparable else None})
                    if task['plot']:
                        paired_data = paired.collect()
                        self.cache_pair_plot(task, paired_data)
                save_table(frame(rows), path)
            result.append(pl.read_parquet(path))
            if i % 10 == 0 or i == len(batches)-1:
                print(f"  Pair batches {i+1}/{len(batches)} ({(time.monotonic()-start)/60:.1f} min)", flush=True)
        return pl.concat(result, how="diagonal_relaxed") if result else pl.DataFrame()

    def cache_pair_plot(self, task, paired):
        token = task['id']
        path = self.cache / "plot_data" / f"{token}.parquet"
        if task['kind'] == 'association':
            save_table(bin_association(paired, self.metadata[task['a']]['operationalization']), path)
            return
        # Full-data 2D bin counts by default; optional deterministic capped sample
        # applies ONLY to display grids, never statistics or binned mean curves.
        full_paired = paired
        cap = self.settings['plot_sample_size']
        if cap and paired.height > cap:
            paired = paired.sample(n=cap, seed=self.settings['seed'], shuffle=True)
        scatter = (paired.select(pl.col('y').alias('x'), pl.col('x').alias('y'))
                   if task['kind'] == 'source' else paired)
        grids = [density_grid(scatter).with_columns(pl.lit("scatter").alias("view"))]
        if task['kind'] == 'source':
            # A=LMN, B=EAC4, so x=pair mean and y=LMN-EAC4.
            ba = paired.select(((pl.col('x')+pl.col('y'))/2).alias('x'),
                               (pl.col('x')-pl.col('y')).alias('y'))
            grids.append(density_grid(ba).with_columns(pl.lit("bland_altman").alias("view")))
        save_table(pl.concat(grids, how="diagonal_relaxed"), path)
        if task['kind'] == 'pollutant':
            paired = full_paired
            # A=PM2.5, B=PM10. Negative/nonzero denominators remain mathematically
            # defined and visible. Zero denominators are counted explicitly.
            ratio = paired.filter(pl.col('y') != 0).select((pl.col('x')/pl.col('y')).alias('ratio'))
            info = paired.select(pl.len().alias('overlap_n'), (pl.col('y') == 0).sum().alias('zero_denominator_n'),
                                 (pl.col('y') < 0).sum().alias('negative_denominator_n'))
            save_table(info, self.cache / "plot_data" / f"{token}_ratio_info.parquet")
            if ratio.height:
                v = ratio['ratio'].sort().to_numpy()
                points = np.unique(v[np.rint(np.linspace(0, len(v)-1, 201)).astype(int)])
                table = pl.DataFrame({'ratio': points, 'cdf': np.searchsorted(v, points, side='right')/len(v)})
            else:
                table = pl.DataFrame(schema={'ratio': pl.Float64, 'cdf': pl.Float64})
            save_table(table, self.cache / "plot_data" / f"{token}_ratio.parquet")


def bin_association(paired, operationalization):
    """Exact pairwise means. Shares: 0/1 categories or 20 fixed-width bins.

    Strength: zero separately plus up to 15 positive quantile bins, dropping
    duplicate edges for ties. Unexpected negatives get their own category.
    """
    if not paired.height:
        return pl.DataFrame(schema={'bin': pl.Int32, 'x_mean': pl.Float64, 'y_mean': pl.Float64,
                                    'n': pl.UInt32, 'x_min': pl.Float64, 'x_max': pl.Float64, 'binning': pl.String})
    x = pl.col('x')
    minimum, maximum = paired['x'].min(), paired['x'].max()
    if operationalization == 'indicator_share':
        if paired['x'].n_unique() <= 2:
            code = x.rank('dense').cast(pl.Int32) - 1
            rule = 'Distinct-value categories (at most two values)'
        else:
            low, high = min(0., minimum), max(1., maximum)
            code = ((x-low)/(high-low)*20).floor().clip(0, 19).cast(pl.Int32)
            rule = '20 fixed-width bins over [min(0, observed min), max(1, observed max)]'
    else:
        positives = paired.filter(x > 0)['x']
        edges = np.unique(np.quantile(positives.to_numpy(), np.linspace(0, 1, 16)[1:-1])) if len(positives) else []
        positive_code = pl.lit(1)
        for edge in edges:
            positive_code = positive_code + (x > float(edge)).cast(pl.Int32)
        code = pl.when(x < 0).then(-1).when(x == 0).then(0).otherwise(positive_code)
        rule = 'Zero separate; up to 15 positive quantile bins; unexpected negatives separate'
    return (paired.lazy().with_columns(code.alias('bin')).group_by('bin')
            .agg(x.mean().alias('x_mean'), pl.col('y').mean().alias('y_mean'), pl.len().alias('n'),
                 x.min().alias('x_min'), x.max().alias('x_max')).sort('x_mean')
            .with_columns(pl.lit(rule).alias('binning')).collect())


def density_grid(paired, bins=40):
    """Aggregate all finite points into a bounded display grid, retaining tails."""
    schema = {'ix': pl.Int32, 'iy': pl.Int32, 'n': pl.UInt32, 'x_min': pl.Float64,
              'x_max': pl.Float64, 'y_min': pl.Float64, 'y_max': pl.Float64, 'bins': pl.Int32}
    if not paired.height:
        return pl.DataFrame(schema=schema)
    limits = {}
    codes = []
    for axis in ['x', 'y']:
        low, high = paired[axis].min(), paired[axis].max()
        if high == low:
            low, high = low-.5, high+.5
        limits.update({f'{axis}_min': low, f'{axis}_max': high})
        codes.append(((pl.col(axis)-low)/(high-low)*bins).floor().clip(0, bins-1).cast(pl.Int32).alias('i'+axis))
    return (paired.lazy().select(codes).group_by('ix', 'iy').agg(pl.len().alias('n'))
            .with_columns(*[pl.lit(v).alias(k) for k,v in limits.items()], pl.lit(bins).alias('bins')).collect())


# %% Comparison design: every eligible pair, figures selected independently

def make_tasks(catalog, settings):
    inv = catalog.filter(pl.col('family') == 'inversion').to_dicts()
    pol = catalog.filter(pl.col('family') == 'pollution').to_dicts()
    tasks = []
    baseline = {'construction': settings['reference_construction'], 'time': settings['reference_time'],
                'spatial_method': settings['reference_spatial_method'], 'operationalization': 'indicator_share',
                'source': settings['reference_source'], 'pollutant': settings['reference_pollutant']}

    def add(a, b, kind, dimension, plot=False):
        token = hashlib.sha256(f"{a['variable_name']}|{b['variable_name']}|{kind}".encode()).hexdigest()[:20]
        tasks.append({'id': token, 'a': a['variable_name'], 'b': b['variable_name'],
                      'kind': kind, 'dimension': dimension, 'plot': plot})

    dims = ['construction', 'operationalization', 'time', 'spatial_method']
    for a,b in itertools.combinations(inv, 2):
        differences = [d for d in dims if a[d] != b[d]]
        if len(differences) != 1:
            continue
        dim = differences[0]
        if dim == 'operationalization':
            if a[dim] != 'indicator_share':
                a,b = b,a
            selected = a['construction'] == baseline['construction'] and a['spatial_method'] == baseline['spatial_method']
            add(a,b,'operationalization',dim,selected)
        else:
            add(a,b,'inversion',dim)
    for a,b in itertools.combinations(pol,2):
        differences = [d for d in ['source','pollutant','spatial_method'] if a[d] != b[d]]
        if len(differences) != 1:
            continue
        dim = differences[0]
        if dim == 'source':
            if a['source'] != 'LMN':
                a,b=b,a
            add(a,b,'source',dim,True)
        elif dim == 'pollutant':
            if a['pollutant'] != 'PM2.5':
                a,b=b,a
            add(a,b,'pollutant',dim,True)
        else:
            add(a,b,'pollution_spatial',dim)
    for a,b in itertools.product(inv, pol):
        matched = a['spatial_method'] == b['spatial_method']
        if not matched and not settings['cross_spatial']:
            continue
        # A compact reference sweep: each selected curve changes one substantive
        # dimension, separately within each operationalization. Methods stay matched.
        choices = {**{k:a[k] for k in ['construction','time','spatial_method']},
                   **{k:b[k] for k in ['source','pollutant']}}
        selected = matched and sum(choices[k] != baseline[k] for k in choices) <= 1
        add(a,b,'association','matched' if matched else 'cross_spatial',selected)
    return tasks, baseline


def attach_pair_metadata(pairs, catalog):
    a = catalog.rename({c:'a_'+c for c in catalog.columns})
    b = catalog.rename({c:'b_'+c for c in catalog.columns})
    return pairs.join(a, left_on='a', right_on='a_variable_name').join(b, left_on='b', right_on='b_variable_name')


def build_association_summary(pairs):
    assoc = pairs.filter(pl.col('kind') == 'association')
    renames = {'a':'inversion_variable', 'b':'pollution_variable',
               'a_mean':'inversion_mean', 'a_sd':'inversion_sd', 'b_mean':'pollution_mean', 'b_sd':'pollution_sd'}
    for d in ['construction', 'operationalization', 'time', 'spatial_method']:
        renames['a_'+d] = 'inversion_'+d
    for d in ['source', 'pollutant', 'spatial_method']:
        renames['b_'+d] = 'pollution_'+d
    cols = ['id','overlap_n','pearson','spearman','plot','dimension'] + list(renames)
    return assoc.select(cols).rename(renames)

# %% Figures use cached, small tables only
class Figures:
    def __init__(self, root):
        self.root, self.index = root, []
        plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
            'axes.spines.top':False, 'axes.spines.right':False,
            'axes.prop_cycle':plt.cycler(color=COLORS), 'savefig.dpi':300,
            'pdf.fonttype':42, 'axes.titleweight':'medium'})

    def save(self, fig, section, name, description, dimensions, data):
        directory = self.root / section
        directory.mkdir(parents=True, exist_ok=True)
        for extension in ['png','pdf']:
            path = directory / f'{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            self.index.append({'filename':str(path.relative_to(self.root)), 'section':section,
                               'description':description, 'variables_dimensions':dimensions,
                               'plotting_data':data})
        plt.close(fig)

    def finish(self):
        with (self.root/'figure_index.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename','section','description','variables_dimensions','plotting_data'])
            writer.writeheader()
            writer.writerows(self.index)


def panel_grid(n, width=4.4, height=3.1, cols=3):
    cols = min(cols, max(n,1))
    fig, axes = plt.subplots(math.ceil(max(n,1)/cols), cols, figsize=(width*cols,height*math.ceil(max(n,1)/cols)),
                             squeeze=False, layout='constrained')
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def finite(value):
    return value is not None and np.isfinite(value)


def title_stats(row):
    def number(v):
        return f'{v:.2f}' if finite(v) else 'undefined'
    return f"r={number(row['pearson'])}; ρ={number(row['spearman'])}; N={row['overlap_n']:,}"


def filter_reference(table, baseline, changing, prefix=''):
    for dim in ['construction','time','spatial_method']:
        if dim != changing:
            table = table.filter(pl.col(prefix+dim) == baseline[dim])
    return table


def plot_inversion_univariate(figures, summary, ecdf, monthly, baseline):
    summary = summary.filter((pl.col('family') == 'inversion') & ~pl.col('positive_only'))
    for op in ORDERS['operationalization']:
        # An all-definition overview: each facet holds time fixed; offsets identify
        # spatial method and whiskers describe p25–p75, never uncertainty.
        data = summary.filter(pl.col('operationalization') == op)
        times = ordered(data['time'].to_list(), 'time')
        fig, axes = panel_grid(len(times), width=5.0, height=3.7)
        for ax, timing in zip(axes,times):
            part = data.filter(pl.col('time') == timing)
            constructions = ordered(part['construction'].to_list(),'construction')
            for j, method in enumerate(METHODS.values()):
                for row in part.filter(pl.col('spatial_method') == method).to_dicts():
                    y = constructions.index(row['construction']) + (j-1.5)*.15
                    if finite(row['mean']):
                        ax.plot([row['p25'],row['p75']],[y,y], color=COLORS[j], lw=1.1, alpha=.7)
                        ax.plot(row['mean'],y,'o',color=COLORS[j],ms=3)
                ax.plot([],[], 'o', color=COLORS[j],label=label(method),ms=4)
            ax.set_yticks(range(len(constructions)), [label(c) for c in constructions], fontsize=8)
            ax.invert_yaxis(); ax.set_title(label(timing)); ax.set_xlabel(label(op))
        if len(times):
            handles, names = axes[0].get_legend_handles_labels()
            if len(axes) > len(times):
                axes[-1].set_visible(True); axes[-1].set_axis_off()
                axes[-1].legend(handles,names,loc='center',fontsize=9,frameon=False)
            else:
                axes[0].legend(fontsize=7,loc='best')
        fig.suptitle(f'{label(op)}: all definitions\nDots: means; lines: middle 50% of municipality-days')
        figures.save(fig, SECTIONS[3], f'inversion_{op}_overview', 'All definitions: means and p25–p75 distribution intervals',
                     'construction × time × spatial method; '+op, 'cache/univariate.parquet')
        # Three reference sweeps, all held dimensions stated on the page.
        fig, axes = panel_grid(3, width=5.2, height=3.8)
        for ax, dim in zip(axes,['construction','time','spatial_method']):
            part = filter_reference(data,baseline,dim)
            values = ordered(part[dim].to_list(),dim)
            for i,value in enumerate(values):
                row = part.filter(pl.col(dim) == value).row(0,named=True)
                if finite(row['mean']):
                    ax.plot([row['p25'],row['p75']],[i,i],color=COLORS[0],lw=2)
                    ax.plot(row['mean'],i,'o',color=COLORS[0],ms=5)
            ax.set_yticks(range(len(values)),[label(v) for v in values],fontsize=8)
            ax.invert_yaxis(); ax.set_xlabel(label(op)); ax.set_title('Vary '+dim.replace('_',' '))
        fig.suptitle(reference_caption(baseline)+'\nMeans and middle 50%; only the named dimension changes',fontsize=10)
        figures.save(fig,SECTIONS[0],f'{op}_mean_sensitivity','Reference sweeps of means and distribution intervals',
                     json.dumps(baseline), 'cache/univariate.parquet')
        for kind, table in [('ecdf',ecdf),('monthly',monthly)]:
            fig, axes = panel_grid(3,width=5.0,height=3.8)
            for ax,dim in zip(axes,['construction','time','spatial_method']):
                part = filter_reference(table.filter(pl.col('operationalization') == op),baseline,dim)
                for value in ordered(part[dim].to_list(),dim):
                    line = part.filter(pl.col(dim) == value).sort('value' if kind == 'ecdf' else 'month')
                    if kind == 'ecdf':
                        ax.step(line['value'],line['cdf'],where='post',label=label(value),lw=1.3)
                    else:
                        ax.plot(line['month'],line['mean'],label=label(value),lw=1.3)
                ax.set_title('Vary '+dim.replace('_',' ')); ax.legend(fontsize=6.5)
                ax.set_xlabel(label(op) if kind == 'ecdf' else 'Calendar month')
                ax.set_ylabel('Cumulative proportion' if kind == 'ecdf' else label(op))
                if kind == 'monthly':
                    ax.set_xticks([1,3,5,7,9,11])
            fig.suptitle(reference_caption(baseline)+'\n'+('Quantile-grid ECDF (201 order statistics; full range)' if kind == 'ecdf' else 'Monthly means pooled across years and municipality-days'),fontsize=10)
            figures.save(fig,SECTIONS[0],f'{op}_{kind}_sensitivity','Change one dimension at a time: '+kind,
                         json.dumps(baseline),f'cache/{kind}.parquet')


def reference_caption(baseline):
    return 'Reference: '+', '.join(label(baseline[d]) for d in ['construction','time','spatial_method'])


def heatmap(ax, data, rows, columns, title, colorbar=True):
    shown = np.ma.masked_invalid(np.asarray(data,dtype=float))
    artist = ax.imshow(shown, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(columns)),[label(c) for c in columns],rotation=40,ha='right',fontsize=7)
    ax.set_yticks(range(len(rows)),[label(r) for r in rows],fontsize=7)
    ax.set_title(title,fontsize=9)
    if np.size(data) <= 40:
        for i,j in itertools.product(range(len(rows)),range(len(columns))):
            value = data[i][j]
            if finite(value):
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=6.5,
                        color='white' if abs(value)>.65 else '#333333')
    if colorbar:
        ax.figure.colorbar(artist,ax=ax,shrink=.7,label='Correlation')


def pair_matrix(table, dimension, metric):
    values = ordered(table['a_'+dimension].to_list()+table['b_'+dimension].to_list(),dimension)
    mat = np.full((len(values),len(values)),np.nan)
    # Diagonals deliberately blank: constants/all-null variables need not have
    # defined self-correlations, and only distinct-variable pairs were computed.
    for r in table.to_dicts():
        i,j = values.index(r['a_'+dimension]),values.index(r['b_'+dimension])
        mat[i,j] = mat[j,i] = r[metric] if finite(r[metric]) else np.nan
    return values,mat


def plot_agreement(figures,pairs,baseline):
    for metric in ['pearson','spearman']:
        fig,axes = panel_grid(6,width=4.6,height=3.8)
        for ax,(op,dim) in zip(axes,itertools.product(ORDERS['operationalization'],['construction','time','spatial_method'])):
            subset = pairs.filter((pl.col('kind') == 'inversion') & (pl.col('dimension') == dim) & (pl.col('a_operationalization') == op))
            subset = filter_reference(subset,baseline,dim,'a_')
            if not subset.height:
                ax.set_visible(False); continue
            values,mat = pair_matrix(subset,dim,metric)
            heatmap(ax,mat,values,values,label(op)+'\nVary '+dim.replace('_',' '))
        fig.suptitle(reference_caption(baseline)+f'\n{metric.title()} agreement; other dimensions held fixed',fontsize=10)
        figures.save(fig,SECTIONS[0],f'inversion_agreement_{metric}','Pairwise agreement along one dimension',json.dumps(baseline),'cache/pair_summary.parquet')
    op_pairs = pairs.filter(pl.col('kind') == 'operationalization')
    fig,axes = panel_grid(4,width=5,height=3.5,cols=2)
    for ax,method in zip(axes,METHODS.values()):
        part = op_pairs.filter(pl.col('a_spatial_method') == method)
        rows = ordered(part['a_construction'].to_list(),'construction')
        cols = ordered(part['a_time'].to_list(),'time')
        if not rows or not cols:
            ax.set_visible(False); continue
        matrix = np.full((len(rows),len(cols)),np.nan)
        for row in part.to_dicts():
            matrix[rows.index(row['a_construction']),cols.index(row['a_time'])] = row['pearson']
        heatmap(ax,matrix,rows,cols,label(method))
    fig.suptitle('Inversion share versus strength: Pearson correlation')
    figures.save(fig,SECTIONS[0],'share_strength_correlation','Matched share–strength agreement',
                 'construction × timing × spatial method','cache/pair_summary.parquet')
    for kind,section in [('inversion',SECTIONS[0]),('pollution_spatial',SECTIONS[1])]:
        subset = pairs.filter((pl.col('kind') == kind) & (pl.col('dimension') == 'spatial_method'))
        if kind == 'inversion':
            subset = subset.filter((pl.col('a_construction') == baseline['construction']) & (pl.col('a_time') == baseline['time']))
            groups = ['a_operationalization']
        else:
            groups = ['a_source','a_pollutant']
        facets = subset.partition_by(groups,maintain_order=True)
        fig,axes = panel_grid(len(facets),width=5,height=3.6,cols=2)
        for ax,part in zip(axes,facets):
            rows = part.to_dicts()
            positions = np.arange(len(rows))
            ax.scatter([r['mean_difference'] for r in rows],positions,s=22,color=COLORS[0])
            ax.set_yticks(positions,[label(r['a_spatial_method'])+' − '+label(r['b_spatial_method']) for r in rows],fontsize=7)
            ax.axvline(0,lw=.7,color='#aaaaaa'); ax.invert_yaxis()
            ax.set_title(' / '.join(label(rows[0][g]) for g in groups))
            unit = ('share' if rows[0]['a_operationalization'] == 'indicator_share' else 'K') if kind == 'inversion' else 'µg/m³'
            ax.set_xlabel('Mean of paired differences ('+unit+')')
        fig.suptitle('Spatial-method level differences'+ ('\n'+reference_caption(baseline) if kind == 'inversion' else ''))
        figures.save(fig,section,f'{kind}_spatial_differences','Signed pairwise mean differences',json.dumps(baseline) if kind == 'inversion' else 'source × pollutant × method pair','cache/pair_summary.parquet')
    for metric in ['pearson','spearman']:
        subset = pairs.filter(pl.col('kind') == 'pollution_spatial')
        facets = subset.partition_by(['a_source','a_pollutant'],maintain_order=True)
        fig,axes = panel_grid(len(facets),width=4.7,height=3.8,cols=2)
        for ax,part in zip(axes,facets):
            values,mat = pair_matrix(part,'spatial_method',metric)
            heatmap(ax,mat,values,values,part['a_source'][0]+' '+part['a_pollutant'][0])
        fig.suptitle('Pollution spatial-method agreement: '+metric)
        figures.save(fig,SECTIONS[1],f'pollution_spatial_{metric}','Pollution agreement holding source and pollutant fixed',
                     'source × pollutant × spatial methods','cache/pair_summary.parquet')


def plot_pollution(figures,summary,ecdf,monthly,pairs):
    for kind,table in [('ecdf',ecdf),('monthly',monthly)]:
        fig,axes = panel_grid(8,width=3.6,height=3.0,cols=4)
        for ax,(pollutant,method) in zip(axes,itertools.product(['PM10','PM2.5'],METHODS.values())):
            subset = table.filter((pl.col('pollutant') == pollutant) & (pl.col('spatial_method') == method))
            for source in ['EAC4','LMN']:
                line = subset.filter(pl.col('source') == source).sort('value' if kind == 'ecdf' else 'month')
                if not line.height:
                    continue
                if kind == 'ecdf':
                    ax.step(line['value'],line['cdf'],where='post',label=source,lw=1.3)
                else:
                    ax.plot(line['month'],line['mean'],label=source,lw=1.3)
            ax.set_title(pollutant+' / '+label(method))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            ax.set_xlabel('Concentration (µg/m³)' if kind == 'ecdf' else 'Calendar month')
            ax.set_ylabel('Cumulative proportion' if kind == 'ecdf' else 'Mean concentration (µg/m³)')
            if kind == 'ecdf':
                limits = table.filter(pl.col('pollutant') == pollutant)['value']
                if len(limits) and limits.min() != limits.max():
                    ax.set_xlim(limits.min(),limits.max())
            else:
                ax.set_xticks([1,3,5,7,9,11])
        fig.suptitle('Pollution '+('quantile-grid ECDF; negative values and full range retained' if kind == 'ecdf' else 'seasonality; pooled municipality-day means'))
        figures.save(fig,SECTIONS[1],f'pollution_{kind}','Pollution distributions/seasonality by source and method',
                     'source × pollutant × spatial method',f'cache/{kind}.parquet')
    source_pairs = pairs.filter(pl.col('kind') == 'source')
    fig,axes = panel_grid(4,width=5,height=3,cols=2)
    for i,pollutant in enumerate(['PM10','PM2.5']):
        part = source_pairs.filter(pl.col('a_pollutant') == pollutant).to_dicts()
        names = [label(r['a_spatial_method']) for r in part]
        for j,metric in enumerate(['pearson','mean_difference']):
            ax = axes[i*2+j]
            ax.scatter([r[metric] for r in part],range(len(part)),color=COLORS[0],label='Pearson' if j==0 else None)
            if j==0:
                ax.scatter([r['spearman'] for r in part],np.arange(len(part))+.15,color=COLORS[1],label='Spearman')
                ax.set_xlim(-1,1); ax.legend(fontsize=7)
            else:
                ax.axvline(0,color='#aaaaaa',lw=.7)
            ax.set_yticks(range(len(part)),names,fontsize=8); ax.set_title(pollutant)
            ax.set_xlabel('Correlation' if j==0 else 'Mean LMN − EAC4 (µg/m³)')
    fig.suptitle('EAC4 versus LMN: same method, overlapping municipality-days')
    figures.save(fig,SECTIONS[3],'pollution_source_summary','Source agreement and signed differences on overlap',
                 'pollutant × common spatial method','cache/pair_summary.parquet')


def draw_density(ax,grid):
    if not grid.height:
        ax.text(.5,.5,'No paired observations',ha='center',transform=ax.transAxes); return
    row = grid.row(0,named=True); bins = row['bins']
    counts = np.zeros((bins,bins))
    for r in grid.to_dicts():
        counts[r['iy'],r['ix']] = r['n']
    artist = ax.pcolormesh(np.linspace(row['x_min'],row['x_max'],bins+1),
                          np.linspace(row['y_min'],row['y_max'],bins+1),np.ma.masked_equal(counts,0),
                          norm=LogNorm(vmin=1,vmax=max(2,counts.max())),cmap='cividis',rasterized=True)
    ax.figure.colorbar(artist,ax=ax,shrink=.7,label='Count (log scale)')


def plot_pair_grids(figures,pairs,settings):
    for kind in ['source','pollutant','operationalization']:
        selected = pairs.filter((pl.col('kind') == kind) & pl.col('plot')).to_dicts()
        views = ['scatter','bland_altman'] if kind == 'source' else ['scatter']
        for view in views:
            fig,axes = panel_grid(len(selected),width=4.4,height=3.6,cols=2 if len(selected)==4 else 3)
            for ax,row in zip(axes,selected):
                grid = pl.read_parquet(figures.root/'cache'/'plot_data'/f"{row['id']}.parquet").filter(pl.col('view') == view)
                draw_density(ax,grid)
                if kind == 'source':
                    ax.set_title(row['a_pollutant']+' / '+label(row['a_spatial_method'])+'\n'+title_stats(row),fontsize=8)
                    if view == 'scatter':
                        ax.set_xlabel('EAC4 (µg/m³)'); ax.set_ylabel('LMN (µg/m³)')
                        if grid.height:
                            lo = min(grid['x_min'][0],grid['y_min'][0]); hi = max(grid['x_max'][0],grid['y_max'][0])
                            ax.plot([lo,hi],[lo,hi],color='#b07050',ls='--',lw=1)
                            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
                    else:
                        ax.set_xlabel('Mean of LMN and EAC4 (µg/m³)'); ax.set_ylabel('LMN − EAC4 (µg/m³)')
                        ax.axhline(0,color='#b07050',ls='--',lw=.8)
                        if finite(row['mean_difference']):
                            ax.axhline(row['mean_difference'],color=COLORS[1],lw=1)
                elif kind == 'pollutant':
                    ax.set_title(row['a_source']+' / '+label(row['a_spatial_method'])+'\n'+title_stats(row),fontsize=8)
                    ax.set_xlabel('PM2.5 (µg/m³)'); ax.set_ylabel('PM10 (µg/m³)')
                else:
                    ax.set_title(label(row['a_time'])+'\n'+title_stats(row),fontsize=8)
                    ax.set_xlabel('Inversion share'); ax.set_ylabel('Inversion strength (K)')
            sample = settings['plot_sample_size']
            fig.suptitle(kind.replace('_',' ').title()+' comparison: '+view.replace('_',' ')+
                         (f'\nDisplay sample capped at {sample:,}; correlations use full overlap' if sample else '\nFull-data aggregated 2D counts'))
            figures.save(fig,SECTIONS[0] if kind == 'operationalization' else SECTIONS[1],f'{kind}_{view}',
                         'Binned 2D counts; '+view,'; '.join(r['a']+' / '+r['b'] for r in selected),'cache/plot_data/<pair id>.parquet; cache/pair_summary.parquet')
        if kind == 'pollutant':
            fig,axes = panel_grid(len(selected),width=4.2,height=3)
            for ax,row in zip(axes,selected):
                path = figures.root/'cache'/'plot_data'/f"{row['id']}_ratio.parquet"
                ratio = pl.read_parquet(path)
                info = pl.read_parquet(path.with_name(f"{row['id']}_ratio_info.parquet")).row(0,named=True)
                ax.step(ratio['ratio'],ratio['cdf'],where='post')
                ax.axvline(1,lw=.8,color='#aaaaaa'); ax.set_xlabel('PM2.5 / PM10'); ax.set_ylabel('Cumulative proportion')
                ax.set_title(row['a_source']+' / '+label(row['a_spatial_method'])+f"\nPM10=0: {info['zero_denominator_n']:,}; PM10<0: {info['negative_denominator_n']:,}",fontsize=8)
            fig.suptitle('PM2.5 / PM10: finite nonzero denominators, including negatives\nZero-denominator ratios undefined; full ratio range retained')
            figures.save(fig,SECTIONS[1],'pollutant_ratios','Ratio ECDF with denominator exclusions explicitly counted',
                         'source × spatial method','cache/plot_data/*_ratio*.parquet')


def plot_association_heatmaps(figures,association):
    matched = association.filter(pl.col('dimension') == 'matched')
    for source,pollutant in itertools.product(['EAC4','LMN'],['PM10','PM2.5']):
        data = matched.filter((pl.col('pollution_source') == source) & (pl.col('pollution_pollutant') == pollutant))
        if not data.height:
            continue
        methods = ordered(data['inversion_spatial_method'].to_list(),'spatial_method')
        for metric in ['pearson','spearman']:
            fig,axes = panel_grid(2*len(methods),width=4.5,height=3.5,cols=len(methods))
            for ax,(op,method) in zip(axes,itertools.product(ORDERS['operationalization'],methods)):
                part = data.filter((pl.col('inversion_operationalization') == op) & (pl.col('inversion_spatial_method') == method))
                rows = ordered(part['inversion_construction'].to_list(),'construction')
                cols = ordered(part['inversion_time'].to_list(),'time')
                if not rows or not cols:
                    ax.set_visible(False); continue
                matrix = np.full((len(rows),len(cols)),np.nan)
                for r in part.to_dicts():
                    matrix[rows.index(r['inversion_construction']),cols.index(r['inversion_time'])] = r[metric]
                heatmap(ax,matrix,rows,cols,label(op)+'\n'+label(method))
            fig.suptitle(f'{source} {pollutant}: raw {metric.title()} correlation\nSpatially matched; each cell uses its own observed overlap')
            figures.save(fig,SECTIONS[3] if metric == 'pearson' else SECTIONS[2],f'association_{source}_{pollutant}_{metric}',
                         'Spatially matched raw inversion–pollution correlation',
                         f'{source}; {pollutant}; construction × timing × operationalization × spatial method',
                         '03_inversion_pollution_bivariate/association_summary.parquet')


def association_sweep(association,baseline,dimension):
    part = association.filter(pl.col('dimension') == 'matched')
    for dim in ['construction','time','spatial_method']:
        if dim != dimension:
            part = part.filter(pl.col('inversion_'+dim) == baseline[dim])
    for dim in ['source','pollutant']:
        if dim != dimension:
            part = part.filter(pl.col('pollution_'+dim) == baseline[dim])
    return part


def plot_dimension_sensitivity(figures,association,baseline):
    for dimension in ['time','construction','operationalization','source','spatial_method']:
        data = association_sweep(association,baseline,dimension)
        ops = ['both'] if dimension == 'operationalization' else ORDERS['operationalization']
        fig,axes = panel_grid(len(ops),width=6,height=4,cols=2)
        field = ('pollution_' if dimension == 'source' else 'inversion_')+dimension
        for ax,op in zip(axes,ops):
            part = data if op == 'both' else data.filter(pl.col('inversion_operationalization') == op)
            values = ordered(part[field].to_list(),dimension)
            for j,metric in enumerate(['pearson','spearman']):
                y = []
                for value in values:
                    cell = part.filter(pl.col(field) == value)
                    if cell.height > 1:
                        raise ValueError('Sensitivity figure mixes held dimensions')
                    y.append(cell[metric][0] if cell.height else np.nan)
                ax.plot(range(len(values)),y,'o-',ms=4,label=metric.title(),lw=1)
            ax.set_xticks(range(len(values)),[label(v) for v in values],rotation=35,ha='right',fontsize=8)
            ax.set_ylim(-1,1); ax.set_ylabel('Raw correlation'); ax.legend(fontsize=8)
            ax.set_title('Share versus strength' if op == 'both' else label(op))
        fig.suptitle('Vary '+dimension.replace('_',' ')+': '+reference_caption(baseline)+
                     '\nReference pollution: '+baseline['source']+' '+baseline['pollutant']+'; spatial methods matched',fontsize=10)
        figures.save(fig,SECTIONS[3],f'association_sensitivity_{dimension}','Change one measurement dimension at a time; pair-specific overlap',
                     json.dumps(baseline)+'; vary '+dimension,'03_inversion_pollution_bivariate/association_summary.parquet')


def plot_binned_associations(figures,association,baseline):
    for dimension in ['construction','time','spatial_method','source','pollutant']:
        data = association_sweep(association,baseline,dimension).filter(pl.col('plot'))
        fig,axes = panel_grid(2,width=6,height=5.8,cols=2)
        field = ('pollution_' if dimension in ['source','pollutant'] else 'inversion_')+dimension
        for ax,op in zip(axes,ORDERS['operationalization']):
            part = data.filter(pl.col('inversion_operationalization') == op)
            for value in ordered(part[field].to_list(),dimension):
                row = part.filter(pl.col(field) == value).row(0,named=True)
                bins = pl.read_parquet(figures.root/'cache'/'plot_data'/f"{row['id']}.parquet")
                ax.plot(bins['x_mean'],bins['y_mean'],'o-',ms=3,lw=1,
                        label=label(value)+'\n'+title_stats(row))
            ax.set_xlabel(label(op)); ax.set_ylabel('Mean pollution (µg/m³)')
            ax.set_title(label(op))
            ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
        fig.suptitle('Raw binned means; vary '+dimension.replace('_',' ')+'\n'+reference_caption(baseline)+
                     '\nReference pollution: '+baseline['source']+' '+baseline['pollutant']+'; methods matched',fontsize=10)
        figures.save(fig,SECTIONS[2],f'binned_{dimension}','Full-overlap binned mean pollution against inversion measure',
                     json.dumps(baseline)+'; vary '+dimension,'cache/plot_data/<pair id>.parquet; association_summary.parquet')


# %% Run and reproducibility
DEFAULTS = {'batch_size':8, 'plot_sample_size':0, 'seed':20260922, 'cross_spatial':False,
            'reference_construction':'925-1000', 'reference_time':'dawn_local',
            'reference_spatial_method':'IDW_4_nearest', 'reference_source':'EAC4', 'reference_pollutant':'PM2.5'}


def configured_path(value, config_path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent/path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--catalog-only', action='store_true')
    parser.add_argument('--plots-only', action='store_true', help='Re-render cached tables without reading measurement data')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--row-limit', type=int, help='Smoke test only; isolated output under smoke_<N> (first N rows)')
    args = parser.parse_args()
    with args.config.open() as f:
        config = json.load(f)
    settings = {**DEFAULTS, **config.get('descriptives_settings',{})}
    if settings['batch_size'] < 2 or settings['plot_sample_size'] < 0:
        raise ValueError('batch_size must be >=2 and plot_sample_size >=0')
    if args.row_limit is not None and args.row_limit <= 0:
        raise ValueError('row-limit must be positive')
    input_path = (configured_path(config['descriptives_input'],args.config) if 'descriptives_input' in config
                  else configured_path(config['merged'],args.config)/'df_daily.parquet')
    root = configured_path(config['descriptives'],args.config)
    if args.row_limit:
        root = root/f'smoke_{args.row_limit}'
        print(f'SMOKE TEST: first {args.row_limit:,} rows only; outputs isolated at {root}',flush=True)
    root.mkdir(parents=True,exist_ok=True)
    for section in SECTIONS:
        (root/section).mkdir(exist_ok=True)
    # The input stat, schema, software versions, settings and this script's digest
    # prevent stale caches from masquerading as results from a changed analysis.
    stat = input_path.stat()
    source = pl.scan_parquet(input_path)
    schema = source.collect_schema()
    catalog,excluded = build_variable_catalog(schema)
    print_catalog(catalog,excluded)
    catalog.write_parquet(root/'variable_catalog.parquet')
    catalog.write_csv(root/'variable_catalog.csv')
    if excluded.height:
        excluded.write_csv(root/'excluded_columns.csv')
    if args.catalog_only:
        return
    if not {'gemeentecode','date','gemeentenaam'}.issubset(schema.names()):
        raise ValueError('Expected municipality-day identifiers are absent')
    if args.row_limit:
        source = source.head(args.row_limit)
    manifest = {'version':VERSION, 'input':str(input_path.resolve()), 'input_size':stat.st_size,
                'input_mtime_ns':stat.st_mtime_ns, 'schema':{k:str(v) for k,v in schema.items()},
                'settings':settings, 'catalog':catalog.to_dicts(), 'row_limit':args.row_limit, 'polars':pl.__version__,
                'numpy':np.__version__, 'calculation_sha256':hashlib.sha256(''.join(inspect.getsource(obj) for obj in
                    [build_variable_catalog, Analysis, make_tasks, bin_association, density_grid,
                     attach_pair_metadata, build_association_summary]).encode()).hexdigest()}
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Cache fingerprint changed. Choose a fresh descriptives directory in config (or archive the old output directory) before recalculating.')
    if args.plots_only and not manifest_path.exists():
        raise ValueError('No completed cache exists for --plots-only')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    analysis = Analysis(source,catalog,root,settings)
    tasks,baseline = make_tasks(catalog,settings)
    # Never silently substitute a reference choice when its definition is absent.
    for family,dimensions in [('inversion',['construction','time','spatial_method']),('pollution',['source','pollutant','spatial_method'])]:
        subset = catalog.filter(pl.col('family') == family)
        for dim in dimensions:
            subset = subset.filter(pl.col(dim) == baseline[dim])
        if not subset.height:
            raise ValueError(f'Reference {family} definition unavailable; adjust descriptives_settings explicitly')
    if not args.plots_only:
        print(f'Calculating univariate summaries and {len(tasks):,} exact pair comparisons.',flush=True)
        summary = analysis.summarize()
        pairs_path = analysis.cache/'pair_summary.parquet'
        if pairs_path.exists():
            pairs = pl.read_parquet(pairs_path)
        else:
            pairs = attach_pair_metadata(analysis.pairs(tasks),catalog)
            save_table(pairs,pairs_path)
        association = build_association_summary(pairs)
        save_table(association,root/SECTIONS[2]/'association_summary.parquet')
        save_table(summary.filter(pl.col('family') == 'inversion'),root/SECTIONS[0]/'inversion_summary.parquet')
        save_table(summary.filter(pl.col('family') == 'pollution'),root/SECTIONS[1]/'pollution_summary.parquet')
    if not args.no_plots:
        print('Rendering figures from cached summaries.',flush=True)
        summary = pl.read_parquet(analysis.cache/'univariate.parquet')
        pairs = pl.read_parquet(analysis.cache/'pair_summary.parquet')
        association = pl.read_parquet(root/SECTIONS[2]/'association_summary.parquet')
        ecdf,monthly = [pl.read_parquet(analysis.cache/f'{name}.parquet') for name in ['ecdf','monthly']]
        figures = Figures(root)
        for description, draw in [
            ('Inversion distributions and seasonality',lambda:plot_inversion_univariate(figures,summary,ecdf,monthly,baseline)),
            ('Measurement agreement',lambda:plot_agreement(figures,pairs,baseline)),
            ('Pollution distributions and source summary',lambda:plot_pollution(figures,summary,ecdf,monthly,pairs)),
            ('Binned source and operationalization comparisons',lambda:plot_pair_grids(figures,pairs,settings)),
            ('Matched association heatmaps',lambda:plot_association_heatmaps(figures,association)),
            ('Association sensitivity',lambda:plot_dimension_sensitivity(figures,association,baseline)),
            ('Binned associations',lambda:plot_binned_associations(figures,association,baseline))]:
            print('  '+description,flush=True)
            draw()
            figures.finish()  # Retain an index even if a later rendering is interrupted.
        print(f"Saved {len(figures.index)//2} figures (PNG + PDF) and figure_index.csv.",flush=True)
    (root/'METHODS.md').write_text(METHODS_NOTE+'\n\nReference settings:\n```json\n'+json.dumps(baseline,indent=2)+'\n```\n')
    print(f'Done: {root}',flush=True)


METHODS_NOTE = """# Descriptive methods

All summaries weight observed municipality-days equally. Monthly means pool
calendar months across all years; they are not balanced municipality/year means.
Each pair uses its own finite observed overlap; differences between correlations
can therefore reflect both measurement and sample composition. No regressions,
residualization, fixed effects, statistical confidence intervals, or causal effects.
Standard deviations use ddof=1; summary quantiles use linear interpolation.
Spearman correlations use exact average ranks computed on each pair's overlap.
Constant variables and overlaps too small for correlation yield undefined values,
shown blank in heatmaps. No synthetic diagonals are inserted in agreement matrices.

EAC4 and LMN concentrations are in µg/m³; EAC4 is converted from kg/m³ during
extraction and LMN is already in µg/m³, as documented in utilities/lmn.py. Temperature differences are Kelvin,
equivalent to Celsius differences. Raw signed t_dif, PM1, and UTC definitions are
explicitly excluded. The indicator-derived measure is called inversion share.
Current aggregate.py averages hourly shares; historical file provenance cannot
be recovered from column names alone. Older files may use different reductions.
The unqualified t_dif uses a pressure level selected from surface pressure, as in
extract.py; it is not a fixed-pressure-level construction. area means polygon-area
weighted zonal mean; idw uses four nearest donors; idw_60km uses a 60 km radius.
Unavailable definitions are reported, never synthesized. Unparsed environmental
columns are listed separately with their reason. Finite suspicious values remain.
Null/NaN/infinite measurements are excluded from numerical summaries; per-variable
counts describe this exclusion, not a general data-quality audit.

Positive-only strength summaries condition on strength > 0. Unconditional
summaries include zeros. ECDFs retain the full range and use up to 201 order
statistics with exact cumulative counts at their values (a compact approximation
between the displayed points). Means with p25–p75 bars are descriptive distribution
intervals, not confidence intervals. Source statistics define A=LMN and B=EAC4;
mean/median differences and MAE/RMSE use only shared municipality-days. Other
same-unit differences are A minus B, with both variable names saved explicitly.
PM2.5/PM10 includes negative denominators; zero-denominator ratios are undefined
and their counts are saved and displayed. Full ratio tails, including negative
and above-one values, are retained.

Binned association means use all paired data. Shares with at most two unique
values use their actual values as categories; otherwise 20 equal-width bins span
0–1, expanded if observed values fall outside that range. Empty bins are omitted.
Strength uses zero separately and up to 15 positive quantile bins (duplicate edges
removed); unexpected negative values get a separate category. Bin counts and
observed endpoints are cached. 2D density plots aggregate full data into 40×40
counts by default; optional seeded sampling applies only to those display grids
only, never ratio distributions or reported pair statistics. Every density grid retains
its complete display-data range. Counts are colored on a log scale.

All eligible inversion comparisons differing in exactly one dimension are saved.
The default association table includes every spatially matched inversion ×
pollution pairing; optional cross_spatial also calculates mismatched pairs, which
never enter main figures. The figures use predeclared reference sweeps plus
all-definition summary plots/heatmaps. Other choices are held fixed when a single
dimension varies; spatial sensitivity changes inversion and pollution assignment
together to retain matching. The two operationalizations remain separately
labelled because their units differ. Pair-specific N is available in all tables.
"""

if __name__ == '__main__':
    main()
