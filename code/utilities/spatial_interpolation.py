"""
Shared spatial interpolation of gridded data to Dutch municipalities. 

Created with Codex
"""

import geopandas as gpd
import numpy as np
import polars as pl
from pathlib import Path
from scipy.sparse import csr_matrix
from shapely.geometry import box


IDW_NEIGHBORS = 4
IDW_RADIUS_KM = 60
IDW_POWER = 2
AREA_CRS = "EPSG:3035"  # European equal-area projection, in metres.


def read_municipalities(path):
    """Combine land/water pieces so each municipality has one geometry."""
    municipalities = gpd.read_file(
        path, layer="gemeenten", columns=["gemeentecode", "gemeentenaam"]
    ).to_crs(AREA_CRS)
    municipalities = municipalities.dissolve(by="gemeentecode", as_index=False)
    return municipalities.sort_values("gemeentecode").reset_index(drop=True)


def _sparse_nearest_weights(neighbors, weights, shape):
    """Create a sparse municipality-by-grid weight matrix."""
    rows = np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])
    return csr_matrix(
        (weights.ravel(), (rows, neighbors.ravel())),
        shape=shape,
    )


def _sparse_radius_weights(distances, radius_km, power, shape):
    """Create IDW weights from all grid points inside a distance radius."""
    rows = []
    columns = []
    values = []
    for municipality_id, municipality_distances in enumerate(distances):
        selected = np.flatnonzero(municipality_distances <= radius_km)
        if len(selected) == 0:
            continue
        selected_weights = 1 / np.maximum(
            municipality_distances[selected], 1e-12
        ) ** power
        selected_weights /= selected_weights.sum()
        rows.extend([municipality_id] * len(selected))
        columns.extend(selected)
        values.extend(selected_weights)
    return csr_matrix((values, (rows, columns)), shape=shape)


def make_weights(grid, municipalities, include_area=True):
    """Build municipality-by-grid matrices once, independently of the data."""
    # Extract latitudes and longitudes
    lat = grid["lat"].to_numpy().astype(float)
    lon = grid["lon"].to_numpy().astype(float)
    latitudes, longitudes = np.unique(lat), np.unique(lon)
    dlat, dlon = np.diff(latitudes), np.diff(longitudes)

    # Check that (1) there is more than one lat, (2) more than on long, (3+4) lat and lon spacing are regular, and (5) the grid contains every lat-lon combination
    if (len(dlat) == 0 or len(dlon) == 0
            or not np.allclose(dlat, dlat[0])
            or not np.allclose(dlon, dlon[0])
            or len(grid) != len(latitudes) * len(longitudes)):
        raise ValueError("Expected a complete, regular latitude/longitude grid.")

    #----------------------- Area weighted zonal mean ------------------------
    # Create GeoDataFrame and converts each grid point into a rectangular cell
    cells = gpd.GeoDataFrame(
        {"grid_id": np.arange(len(grid))},
        geometry=[box(x - dlon[0]/2, y - dlat[0]/2,
                      x + dlon[0]/2, y + dlat[0]/2) for y, x in zip(lat, lon)],
        crs="EPSG:4326",
    )

    # Densify edges before projection so lines of constant longitude and latitude are represented accurately after projection.
    cells.geometry = cells.geometry.segmentize(0.01)

    # Project polygons to the same equal-area projection used for municipalities.
    cells = cells.to_crs(AREA_CRS)

    # Create a copy containing only municipality geometries
    zones = municipalities[["geometry"]].copy()
    zones["municipality_id"] = np.arange(len(zones))

    # Find polygon overlaps between municipality and ERA5 cells
    overlap = gpd.overlay(zones, cells, how="intersection", keep_geom_type=False)

    # Build the area-weighted matrix. Rows = municipalities; columns = ERA5 grid cells; values = overlapping area
    area = csr_matrix(
        (overlap.area.to_numpy(),
         (overlap["municipality_id"], overlap["grid_id"])),
        shape=(len(zones), len(grid)),
    )

    # Check if spatial coverage is not close to 1
    coverage = np.asarray(area.sum(axis=1)).ravel() / municipalities.area.to_numpy()
    if include_area and np.any(coverage < 0.999):
        raise ValueError("The ERA5 cell footprint does not cover every municipality.")

    # Convert overlap areas to normalized weights
    if include_area:
        area = area.multiply(1 / np.asarray(area.sum(axis=1))).tocsr()

    #------------------------ IDW --------------------------
    # Geometric centroids in an equal-area CRS; distances are great-circle km.
    centroids = municipalities.geometry.centroid.to_crs("EPSG:4326")

    # Convert centroid latitude and longitude to radians and reshape into a column vector.
    target_lat = np.radians(centroids.y.to_numpy())[:, None]
    target_lon = np.radians(centroids.x.to_numpy())[:, None]

    # Convert ERA5 grid into radians and reshape into row vectors.
    source_lat, source_lon = np.radians(lat)[None, :], np.radians(lon)[None, :]

    # Calculate great-circle distances
    a = (np.sin((source_lat - target_lat)/2)**2
         + np.cos(target_lat) * np.cos(source_lat)
         * np.sin((source_lon - target_lon)/2)**2)

    # Converts angular distance to kilometres. Rows = municipaities, columns = ERA5 grid points, values = distance in km.
    distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Select the nearest grid points
    neighbors = np.argsort(distances, axis=1)[:, :IDW_NEIGHBORS]

    # Retain the n closest points
    selected = np.take_along_axis(distances, neighbors, axis=1)

    # Calculate weights
    weights = 1 / np.maximum(selected, 1e-12)**IDW_POWER

    # Handling exact matches
    exact = selected[:, 0] < 1e-9
    weights[exact] = 0
    weights[exact, 0] = 1

    # Normalize weights
    weights /= weights.sum(axis=1, keepdims=True)

    # Creating the sparse IDW matrix. Rows = municipalities, columns = ERA5 grid points, values = IDW weights
    idw = _sparse_nearest_weights(neighbors, weights, shape=area.shape)

    # Create IDW weights from all grid points within 60 km of each municipality centroid.
    idw_60km = _sparse_radius_weights(
        distances, IDW_RADIUS_KM, IDW_POWER, shape=area.shape
    )

    # Create nearest-neighbor weights.
    nearest_neighbors = neighbors[:, :1]
    nearest_weights = np.ones_like(nearest_neighbors, dtype=float)
    nearest = _sparse_nearest_weights(
        nearest_neighbors, nearest_weights, shape=area.shape
    )

    # Create municipality metadata
    metadata = pl.DataFrame({
        "gemeentecode": municipalities["gemeentecode"].to_list(),
        "gemeentenaam": municipalities["gemeentenaam"].to_list(),
        "centroid_lat": centroids.y.to_numpy(),
        "centroid_lon": centroids.x.to_numpy(),
        "grid_area_coverage": coverage,
    })
    matrices = {
        "idw": idw,
        "idw_60km": idw_60km,
        "nearest": nearest,
    }
    if include_area:
        matrices = {"area": area, **matrices}
    return matrices, metadata


def weighted_mean(values, weights):
    """This function renormalizes spatial weights over available observations rather than treating missing vlues as real 0's"""
    valid = np.isfinite(values)
    numerator = weights @ np.where(valid, values, 0).T
    denominator = weights @ valid.astype(float).T
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result.T  # hours x municipalities


def aggregate_month(frame, grid, matrices, metadata, variables):
    """Align rows explicitly; never assume the Parquet's grid ordering."""
    # Match weather row to its grid ID
    frame = frame.join(grid.with_row_index("grid_id"), on=["lat", "lon"], how="left")
    if frame["grid_id"].null_count() or frame.select("time", "grid_id").is_duplicated().any():
        raise ValueError("Found an unknown grid point or duplicate grid-point hour.")

    # Assign a time ID to each time
    times = frame.select("time").unique().sort("time").with_row_index("time_id")

    # Add time ID back to the monthly data
    frame = frame.join(times, on="time").sort(["time_id", "grid_id"])

    # Convert IDs intp NumPy arrays for fast matrix operations
    time_ids, grid_ids = frame["time_id"].to_numpy(), frame["grid_id"].to_numpy()

    # Create every combination of hour x municipalty
    output = times.select("time").join(metadata, how="cross")

    # Create an empty list for the calculated weather columns
    columns = []
    for variable in variables:
        # Missing rows and null values both become NaN. This also permits a
        # grid point to be missing for some hours without shifting other rows.
        values = np.full((len(times), len(grid)), np.nan)
        values[time_ids, grid_ids] = frame[variable].cast(pl.Float64).to_numpy()
        for method, weights in matrices.items():
            if variable == "wd":
                # wd is in radians: an ordinary mean of 359 and 1 degrees
                # would wrongly point south. Use a circular mean instead.
                sine = weighted_mean(np.sin(values), weights)
                cosine = weighted_mean(np.cos(values), weights)
                result = np.mod(np.arctan2(sine, cosine), 2 * np.pi)
                result[np.hypot(sine, cosine) < 1e-12] = np.nan
            else:
                result = weighted_mean(values, weights)

            # Add weighted variables to columns
            columns.append(pl.Series(
                f"{variable}_{method}_mean", result.ravel(), nan_to_null=True
            ))
    return output.with_columns(columns)


def process_files(
    input_dir,
    output_dir,
    boundaries_path,
    file_pattern,
    output_prefix,
    include_area=True,
):
    """Process any gridded Parquet dataset with time, lat, lon columns."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {input_dir}")

    first = pl.scan_parquet(files[0])
    grid = first.select("lat", "lon").unique().sort(["lat", "lon"]).collect()
    municipalities = read_municipalities(Path(boundaries_path))
    matrices, metadata = make_weights(
        grid, municipalities, include_area=include_area
    )
    print(f"Built weights: {len(grid)} grid points -> {len(municipalities)} municipalities")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        source = pl.scan_parquet(path)
        schema = source.collect_schema()
        variables = [name for name, dtype in schema.items()
                     if name not in {"time", "lat", "lon"} and dtype.is_numeric()]
        # Reuse weights only if the source grid is the same in every file.
        file_grid = source.select("lat", "lon").unique().sort(["lat", "lon"]).collect()
        if not grid.equals(file_grid):
            raise ValueError(f"Grid changed in {path.name}; rebuild spatial weights.")
        months = source.select(pl.col("time").dt.truncate("1mo")).unique().sort("time").collect()
        for month in months["time"]:
            print(f"Processing {path.name}: {month:%Y-%m}", flush=True)
            frame = source.filter(pl.col("time").dt.truncate("1mo") == month).collect()
            output = aggregate_month(frame, grid, matrices, metadata, variables)
            output.write_parquet(output_dir / f"{output_prefix}_{month:%Y_%m}.parquet")
            del frame, output
