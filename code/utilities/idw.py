import polars as pl
import numpy as np
import time
from tqdm import tqdm
import geopandas as gpd
 


def vector_haversine(xy1, xy2):
    ''' Calculate the great-circle distance between two points 

    Parameters
    ----------
    xy1 : numpy array representation of coordinates (shape of n x 2)
    xy2 : numpy array representation of coordinates (shape of m x 2)

    Output
    ------
    distance : numpy array of haversine distance between points (shape of n x m) in kilometers
                -> e.g. row 1 represents distances from xy1[0] to xy2[:] 
                        row 2 represents distances from xy1[1] to xy2[:]

    '''

    # Load vectors for lat/lon and convert to radians
    lat1 = np.radians(xy1[:, 0])
    lon1 = np.radians(xy1[:, 1])
    lat2 = np.radians(xy2[:, 0])
    lon2 = np.radians(xy2[:, 1])

    # Create difference map of x and y vectors
    dlat = np.subtract.outer(lat1, lat2)
    dlon = np.subtract.outer(lon1, lon2)

    # Run haversine formula using matrix algebra
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = (2 * 6371) * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return distance


def haversine_weights(xy1, xy2, max_distance=None, p=2, nearest_only=False):
    ''' Find haversine inverse distance weighting for xy1 and xy2
        
    Parameters
    ----------
    xy1 : represents coordinates to be interpolated
        -> numpy array of shape n x 2
    xy2 : represents coordinates to be used for interpolation
        -> numpy array of shape m x 2
    z   : represents observational values to be used for interpolation
        -> numpy array of shape m x 1
    max_distance: maximum distance of points to be used for interpolation.
        -> integer or float (represents KM)
    p   : power for observations in z to be raised to
        The larger p, the less weight is put on further away points
        -> integer value; default is 2
    Output
    ------
    zi : weighted average of observations of xy2 array, interpolated to coordinates of xy1
        -> array of shape m x 1
    weights: weight array used for computations.
        -> array of shape n x m

    '''
    if max_distance == 0:
        print('ERROR: Cannot have max_distance set to 0. leave empty for no max distance.')
        return(None, None)

    distance_matrix = vector_haversine(xy2, xy1)

    if nearest_only:
        m, n = distance_matrix.shape
        weights = np.zeros((m, n), dtype=float)

        # nearest source index for each target (column-wise argmin over rows)
        nearest_row = np.argmin(distance_matrix, axis=0)          # (n,)
        min_dist = distance_matrix[nearest_row, np.arange(n)]     # (n,)

        # optional max distance filter
        if max_distance is not None:
            ok = (min_dist <= max_distance)
        else:
            ok = np.ones(n, dtype=bool)

        # assign weight 1 only where within range
        cols_ok = np.where(ok)[0]
        if cols_ok.size:
            weights[nearest_row[cols_ok], cols_ok] = 1.0

        # columns with no ok donor remain all zeros (no weights)
        return weights

    # In IDW, weights are 1 / distance
    weights = 1.0 / (distance_matrix**p)

    # Set weights to 0 if outside of max range
    if max_distance is not None:
        comparison = 1/(max_distance**p)
        weights[weights < comparison] = 0.0

    # Create sum of weights to make weights sum to one
    weight_sum = np.sum(weights, axis=0)

    # If sum of weights is 0, set to NAN, so division is possible
    weight_sum[weight_sum == 0] = np.nan

    # Divide weights by sum of weights
    weights /= weight_sum
    return(weights)


def apply_haversine_idw(df, identifier, variables_of_interest, centroid_array, p, max_distance, nearest_only=False):
    ''' Applies haversine idw of given dataframe to centroid array (can take NAs)
        
    Parameters
    ----------
    df : dataframe 
    centroid_array : array of points to be interpolated
    Output
    ------
    zi : weighted average of observations of xy2 array, interpolated to coordinates of xy1
        -> array of shape m x 1
    weights: weight array used for computations.
        -> array of shape n x m

    '''
    if max_distance == 0:
        print('Error: Cannot have max distance of 0. Leave empty for no max distance.')
        return(None)

    # Extract unique times
    unique_times = (
        df.filter(pl.col(identifier).is_not_null())
        .select(pl.col(identifier).unique().sort())
        .to_series().to_numpy()
    )

    # centroids
    xy1 = centroid_array

    # points for interpolation
    xy2 = df.filter(pl.col(identifier) == unique_times[0]).select(['lat', 'lon']).to_numpy()

    startTime=time.time()

    # Extract all lat/lon values for every time frame
    lat_lon_extraction = df.group_by(identifier).agg([
        pl.col("lat").alias("lat"),
        pl.col("lon").alias("lon")
    ]).select([
        pl.struct(["lat", "lon"]).alias("lat_lon")
    ])["lat_lon"].to_numpy()

    # Reshape to [[lat, lon], ..., [lat, lon]]
    lat_lon_for_time = np.array([
        np.column_stack((group[0], group[1]))
        for group in lat_lon_extraction
    ], dtype=np.float32)

    # Check if observations are the same for all time periods
    if not np.all(lat_lon_for_time == lat_lon_for_time[0]):
        print('ERROR: Fast IDW not possible, data grid arrays differ between times')
        return None

    weights = haversine_weights(xy1, xy2, p=p, max_distance=max_distance, nearest_only= nearest_only)
    print("-- Computed weights")
    output_variables = {}
    print("-- Weighting variables")
    for variable in tqdm(variables_of_interest):
        extract = df.group_by(identifier, maintain_order = True).agg([
            pl.col(variable).alias(variable)
        ])[variable].to_numpy()

        extract = np.vstack(extract)        
        
        mask = ~np.isnan(extract) 
        masked = np.nan_to_num(extract, copy=True)
        num = masked @ weights
        den = mask.astype(np.float32) @ weights
        den[den == 0] = np.nan
        output_variables[variable] = (num/den).T.flatten(order = "F")
        
    print("--Variables weighted")
    # Create a Polars dataframe from the dictionary
    df_weighted_variable = pl.DataFrame({variable: output_variables[variable].astype(np.float32) for variable in variables_of_interest})

    shape = (unique_times.shape[0], xy1.shape[0])
    multi_index = np.indices(shape).reshape(len(shape), -1).T

    df_weighted_variable = df_weighted_variable.with_columns([
        pl.Series(identifier, unique_times[multi_index[:,0]]),
        pl.Series('lat', xy1[multi_index[:,1]][:, 0], dtype=pl.Float32),
        pl.Series('lon', xy1[multi_index[:,1]][:, 1], dtype=pl.Float32)
    ])
    print("-- Created df")

    desired_order = [identifier, 'lat', 'lon'] + variables_of_interest

    # Reorder the columns
    df_weighted_variable = df_weighted_variable.select(desired_order)

    # Display the updated dataframe
    endTime = time.time()
    print(f'Finished in {round(endTime - startTime, 2)}s')
    print(f'Interpolated {len(xy2)} coordinates to {len(xy1)} coordinates for ')
    print(f'{len(variables_of_interest)} columns and {len(df_weighted_variable)} rows.')
    print(f'Average time per column: {round(endTime - startTime, 2) / len(variables_of_interest)}s for {len(df_weighted_variable)} rows')
    print(f"Final dataset size: {df_weighted_variable.to_arrow().nbytes / 1000000}MBs")

    return(df_weighted_variable)


def chunk_arrays(arr1, size):
    for i in range(0, len(arr1), size):
        yield arr1[i:i+size]


def get_centroids(boundaries_path, level):
    
    if level == "LAD": 
        shapefile_path = f"{boundaries_path}/LAD/LAD_MAY_2024_UK_BUC.shp"

        # read the shapefile
        gdf = gpd.read_file(shapefile_path)

        centroid_points = gdf["geometry"].to_crs('+proj=cea').centroid.to_crs(epsg=4326)
        centroid_array = np.column_stack((centroid_points.y, centroid_points.x))

        centroid_names = gdf["LAD24NM"].to_numpy()
        centroid_codes = gdf["LAD24CD"].to_numpy()

    elif level == "ward": 
        shapefile_path = f"{boundaries_path}/ward/WD_MAY_2024_UK_BSC.shp"

        # read the shapefile
        gdf = gpd.read_file(shapefile_path)

        centroid_points = gdf["geometry"].to_crs('+proj=cea').centroid.to_crs(epsg=4326)
        centroid_array = np.column_stack((centroid_points.y, centroid_points.x))
        
        centroid_names = gdf["WD24NM"].to_numpy()
        centroid_codes = gdf["WD24CD"].to_numpy()
                    
    return(centroid_array, centroid_names, centroid_codes)