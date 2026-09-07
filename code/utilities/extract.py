import polars as pl
import netCDF4
import numpy as np
import multiprocessing


def _time_slice_indices(valid_time, from_time, to_time):
    valid_time = np.asarray(valid_time)
    from_index = np.searchsorted(valid_time, from_time, side="left")
    to_index = np.searchsorted(valid_time, to_time, side="right")

    if from_index >= to_index:
        raise ValueError(
            f"No valid_time values found between {from_time} and {to_time}."
        )

    return from_index, to_index


def extract_and_compute(filepath_temperature, filepath_weather, filepath_precipitation, cut_time = True):
    ''' Function to merge temperature, weather, and precipitation NETCDF files.
        Computes new columns based on given R code.

    Parameters:
    -----------
    filepath_temperature : str
        Filepath to the temperature netCDF file.
    
    filepath_weather : str
        Filepath to the weather netCDF file.

    filepath_precipitation : str
        Filepath to the precipitation netCDF file.

    cut_time : bool
        If True, the function will cut the time to the same range for both files. 
        If False, the function will use the full time range of the weather file.
        (experimental)
    Returns:
    --------
    df_combined : polars.DataFrame
        A Polars DataFrame containing the computed columns and the original columns from the netCDF files.
    '''

    # ----------------- Temperature data -----------------

    # Read temperature data
    nc = netCDF4.Dataset(filepath_temperature, mode='r')

    # Extract variables from temperature file
    lat = nc.variables['latitude'][:]  # 27
    lon = nc.variables['longitude'][:]  # 33
    valid_time = nc.variables['valid_time'][:]  # 744
    level = nc.variables['pressure_level'][:]  
    n_levels = len(level)
    t_arr_flat = nc.variables['t'][:,:n_levels].flatten() 
    shape = list(nc.variables['t'].shape)
    shape[1] = n_levels 
    multi_index = np.indices(shape).reshape(len(shape), -1).T

    # Extract min and max times of temperature for limited loading of weather data
    if cut_time:
        from_time = nc.variables['valid_time'][:].min()
        to_time = nc.variables['valid_time'][:].max()
    else:
        None

    nc.close()

    # Create temperature Polars DataFrame
    df_temperature = pl.DataFrame({
        "time": pl.Series((valid_time[multi_index[:, 0]]*1000), dtype=pl.Datetime('ms')),  # Cast time to UInt32 to save space
        "level": pl.Series(level[multi_index[:, 1]], dtype=pl.UInt16),
        "lat": pl.Series((lat[multi_index[:, 2]].filled(0)), dtype=pl.Float32),
        "lon": pl.Series((lon[multi_index[:, 3]].filled(0)), dtype=pl.Float32),
        "Temperature": pl.Series(t_arr_flat, dtype=pl.Float32)
    })

    # ----------------- Weather data -----------------
    # Read weather data
    nc = netCDF4.Dataset(filepath_weather, mode='r')

    # Load full time column
    valid_time = nc.variables['valid_time'][:] #744

    # Extract index location of time range based on min/max of temperature data
    # (Only extract weather data for time range that is needed)
    if cut_time:
        from_index, to_index = _time_slice_indices(valid_time, from_time, to_time)
    else:
        from_index = None
        to_index = None

    # Extract variables from weather file
    lat = nc.variables['latitude'][:] #27
    lon = nc.variables['longitude'][:] #33
    valid_time = nc.variables['valid_time'][from_index:to_index] #744
    t2m_arr_flat = nc.variables['t2m'][from_index:to_index].flatten()
    sp_arr_flat = nc.variables['sp'][from_index:to_index].flatten()
    u10_arr_flat = nc.variables['u10'][from_index:to_index].flatten()
    v10_arr_flat = nc.variables['v10'][from_index:to_index].flatten()
    tcc_arr_flat = nc.variables['tcc'][from_index:to_index].flatten()
    d2m_arr_flat = nc.variables['d2m'][from_index:to_index].flatten()

    # Create multi_index for the dataframe
    multi_index = np.indices(nc.variables['sp'][from_index:to_index].shape).reshape(len(nc.variables['sp'][from_index:to_index].shape), -1).T

    nc.close()

    # Create weather Polars DataFrame
    df_weather = pl.DataFrame({
        "time": pl.Series((valid_time[multi_index[:, 0]]*1000), dtype=pl.Datetime('ms')), # Cast time to UInt32 to save space
        "lat": pl.Series((lat[multi_index[:, 1]].filled(0)), dtype=pl.Float32), 
        "lon": pl.Series((lon[multi_index[:, 2]].filled(0)), dtype=pl.Float32),
        "t2m": pl.Series(t2m_arr_flat, dtype=pl.Float32),
        "sp": pl.Series(sp_arr_flat, dtype=pl.Float32),
        "u10": pl.Series(u10_arr_flat, dtype=pl.Float32),
        "v10": pl.Series(v10_arr_flat, dtype=pl.Float32),
        "tcc": pl.Series(tcc_arr_flat, dtype=pl.Float32),
        "d2m": pl.Series(d2m_arr_flat, dtype=pl.Float32),
    })

    # ----------------- Total Precipitation data -----------------
    # Read weather data
    nc = netCDF4.Dataset(filepath_precipitation, mode='r')

    # Load full time column
    valid_time = nc.variables['valid_time'][:] #744

    # Extract index location of time range based on min/max of temperature data
    # (Only extract weather data for time range that is needed)
    if cut_time:
        from_index, to_index = _time_slice_indices(valid_time, from_time, to_time)
    else:
        from_index = None
        to_index = None

    # Extract variables from weather file
    lat = nc.variables['latitude'][:] #27
    lon = nc.variables['longitude'][:] #33
    valid_time = nc.variables['valid_time'][from_index:to_index]
    tp_arr_flat = nc.variables["tp"][from_index:to_index].flatten()

    # Create multi_index for the dataframe
    multi_index = np.indices(nc.variables['tp'][from_index:to_index].shape).reshape(len(nc.variables['tp'][from_index:to_index].shape), -1).T

    nc.close()

    # Create weather Polars DataFrame
    df_precipitation = pl.DataFrame({
        "time": pl.Series((valid_time[multi_index[:, 0]]*1000), dtype=pl.Datetime('ms')), # Cast time to UInt32 to save space
        "lat": pl.Series((lat[multi_index[:, 1]].filled(0)), dtype=pl.Float32), 
        "lon": pl.Series((lon[multi_index[:, 2]].filled(0)), dtype=pl.Float32),
        "tp": pl.Series(tp_arr_flat, dtype=pl.Float32)
    })

    # join precip and weatherdata
    df_weather = df_weather.join(df_precipitation, on=["lat", "lon", "time"], how="inner")

    # ----------------- Compute pressure levels for comparison -----------------

    # -------------------------------------------------------------------------------- Important: lacking bin count for pressure levels in China and Peru

    '''
    If 800 > sp >= 775: temp @ 750 - t2m
    If 775 > sp >= 750: temp @ 700 - t2m
    If 750 > sp >= 700: temp @ 650 - t2m
    If 700 > sp >= 650: temp @ 600 - t2m
    If 650 > sp >= 600: temp @ 550 - t2m
    If 600 > sp >= 550: temp @ 500 - t2m
    If 550 > sp >= 500: temp @ 450 - t2m
    If 500 > sp >= 450: temp @ 400 - t2m
    If 450 > sp >= 400: temp @ 350 - t2m
    If 400 > sp >= 350: temp @ 300 - t2m
    If 350 > sp >= 300: temp @ 250 - t2m
    If 300 > sp >= 250: temp @ 225 - t2m
    If 250 > sp >= 225: temp @ 200 - t2m
    '''
    # Define the bins and labels
    bins_np = np.array([
        22500, 25000, 30000, 
        35000, 40000, 45000, 
        50000, 55000, 60000, 
        65000, 70000, 75000, 
        77500, 80000, 82500,
        85000, 87500, 90000, 
        92500, 95000, 97500, 
        99000, 101000, float('inf')])

    labels_np = np.array([
        200, 225, 250, 
        300, 350, 400, 
        450, 500, 550, 
        600, 650, 700, 
        750, 775, 800, 
        825, 850, 875, 
        900, 925, 950, 
        975, 1000])

    # Assign labels based on bins using NumPy digitize
    pressure_indices = np.digitize(df_weather["sp"].to_numpy(), bins_np, right=False) - 1

    # Create array of labels for all rows
    pressure_labels = labels_np[pressure_indices]

    # Add level column to the weather dataframe based on computed pressure labels
    df_weather = df_weather.with_columns(
        pl.Series("level", pressure_labels).cast(pl.UInt16)  # Cast to string if necessary
    )

    # Find temperature of associated level for comparison
    # (Joins the dataframes on time, lat, lon and level)
    df_combined = df_temperature.join(
        df_weather,
        on=["time", "lat", "lon", "level"],  # Columns to join on
        how="inner"
    )

    # ----------------- Add/Transform variables of interest -----------------

    # Columns for polynomial transformation
    poly_columns = ['sp', 'tp', 'u10', 'v10', 'tcc', 'd2m', 'ws', 'wd']

    # Convert pressure, create wind speed and wind direction
    df_combined = df_combined.with_columns([
        (pl.col("sp") / 100).alias("sp"),       # Convert pressure from Pa to hPa
        (pl.col("u10")**2 + pl.col("v10")**2).sqrt().alias("ws"), # Create wind speed
        (pl.col("t2m") - 273.15).alias("t2m"),
        (pl.col("Temperature") - 273.15).alias("Temperature"),
        ((3 * np.pi / 2 - pl.arctan2(pl.col("v10"), pl.col("u10"))) % (2 * np.pi)).alias("wd") # Create wind direction
        ])

    # Create thermal inversion indicators and add polynominal tranformations
    df_combined = df_combined.with_columns([
        (pl.col("Temperature") - pl.col("t2m")).alias("t_dif"),
        ((pl.col("Temperature") - pl.col("t2m")) > 0).cast(pl.Int8).alias("t_dif_indicator")     # Create the indicator column, 1 if t_dif > 0, else 0
    ] + [
        (pl.col(col_name) ** 2).alias(f"{col_name}_squared")     # Create a new column with the cubed value
        for col_name in poly_columns

    ] + [
        (pl.col(col_name) ** 3).alias(f"{col_name}_cubed")  # Create a new column with the squared value
        for col_name in poly_columns
    ])

    # Remove original temperature and level columns
    df_combined = df_combined.drop(['level'])

    # Return final dataframe
    return(df_combined)


def extract_weather(ERA5_folder, out_folder, years, num_processors=2):
    print("Begin processing...")

    months = [f"{m:02d}" for m in range(1, 13)]
    all_outputs = []

    for year in years:
        print(f"\n=== Processing year {year} ===")
        file_pairs = [
            (f"{ERA5_folder}/ERA5_temp_{year}_{month}.nc",
             f"{ERA5_folder}/ERA5_weather_{year}.nc", 
             f"{ERA5_folder}/ERA5_precip_{year}.nc", 
             )
            for month in months
        ]

        with multiprocessing.Pool(processes=num_processors) as p:
            output = p.starmap(extract_and_compute, file_pairs)

        # Concatenate results for that year
        df_year = pl.concat(output)
        output = None  # free memory

        # Write to separate parquet file for each year
        outpath = f"{out_folder}/ERA5_unmatched_{year}.parquet"
        print(f"Writing {year} to {outpath}")
        df_year.write_parquet(outpath)

        all_outputs.append(outpath)
        df_year = None  # free memory

    print("All years processed and written to disk.")
    return all_outputs


#####################################################
################## Extract pollution data 
def extract_pollution(nc_path, source = "EAC4", lat_min = None, lat_max = None, lon_min = None, lon_max = None):
    if source == "WashU": 
        # Read air pollution data 
        nc = netCDF4.Dataset(nc_path, mode='r')

        # Extract lat and lon variables into arrays
        lat = nc.variables['lat'][:] 
        lon = nc.variables['lon'][:] 

        # Get the index position of the filtered lat/lon values 
        lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

        # Get and slice the PM25 array 
        pm25 = nc.variables['PM25'][lat_indices.min():lat_indices.max()+1,
                                        lon_indices.min():lon_indices.max()+1]

        # Get the actual lat/lon values 
        lat = lat[lat_indices.min():lat_indices.max()+1]
        lon = lon[lon_indices.min():lon_indices.max()+1]

        # Create multi_index for the dataframe
        shape = list(pm25.shape)
        multi_index = np.indices(shape).reshape(len(shape), -1).T

        # Flatten PM2.5
        pm25_flat = pm25.flatten()

        # Create air pollution polars dataframe 
        df_pm25 = pl.DataFrame({
            "lat": pl.Series(lat[multi_index[:, 0]], dtype=pl.Float32),
            "lon": pl.Series(lon[multi_index[:, 1]], dtype=pl.Float32),
            "PM25": pl.Series(pm25_flat, dtype=pl.Float32)
        })

        # Add in time column 
        str_time = nc_path[-9:-3]
        time_value = datetime.strptime(str_time, "%Y%m")
        df_pm25 = df_pm25.with_columns(
            pl.lit(time_value).cast(pl.Datetime).alias("time")
        )

        return df_pm25
    
    elif source == "EAC4":
        # Read air pollution data 
        nc = netCDF4.Dataset(nc_path, mode='r')
        # Extract lat and lon variables into arrays
        lat = nc.variables['latitude'][:] 
        lon = nc.variables['longitude'][:] 

        # Change scale of lon from -180 to 180 (instead of 0 to 360)
        lon = (lon + 180) % 360 - 180           

        # Get the index position of the filtered lat/lon values 
        lat_indices = np.where((lat >= min(lat)) & (lat <= max(lat)))[0]
        lon_indices = np.where((lon >= min(lon)) & (lon <= max(lon)))[0]

        # Get dimension names
        dim_names = set(nc.dimensions.keys())
        data_vars = [v for v in nc.variables.keys() if v not in dim_names]

        # Get and slice the variable arrays 
        extracted_var = {}
        for var in data_vars:
            arr = nc.variables[var][:, lat_indices, lon_indices]
            extracted_var[var] = arr.flatten()

        # Get the actual lat/lon values and time dimension
        lat = lat[lat_indices]
        lon = lon[lon_indices]
        valid_time = nc.variables['valid_time'][:]

         # Create multi_index for the dataframe
        multi_index = np.indices(nc.variables[data_vars[0]][:, lat_indices, lon_indices].shape).reshape(len(nc.variables[data_vars[0]][:, lat_indices, lon_indices].shape), -1).T

         # Create air pollution polars dataframe 
        df_pm_EAC = pl.DataFrame({
            "time": pl.Series((valid_time[multi_index[:, 0]]*1000), dtype=pl.Datetime('ms')),
            "lat": pl.Series(lat[multi_index[:, 1]], dtype=pl.Float32),
            "lon": pl.Series(lon[multi_index[:, 2]], dtype=pl.Float32),
            **{
            var: pl.Series(values, dtype=pl.Float32)
            for var, values in extracted_var.items()
        }
        })

        return df_pm_EAC
