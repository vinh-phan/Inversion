import cdsapi

###########################################
################## Temperature download
def temp_download_cds(location, month, year, times, output_path, key):
    ''' Function to download temperature data from cds as NetCDF file.
        Saves: temperature data for %location% for %month% of %year% to {location}/ERA5_temp_{location}_{year}_{month}.nc

    Parameters:
    -----------
    location : str
        Location to be downloaded
    
    month : str
        Month to be downloaded
        format: mm (e.g. 01, 02, ...)

    year : str
        Year to be downloaded
        format: yyyy (e.g. 2020, 2021, ...)
    
    locations : dict
        Dictionary containing the bounding box for each location.

    times : dict
        Dictionary containing the time range to be extracted, for each location.

    output_path : str
        Path to save the downloaded NetCDF files.

    key: str
        Key to access ERA5 API

    Returns:
    --------
    None
        Saves the downloaded NetCDF file to the specified path.
    '''
    print(f"-------------------- Downloading temperature data for Month {month} of Year {year} --------------------")

    # define the dataset and request body
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature"],
        "year": [
            year
        ],
        "month": month,

        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"],

        "time": times,
        
        "pressure_level": [
            "700", "750", "775", 
            "800", "825", "850", 
            "875", "900", "925", 
            "950", "975", "1000"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": location
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/ERA5_temp_{year}_{month}.nc")
    print(f"===== Finished downloading temperature data for Month {month} of Year {year}. Saving as ERA5_temp_{location}_{year}_{month}.nc")


###########################################
################## Weather download (excluding total precipitation)
def weather_download_cds(location, month, year, times, output_path, key):
    ''' Function to download weather data from cds.

    Parameters:
    -----------
    location : str
        Location to be downloaded
    
    year : str
        Year to be downloaded
        format: yyyy (e.g. 2020, 2021, ...)
    
    locations : dict
        Dictionary containing the bounding box for each location.

    times : dict
        Dictionary containing the time range to be extracted, for each location.

    output_path : str
        Path to save the downloaded NetCDF files.

    Returns:
    --------
    None
        Saves the downloaded ZIP file to the specified path.
    '''
    print(f"-------------------- Downloading weather data for Year {year} --------------------")

    # define the dataset and request body
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_pressure",
            "total_cloud_cover"
        ],
        "month": month
        ,
        "year": year,
        
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": times,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": location
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/ERA5_weather_{year}.nc") 
    print(f"===== Finished downloading weather data for Year {year}. Saving to /{location}/ERA5_weather_{year}.nc")


###########################################
################## Precipitation download 
def precipitation_download_cds(location, month, year, times, output_path, key):
    ''' Function to download precipitation data from cds.

    Parameters:
    -----------
    location : str
        Location to be downloaded
    
    year : str
        Year to be downloaded
        format: yyyy (e.g. 2020, 2021, ...)
    
    locations : dict
        Dictionary containing the bounding box for each location.

    times : dict
        Dictionary containing the time range to be extracted, for each location.

    output_path : str
        Path to save the downloaded NetCDF files.

    Returns:
    --------
    None
        Saves the downloaded ZIP file to the specified path.
    '''
    print(f"-------------------- Downloading precipitation data for Year {year} --------------------")

    # define the dataset and request body
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "total_precipitation"
        ],
        "month": month
        ,
        "year": year,
        
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": times,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": location
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/ERA5_precip_{year}.nc") 
    print(f"===== Finished downloading precipitation data for Year {year}. Saving to /{location}/ERA5_precip_{year}.nc")


###########################################
################## Create worker wrapper (to help with parallel processing with API keys)
def worker(data_type, year, month, key, location, times, output_folder):
    if data_type == "temperature":
        temp_download_cds(
            location=location,
            month=month,
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "weather":
        weather_download_cds(
            location=location,
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "precipitation":
        precipitation_download_cds(
            location=location,
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )




