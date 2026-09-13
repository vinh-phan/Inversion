import cdsapi

###########################################
################## ERA5 temperature download
def era5_temp_download(location, month, year, times, output_path, key):
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
    print(f"===== Finished downloading temperature data for Month {month} of Year {year}. Saving as ERA5_temp_{year}_{month}.nc")


###########################################
################## ERA5 weather download (excluding total precipitation)
def era5_weather_download(location, month, year, times, output_path, key):
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
    client.retrieve(dataset, request).download(f"{output_path}/ERA5_weather_{year}_{month}.nc") 
    print(f"===== Finished downloading weather data for Month {month} of Year {year}. Saving as ERA5_weather_{year}_{month}.nc")


###########################################
################## ERA5 precipitation download 
def era5_precipitation_download(location, month, year, times, output_path, key):
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
    print(f"===== Finished downloading precipitation data for Year {year}. Saving as ERA5_precip_{year}.nc")


###########################################
################## CERRA temperature download
def cerra_temp_download(month, year, times, output_path, key):
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
    dataset = "reanalysis-cerra-pressure-levels"
    request = {
        "product_type": ["analysis"],
        "data_type": ["reanalysis"],
        "variable": ["temperature"],
        "year": [
            year
        ],
        "month": [month],

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
            "750", 
            "800", "825", "850", 
            "875", "900", "925", 
            "950", "975", "1000"
        ],
        "data_format": "netcdf"
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/CERRA_temp_{year}_{month}.nc")
    print(f"===== Finished downloading temperature data for Month {month} of Year {year}. Saving as CERRA_temp_{year}_{month}.nc")


###########################################
################## ERA5 weather download (excluding total precipitation)
def cerra_weather_download(month, year, times, output_path, key):
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
    dataset = "reanalysis-cerra-single-levels"
    request = {
        "product_type": ["analysis"],
        "data_type": ["reanalysis"],
        "variable": [
            "10m_wind_direction",
            "10m_wind_speed",
            "2m_relative_humidity",
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
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/CERRA_weather_{year}_{month}.nc") 
    print(f"===== Finished downloading weather data for Month {month} of Year {year}. Saving as CERRA_weather_{year}_{month}.nc")


###########################################
################## ERA5 precipitation download 
def cerra_precipitation_download(month, year, times, output_path, key):
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
    dataset = "reanalysis-cerra-single-levels"
    request = {
        "variable": ["total_precipitation"],
        "level_type": "surface_or_atmosphere",
        "data_type": ["reanalysis"],
        "product_type": "forecast",

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
        "leadtime_hour": [
        "1",
        "3",
        "24"
         ],
        "data_format": "netcdf"
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://cds.climate.copernicus.eu/api", 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/CERRA_precip_{year}.nc") 
    print(f"===== Finished downloading precipitation data for Year {year}. Saving as CERRA_precip_{year}.nc")



###########################################
################## EAC4 air pollution download 
def eac4_download(location, year, times, output_path, key):
    ''' Function to download EAC4 air pollution data from cds.

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
    print(f"-------------------- Downloading EAC4 PM data for Year {year} --------------------")
    # 

    # define the dataset and request body
    dataset = "cams-global-reanalysis-eac4"
    request = {
        "variable": [
            "particulate_matter_1um",
            "particulate_matter_2.5um",
            "particulate_matter_10um"
        ],
        "date": [f"{year}-01-01/{year}-12-31"],
        
        "time": times,
        "data_format": "netcdf_zip",
        "area": location
    }

    # establish connection to the CDS API
    client = cdsapi.Client(
        url = "https://ads.atmosphere.copernicus.eu/api", # Note that this URL is different from that for ERA5 
        key = key 
    )

    # download & save the data
    client.retrieve(dataset, request).download(f"{output_path}/EAC4_pm_{year}.nc") 
    print(f"===== Finished downloading EAC4 PM data for Year {year}. Saving to /{location}/EAC4_pm_{year}.nc")


###########################################
################## Create worker wrapper (to help with parallel processing with API keys)
def worker(data_type, year, month, key, location, times, output_folder):
    if data_type == "temperature":
        era5_temp_download(
            location=location,
            month=month,
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "weather":
        era5_weather_download(
            location=location,
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "precipitation":
        era5_precipitation_download(
            location=location,
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "eac4_pm":
        eac4_download(
            location=location,
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )

    elif data_type == "cerra_temperature":
        cerra_temp_download(
            month=month,
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )
            
    elif data_type == "cerra_weather":
        cerra_weather_download(
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )
            
    elif data_type == "cerra_precipitation":
        cerra_precipitation_download(
            month = month, 
            year=year,
            times=times,
            output_path=output_folder,
            key=key
        )
    



