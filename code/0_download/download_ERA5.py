# Import packages 
import multiprocessing as mp
import sys 
import os 
from itertools import cycle
from pathlib import Path
from dotenv import load_dotenv
import json

# Setting paths 
script_folder = Path(__file__).resolve().parents[1]
config_path = script_folder / "config" / "config.json"

# Set path to prewritten function 
sys.path.append(os.path.abspath(f"{script_folder}/utilities"))

# Define data output path (private to each user)
with open(config_path, "r") as f:
    config = json.load(f)
output_folder = Path(config["ERA5_raw"])

# Import user-written functions
import download as dl 

###########################################
################## Define the parameters for download 
years = [str(y) for y in range(2000, 2025)]

location= [53.8, 3.2, 50.6, 7.4]

months = [                              # Only for download of temperature data 
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09", 
            "10", "11", "12" 
            ]

times = [ "00:00",     # Time is in UTC, downloading multiple times to convert to Dutch local times from 12AM to 6AM, 12PM, and 6PM
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "10:00",
    "11:00",
    "16:00",
    "17:00",
    "22:00",
    "23:00",]

###########################################
################## Define API keys and tasks for parallelization 
# Get keys for API
load_dotenv(f"{script_folder}/.env")
CDS_KEYS = [k.strip() for k in os.environ["CDS_KEYS"].split(",") if k.strip()]

key_cycle = cycle(CDS_KEYS)

# ERA5 temperature task list
tasks_temp = [
    ("temperature", year, month, next(key_cycle), location, times, output_folder)
    for year in years
    for month in months
]

# ERA5 weather task list 
tasks_weather = [
    ("weather", year, month, next(key_cycle), location, times, output_folder)
    for year in years
    for month in months
]

# ERA5 precipitation task list 
tasks_precip = [
    ("precipitation", year, months, next(key_cycle), location, times, output_folder)
    for year in years
]

tasks = tasks_temp + tasks_weather + tasks_precip


###########################################
################## Download data  
if __name__ == "__main__":
    with mp.Pool(processes=len(CDS_KEYS)) as pool:
        pool.starmap(dl.worker, tasks)
