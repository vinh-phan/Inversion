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
output_folder = Path(config["EAC4_raw"])

# Import user-written functions
import download as dl 

###########################################
################## Define the parameters for download 
years = [str(y) for y in range(2003, 2026)]

location= [53.8, 3.2, 50.6, 7.4]

times = [ "00:00",     # Time is in UTC
    "03:00",
    "06:00",
    "09:00",
    "12:00",
    "15:00",
    "18:00",
    "21:00",]

###########################################
################## Define API keys and tasks for parallelization 
# Get keys for API
load_dotenv(f"{script_folder}/.env")
CDS_KEYS = [k.strip() for k in os.environ["CDS_KEYS"].split(",") if k.strip()]

key_cycle = cycle(CDS_KEYS)

# EAC4 task list
tasks_EAC4 = [
    ("eac4_pm", year, None, next(key_cycle), location, times, output_folder)
    for year in years
]


###########################################
################## Download data  
if __name__ == "__main__":
    with mp.Pool(processes=len(CDS_KEYS)) as pool:
        pool.starmap(dl.worker, tasks_EAC4)