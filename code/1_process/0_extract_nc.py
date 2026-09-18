"""
This script extracts data from NC files into Python dataframe (polars) 
and save these as parquet files. 
"""

##########################################
################## Setting up 
# Import packages 
import sys 
import os
import json
import polars as pl
from pathlib import Path

# Set paths 
script_folder = Path(__file__).resolve().parents[1]
config_path = script_folder / "config" / "config.json"

# Set path to prewritten function 
sys.path.append(os.path.abspath(f"{script_folder}/utilities"))
import extract 

# Define private paths
with open(config_path, "r") as f:
    config = json.load(f)
data_in_path = Path(config["ERA5_raw"])
data_out_path = Path(config["ERA5_extracted"])
eac4_in_path = Path(config["EAC4_raw"])
eac4_out_path = Path(config["EAC4_extracted"])


def extract_eac4():
    # Match yearly files only, excluding ZIP archives and data_sfc.nc.
    files = sorted(eac4_in_path.glob("EAC4_pm_[0-9][0-9][0-9][0-9].nc"))
    frames = []
    for filepath in files:
        print(f"Extracting {filepath.name}")
        frames.append(extract.extract_pollution(str(filepath), source="EAC4"))

    df_eac4 = pl.concat(frames).sort(["time", "lat", "lon"])
    outpath = eac4_out_path / "EAC4.parquet"
    df_eac4.write_parquet(outpath)
    print(f"Wrote {df_eac4.height:,} rows to {outpath}")
    return outpath

##########################################################
################## Import NC files and save as dataframes (parquet) 
if __name__ == "__main__":
    extract.extract_weather(ERA5_folder = f"{data_in_path}",
               out_folder = f"{data_out_path}", 
               num_processors=2, years=range(2000, 2026))
    extract_eac4()
    
    
