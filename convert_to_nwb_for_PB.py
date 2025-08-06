"""_summary_
"""
# Import other modules
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_to_nwb
import converters.units_to_nwb
import converters.analysis_to_nwb
import converters.intervals_to_nwb

# Import libraries
import os
import h5py
import importlib
import argparse
import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO, validate




############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_PB(mat_file, output_folder, mouses_name = None):
    """
    Converts data from a config file to an NWB file.
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files

    """
    csv_data = converters.Initiation_nwb.files_to_dataframe(mat_file = mat_file, choice_mouses = mouses_name)
    csv_data.columns = csv_data.columns.str.strip() 
    
    all_sessions = csv_data["Session"]
    
    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    for index, csv_data_row in csv_data.iterrows():

        # Check the behavior type of the session
        if csv_data_row["Behavior Type"] == "":
            Rewarded = True
        elif csv_data_row["Behavior Type"] == "":
            Rewarded = False
        else :
            raise ValueError(f"Unknown behavior type: {csv_data_row['Behavior Type']}")


        print("Converting data to NWB format for mouse:", list(all_sessions)) if index == 0 else None
        print("📃 Creating configs for NWB conversion :") if index == 0 else None
        importlib.reload(converters.Initiation_nwb)
        # Create the config file for the NWB conversion
        output_path, config_file = converters.Initiation_nwb.files_to_config(csv_data_row=csv_data_row, output_folder=output_folder, mat_file=mat_file)  

    """
    print("📑 Created NWB file :")
    importlib.reload(converters.general_to_nwb)
    print(config_file['session_metadata']["session_description"])
    nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path) # same for rewarded and non-rewarded sessions                                      
    
    print("     o 📌 Add general metadata")
    importlib.reload(converters.acquisition_to_nwb)
    signal, regions = converters.acquisition_to_nwb.extract_lfp_signal(data, mat_file)
    electrode_table_region, unique_values = converters.general_to_nwb.add_general_container(nwb_file=nwb_file, data=data, mat_file=mat_file, regions=regions) # same for rewarded and non-rewarded sessions
    print("         - Subject metadata")
    print("         - Session metadata")
    print("         - Device metadata")
    print("         - Extracellular electrophysiology metadata")
    
    print("     o 📶 Add acquisition container")
    converters.acquisition_to_nwb.add_lfp_acquisition(nwb_file=nwb_file, signal_array=signal, electrode_region=electrode_table_region) # same for rewarded and non-rewarded sessions  

    print("     o ⏸️ Add intervall container")
    importlib.reload(converters.intervals_to_nwb)
    if Rewarded:
        converters.intervals_to_nwb.add_intervals_container_Rewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)
    #else:
        #converters.intervals_to_nwb.add_intervals_container_NonRewarded(nwb_file=nwb_file, data=data, mat_file=mat_file)

    print("     o 🧠 Add units container")
    importlib.reload(converters.units_to_nwb)
    sampling_rate =  30000
    converters.units_to_nwb.add_units_container(nwb_file=nwb_file, data=data, unique_values=unique_values, mat_file=mat_file , sampling_rate = sampling_rate , regions=regions) # same for rewarded and non-rewarded sessions

    print("     o ⚙️ Add processing container")
    importlib.reload(converters.behavior_to_nwb)
    importlib.reload(converters.analysis_to_nwb)
    if Rewarded:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file, data=data, config=config_file)
    else:
        print("         - Behavior data")
        converters.behavior_to_nwb.add_behavior_container_NonRewarded(nwb_file=nwb_file, data=data, config_file=config_file)

    print("         - No ephys data for AN sessions")
    print("         - Analysis complementary information")
    converters.analysis_to_nwb.add_analysis_container(nwb_file=nwb_file, Rewarded=Rewarded, psth_window=psth_window, psth_bin=psth_bin) # almost same for rewarded and non-rewarded sessions

    importlib.reload(converters.nwb_saving)
    nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder) # same for rewarded and non-rewarded sessions

    print(" ")
    print("🔎 Validating NWB file before saving...")
    with NWBHDF5IO(nwb_path, 'r') as io:
        errors = validate(io=io)

    if not errors:
        print("     o ✅ File is valid, no errors detected.")
    else:
        print("     o ❌ Errors detected:")
        for err in errors:
            print("         -", err)
    print(" ")
    print("💾 Saving NWB file")
    if not errors:
        print("     o 📂 NWB file saved at:")
        print("         -", nwb_path)
    else:
        print("     o ❌ NWB file is invalid, deleting file...")
        os.remove(nwb_path)
    print("**************************************************************************")

    # Delete .yaml config file 
    if os.path.exists(output_path):
        os.remove(output_path)

    """

#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for AN sessions")
    parser.add_argument("mat_file", type=str, help="Path to the .mat file containing the data")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB file will be saved")


    args = parser.parse_args()

    convert_data_to_nwb_PB(
        mat_file=args.mat_file,
        output_folder=args.output_folder,
    )