"""_summary_
"""
# Import other modules
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_to_nwb
import converters.intervals_to_nwb

# Import libraries
import os
import h5py
import importlib
import argparse
import shutil
import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO, validate
from pathlib import Path
from tqdm import tqdm
import gc

subject_session_selection = pd.read_csv('Subject_Session_Selection.csv')

############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_PB(output_folder, mouses_name = None):
    """
    Converts data from a config file to an NWB file.
    """
    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    mat_file = "/Users/lorisfabbro/Documents/MATLAB/PB/mPFC_Preprocessed.mat"
    
    print("📥 Collecting data from .mat file:", mat_file)

    importlib.reload(converters.Initiation_nwb)
    csv_data = converters.Initiation_nwb.files_to_dataframe(mat_file = mat_file, choice_mouses = mouses_name, dataframe_subject=subject_session_selection)
    csv_data.columns = csv_data.columns.str.strip()
    gc.collect()

    display(csv_data[['Mouse Name', 'User (user_userName)', 'Cell_ID', 'Ear tag',
        'Start date (dd.mm.yy)', 'End date', 'Sex_bin', 'strain', 'mutations',
        'Birth date', 'licence', 'DG', 'ExpEnd', 'Created on', 'Session',
        'Session Date (yyymmdd)', 'Start Time (hhmmss)', 'Behavior Type',
        'Session Type', 'Opto Session', 'Mouse Age (d)', 'Weight of Reference',
        'Weight Session']].head(5))


    print("Converting data to NWB format for mouse:", list(csv_data["Cell_ID"]))
    failures = []  # (mouse_name, row_idx, err_msg)
    bar = tqdm(total=len(csv_data), desc="Processing ")
    for _, csv_data_row in csv_data.iterrows():
        bar.set_postfix_str(str(csv_data_row["Mouse Name"])) 
        bar.update(1)
    #try:
        """
        # Check the behavior type of the session
        if csv_data_row["Behavior Type"] == "":
            Rewarded = True
        elif csv_data_row["Behavior Type"] == "":
            Rewarded = False
        else :
            raise ValueError(f"Unknown behavior type: {csv_data_row['Behavior Type']}")
        """

        # Creating configs for NWB conversion
        importlib.reload(converters.Initiation_nwb)
        output_path, config_file = converters.Initiation_nwb.files_to_config(subject_info=csv_data_row.drop("sweeps"), output_folder=output_folder)  #same for all sessions

        # 📑 Created NWB files
        importlib.reload(converters.general_to_nwb)
        nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path)  #same between Rewarded and NonRewarded sessions

        # o ⏸️ Add intervall container
        importlib.reload(converters.intervals_to_nwb)
        converters.intervals_to_nwb.add_intervals_container(nwb_file=nwb_file, csv_data_row=csv_data_row)

        if False:
            # o 📌 Add general metadata
            importlib.reload(converters.acquisition_to_nwb)
            signal_LFP, regions, EMG, EEG = converters.acquisition_to_nwb.extract_lfp_signal(csv_data_row=csv_data_row)
            electrode_table_region, labels = converters.general_to_nwb.add_general_container(nwb_file=nwb_file, csv_data_row=csv_data_row, regions=regions)  #same between Rewarded and NonRewarded sessions
            if i == 1 :
                print("         - Subject metadata")
                print("         - Session metadata") 
                print("         - Device metadata") 
                print("         - Extracellular electrophysiology metadata") 
            else :
                None

        
            # o 📶 Add acquisition container
            converters.acquisition_to_nwb.add_acquisitions_3series(nwb_file, lfp_array=signal_LFP, electrode_region_all=electrode_table_region, channel_labels=labels, emg=EMG, eeg=EEG)  #same between Rewarded and NonRewarded sessions


        # o ⚙️ Add behavior container
        importlib.reload(converters.behavior_to_nwb)
        #converters.behavior_to_nwb.add_behavior_container(nwb_file=nwb_file,csv_data_row=csv_data_row)

        # 🔎 Validating NWB file and saving...
        importlib.reload(converters.nwb_saving)
        nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)  # same between Rewarded and NonRewarded sessions

        with NWBHDF5IO(nwb_path, 'r') as io:
            nwb_errors = validate(io=io)

        if nwb_errors:
            os.remove(nwb_path)
            raise RuntimeError("NWB validation failed: " + "; ".join(map(str, nwb_errors)))

        # Delete .yaml config file 
        if os.path.exists(output_path):
            os.remove(output_path)
        gc.collect()
        
    #except Exception as e:
    #    failures.append((csv_data_row["Cell_ID"], str(e)))
    #    continue
#if len(failures) > 0:
    #print(f"⚠️ Conversion completed except for :")
    #for i, (id, error) in enumerate(failures):
    #    print(f"    - {id}: {error}")
    bar.close()
    for f in Path(output_folder).glob("*.yaml"):  
        f.unlink()
    print("**************************************************************************")



#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for PB sessions")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB file will be saved")
    parser.add_argument("--mouses_name", nargs='+', default=None, help="Mouse name(s) to process (e.g., LB010)")

    args = parser.parse_args()

    convert_data_to_nwb_PB(
        output_folder=args.output_folder,
        mouses_name=args.mouses_name
    )