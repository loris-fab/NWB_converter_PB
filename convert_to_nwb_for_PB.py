"""_summary_
"""
# Import other modules
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_to_nwb
import converters.analysis_to_nwb
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




############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################


def convert_data_to_nwb_PB(mat_file, output_folder, mouses_name = None):
    """
    Converts data from a config file to an NWB file.
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files

    """
    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print("📥 Collecting data from .mat file:", mat_file)
    importlib.reload(converters.Initiation_nwb)
    csv_data = converters.Initiation_nwb.files_to_dataframe(mat_file = mat_file, choice_mouses = mouses_name)
    csv_data.columns = csv_data.columns.str.strip() 
    display(csv_data[['Mouse Name', 'User (user_userName)', 'Cell_ID', 'Ear tag',
        'Start date (dd.mm.yy)', 'End date', 'Sex_bin', 'strain', 'mutations',
        'Birth date', 'licence', 'DG', 'ExpEnd', 'Created on', 'Session',
        'Session Date (yyymmdd)', 'Start Time (hhmmss)', 'Behavior Type',
        'Session Type', 'Opto Session', 'Mouse Age (d)', 'Weight of Reference',
        'Weight Session']].head(5))

    all_sessions = csv_data["Session"]
    print("Converting data to NWB format for mouse:", list(all_sessions))
    i = 0
    for index, csv_data_row in csv_data.iterrows():
        i += 1

        """
        # Check the behavior type of the session
        if csv_data_row["Behavior Type"] == "":
            Rewarded = True
        elif csv_data_row["Behavior Type"] == "":
            Rewarded = False
        else :
            raise ValueError(f"Unknown behavior type: {csv_data_row['Behavior Type']}")
        """

        print("📃 Creating configs for NWB conversion :") if index == 1 else None
        importlib.reload(converters.Initiation_nwb)
        output_path, config_file = converters.Initiation_nwb.files_to_config(subject_info=csv_data_row.drop("sweeps"), output_folder=output_folder)  #same for all sessions

        print("📑 Created NWB files :") if i == 1 else None
        importlib.reload(converters.general_to_nwb)
        nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path)  #same between Rewarded and NonRewarded sessions

        """                                
        print("     o 📌 Add general metadata") if i == 1 else None
        importlib.reload(converters.acquisition_to_nwb)
        signal_LFP, regions, EMG, EEG = converters.acquisition_to_nwb.extract_lfp_signal(csv_data_row=csv_data_row)
        electrode_table_region, labels = converters.general_to_nwb.add_general_container(nwb_file=nwb_file, csv_data_row=csv_data_row, regions=regions)  #same between Rewarded and NonRewarded sessions
        if index == 1 :
            print("         - Subject metadata")
            print("         - Session metadata") 
            print("         - Device metadata") 
            print("         - Extracellular electrophysiology metadata") 
        else :
            None
        """
        
        if False:
            print("     o 📌 Add general metadata") if i == 1 else None
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

        
            print("     o 📶 Add acquisition container") if i == 1 else None
            converters.acquisition_to_nwb.add_acquisitions_3series(nwb_file, lfp_array=signal_LFP, electrode_region_all=electrode_table_region, channel_labels=labels, emg=EMG, eeg=EEG)  #same between Rewarded and NonRewarded sessions


            print("     o ⚙️ Add processing container") if i == 1 else None
            importlib.reload(converters.behavior_to_nwb)
            importlib.reload(converters.analysis_to_nwb)
            if Rewarded:
                print("         - Behavior data") if i == 1 else None
                trial_onsets, stim_data , response_data_type, window_trial =converters.behavior_to_nwb.add_behavior_container_Rewarded(nwb_file=nwb_file,csv_data_row=csv_data_row)
                info_trials = [trial_onsets, stim_data , response_data_type, window_trial]
            else:
                print("         - Behavior data") if i == 1 else None
                converters.behavior_to_nwb.add_behavior_container_NonRewarded(nwb_file=nwb_file,csv_data_row=csv_data_row)

            print("         - No ephys data for AN sessions") if i == 1 else None
            print("         - Analysis complementary information") if i == 1 else None
            #converters.analysis_to_nwb.add_analysis_container(nwb_file=nwb_file, psth_window=psth_window, rewarded=Rewarded)  #same between Rewarded and NonRewarded sessions
            #print("             > Added LFP_mean_across_all_units to analysis module") if i == 1 else None
            #print("             > Added global_LFP to analysis module") if i == 1 else None
         
            print("     o ⏸️ Add intervall container")
            importlib.reload(converters.intervals_to_nwb)
            converters.intervals_to_nwb.add_intervals_container_Rewarded(nwb_file=nwb_file, csv_data_row=csv_data_row)
        else:  
            importlib.reload(converters.nwb_saving)
            nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)  # same between Rewarded and NonRewarded sessions

            print(" ") if i == 1 else None
            print(f"🔎 Validating NWB file before saving...") if i == 1 else None
            with NWBHDF5IO(nwb_path, 'r') as io:
                errors = validate(io=io)

            if not errors:
                print(f"     o ✅ File {nwb_path} is valid, no errors detected and saved successfully.")
            else:
                print(f"     o ❌ NWB file {nwb_path} is invalid, deleting file..")
                os.remove(nwb_path)
                for err in errors:
                    print("         -", err)

            # Delete .yaml config file 
            if os.path.exists(output_path):
                os.remove(output_path)

            # Stop after processing 2 sessions for testing purposes
            #if i == 1 :
            #    break 
            #print("Don't forget to delete the testing purpose")
    print("**************************************************************************")



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