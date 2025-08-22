"""_summary_
"""
# Import other modules
import converters.behavior_to_nwb
import converters.nwb_saving
import converters.general_to_nwb
import converters.Initiation_nwb
import converters.acquisition_and_unit_to_nwb
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


def convert_data_to_nwb_PB(input_folder, output_folder, mouses_name = None):
    """
    Convert preprocessed .mat data into NWB files.

    Parameters
    ----------
    input_folder : str or Path
        Path to input .mat file.
    output_folder : str or Path
        Directory where NWB files will be saved.
    mouses_name : list of str, optional
        Mouse name(s) to process. If None, all available mice are processed.

    Returns
    -------
    None
    """
    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    
    print("📥 Collecting data from .mat file:", input_folder)

    importlib.reload(converters.Initiation_nwb)
    csv_data = converters.Initiation_nwb.files_to_dataframe(mat_file = input_folder, choice_mouses = mouses_name, dataframe_subject=subject_session_selection)
    csv_data.columns = csv_data.columns.str.strip()
    gc.collect()

    print("Converting data to NWB format for mouse:", list(csv_data["Cell_ID"]))
    failures = []  # (mouse_name, row_idx, err_msg)

    bar = tqdm(total=len(csv_data), desc="Processing ")
    for cell_id in csv_data["Cell_ID"].unique():
        csv_data_row = csv_data[csv_data["Cell_ID"] == cell_id].iloc[0]
        bar.set_postfix_str(str(csv_data_row["Mouse Name"])) 
        bar.update(1)
        try:
            # Creating configs for NWB conversion
            importlib.reload(converters.Initiation_nwb)
            output_path, _ = converters.Initiation_nwb.files_to_config(subject_info=csv_data_row, output_folder=output_folder)  #same for all sessions

            # 📑 Created NWB files
            importlib.reload(converters.general_to_nwb)
            nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path)  #same between Rewarded and NonRewarded sessions

            # o ⏸️ Add intervall container
            importlib.reload(converters.intervals_to_nwb)
            converters.intervals_to_nwb.add_intervals_container(nwb_file=nwb_file, csv_data_row=csv_data_row)

            # o 🧪 Add acquisition and units container
            importlib.reload(converters.acquisition_and_unit_to_nwb)
            converters.acquisition_and_unit_to_nwb.add_to_nwb_acquisition_and_units_containers(nwb_file=nwb_file, csv_data_row=csv_data_row)

            # o ⚙️ Add behavior container
            importlib.reload(converters.behavior_to_nwb)
            converters.behavior_to_nwb.add_behavior_container(nwb_file=nwb_file,csv_data_row=csv_data_row)

            # 🔎 Validating NWB file and saving...
            importlib.reload(converters.nwb_saving)
            if csv_data_row["task"] == "WDT":
                output_folder_save = os.path.join(output_folder, "WDT")
            else:
                output_folder_save = os.path.join(output_folder, "No Task")
            os.makedirs(output_folder_save, exist_ok=True)
            nwb_path = converters.nwb_saving.save_nwb_file(nwb_file=nwb_file, output_folder=output_folder_save)  # same between Rewarded and NonRewarded sessions

            with NWBHDF5IO(nwb_path, 'r') as io:
                nwb_errors = validate(io=io)

            if nwb_errors:
                os.remove(nwb_path)
                raise RuntimeError("NWB validation failed: " + "; ".join(map(str, nwb_errors)))

            # Delete .yaml config file 
            if os.path.exists(output_path):
                os.remove(output_path)
            del csv_data_row
            gc.collect()
            
        except Exception as e:
            failures.append((csv_data_row["Cell_ID"], str(e)))
            continue
    if len(failures) > 0:
        print(f"⚠️ Conversion completed except for :")
        for i, (id, error) in enumerate(failures):
            print(f"    - {id}: {error}")
    bar.close()
    for f in Path(output_folder).glob("*.yaml"):  
        f.unlink()
    print("**************************************************************************")



#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ MAIN _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to NWB format for PB sessions")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the .mat files")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the NWB file will be saved")
    parser.add_argument("--mouses_name", nargs='+', default=None, help="Mouse name(s) to process (e.g., LB010)")

    args = parser.parse_args()

    convert_data_to_nwb_PB(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        mouses_name=args.mouses_name
    )