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
from datetime import datetime, timedelta
from pynwb import NWBHDF5IO, validate
from typing import List, Optional
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import importlib
import argparse
import time
import h5py
import gc
import os


############################################################
# Functions for converting data to NWB format for AN sessions
#############################################################

def convert_data_to_nwb_PB(input_folder, output_folder, choice_mouses = None):
    """
    Convert preprocessed .mat ("mPFC_Preprocessed.mat" in <input_folder>) data into NWB files.

    Parameters
    ----------
    input_folder : str or Path
        Path to the input .mat file.
    output_folder : str or Path
        Directory where NWB files will be saved. 
    choice_mouses : list of str, optional
        Mouse name(s) to process. If None, all available mice are processed.

    Returns
    -------
    None but saves NWB files to the <output_folder>.
    """

    subject_session_selection = pd.read_csv('Subject_Session_Selection.csv')

    print("**************************************************************************")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print("📥 Collecting data from .mat file: ", input_folder)

    # Reload the module 
    importlib.reload(converters.Initiation_nwb)
    importlib.reload(converters.general_to_nwb)
    importlib.reload(converters.intervals_to_nwb)
    importlib.reload(converters.acquisition_and_unit_to_nwb)
    importlib.reload(converters.behavior_to_nwb)
    importlib.reload(converters.nwb_saving)


    ##############################################################
    ##################### Load data from .mat #####################
    ###############################################################

    #------------------------- helpers ----------------------------
    def reward_in_trial(
        reward_times: List[datetime],
        trial_start: datetime,
        trial_stop: datetime,
        has_reward: bool,
        start_time: datetime
    ) -> Optional[datetime]:
        """Return The reward time in seconds if exactly one is found within [trial_start, trial_stop]"""
        if not has_reward:
            return None
        if trial_stop < trial_start:
            raise ValueError("trial_stop is before trial_start")

        hits = [t for t in reward_times if trial_start <= t <= trial_stop]

        if len(hits) == 1:
            return (hits[0]-start_time).total_seconds()
        elif len(hits) == 0 or len(hits) > 1:
            raise ValueError("Inconsistency: has_reward=True but no reward in interval.")
        raise ValueError(f"Inconsistency: multiple rewards in interval: {hits}")

    def read_cell_numeric_vectors(f, ref_array):
        """Return np.array for each cell"""
        out = []
        for r in np.ravel(ref_array):
            obj = f[r]
            val = np.array(obj[()])
            out.append(val.squeeze())
        return out

    def read_cell_strings(f, ref_array):
        """Return a list of strings (or None if empty)."""
        out = []
        for r in np.ravel(ref_array):
            obj = f[r]
            val = np.array(obj[()])
            if val.size == 0:
                out.append(None)
            elif val.dtype.kind in ('S', 'U'):  
                s = val.tobytes().decode('utf-8', errors='ignore') if val.dtype.kind == 'S' else str(val)
                out.append(s.strip() if s else None)
            else:
                try:
                    chars = ''.join(chr(int(x)) for x in val.ravel() if int(x) > 0)
                    out.append(chars if chars else None)
                except Exception:
                    out.append(None)
        return out

    def safe_abs_times(rel_times, t0):
        """Relative times (s) -> absolute times (datetime) from t0.
        Handles None/NaN/empty arrays."""
        if rel_times is None or (isinstance(rel_times, float) and np.isnan(rel_times)):
            return None
        arr = np.atleast_1d(rel_times).astype(float)
        if arr.size == 0:
            return []
        return [t0 + timedelta(seconds=float(x)) for x in arr]

    def safe_list(x):
        """Convert to list, handling None/empty arrays."""
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            if x.size == 0 or (x.shape == (2,) and np.array_equal(x, [0, 0])):
                return []
        arr = np.atleast_1d(x)
        return arr.tolist()

    def _decode_chars(x):
        """Decode character array to string."""
        a = np.array(x)
        return ''.join(chr(int(v)) for v in a.ravel() if int(v) > 0)

    def read_cell_strings(f, path):
        """Read cell strings from HDF5 file."""
        refs = f[path][()]
        return np.array([_decode_chars(f[r][()]) for r in np.ravel(refs)])

    def read_sweep_start_times(f, path):
        """Read sweep start times from HDF5 file."""
        ST = np.array(f[path][()])  # shape (6, N) in general
        times = []
        for raw in ST.T:
            raw = np.array(raw, dtype=float)
            if np.all(np.isnan(raw)): 
                times.append(np.nan); continue
            y, m, d, H, M, S = map(int, raw[:6])
            if S == 60: S, M = 0, M + 1
            times.append(datetime(y, m, d, H, M, S))
        return np.array(times)

    def _decode_any(x):
        """Decode various formats to string."""
        a = np.array(x)
        if a.size == 0:
            return None
        if a.dtype.kind == 'S':
            return a.tobytes().decode('utf-8', errors='ignore').strip()
        if a.dtype.kind == 'U':
            return str(a).strip()
        if a.dtype.kind in ('i', 'u'):
            return ''.join(chr(int(v)) for v in a.ravel() if int(v) > 0)
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode('utf-8', errors='ignore').strip()
        if isinstance(x, (str, np.str_)):
            return x.strip()
        return None

    def read_cell_strings_from_refs(f, ref_array):
        """Support : simple string or cell-array of strings (nested references)."""
        out = []
        for r in np.ravel(ref_array):
            obj = f[r][()]
            arr = np.array(obj, dtype=object)
            if arr.dtype == object:
                items = []
                for el in arr.ravel():
                    if isinstance(el, h5py.h5r.Reference):
                        items.append(_decode_any(f[el][()]))
                    else:
                        items.append(_decode_any(el))
                items = [s for s in items if s not in (None, '', ' ')]
                out.append(items if len(items) > 1 else (items[0] if items else None))
            else:
                out.append(_decode_any(arr))
        return np.array(out, dtype=object)
    #-------------------------------------------------------------

    # Load mPFC_Preprocessed.mat with h5py for all sessions
    with tqdm(total=5) as pbar:

        with h5py.File(input_folder, 'r') as f:

            # --- Mouse metadata from HDF5 ----
            # Step 1: Mouse metadata loaded
            Cell_ID        = read_cell_strings(f, 'Data_Full/Cell_ID')
            mouses_name    = np.array([s.split('_')[0] for s in Cell_ID])
            Mouse_Sex      = read_cell_strings(f, 'Data_Full/Mouse_Sex')
            sessiontypes   = read_cell_strings(f, 'Data_Full/Session_Type')
            Sweep_type  = read_cell_strings(f, 'Data_Full/Sweep_Type')
            Cell_Depth = np.array(f['Data_Full/Cell_Depth'][()]).squeeze()

            time.sleep(1)
            tqdm.write("Loading mouse metadata ...")
            pbar.update(1)

            # --- SWEEPS ----
            # Start times
            sweep_start_times = read_sweep_start_times(f, 'Data_Full/Sweep_StartTime')

            # Signal data 
            mp_refs   = f['Data_Full/Sweep_MembranePotential'][()]
            cm_refs   = f['Data_Full/Sweep_CurrentMonitor'][()]
            wa_refs   = f['Data_Full/Sweep_WhiskerAngle'][()]


            vm_list   = read_cell_numeric_vectors(f, mp_refs)  
            cur_list  = read_cell_numeric_vectors(f, cm_refs)
            wa_list   = read_cell_numeric_vectors(f, wa_refs)

            time.sleep(1)
            tqdm.write("Loading sweep signal data (1/3) ...")
            pbar.update(1)

            # Lick
            lick_refs   = f['Data_Full/Sweep_Lick'][()]
            lick_power_refs = f['Data_Full/Sweep_Lick_Power'][()]
            lick_threshold  = np.array(f['Data_Full/Sweep_Lick_Threshold'][()]).squeeze() 

            lick_list        = read_cell_numeric_vectors(f, lick_refs)
            lick_power_list  = read_cell_numeric_vectors(f, lick_power_refs)

            # AP times
            ap_refs   = f['Data_Full/Sweep_AP_Times'][()]
            ap_list   = read_cell_numeric_vectors(f, ap_refs)

            time.sleep(1)
            tqdm.write("Loading sweep signal data (2/3) ...")
            pbar.update(1)

            # Stim
            stim_on_refs = f['Data_Full/Sweep_Stim_Onset_Times'][()]
            stim_amp_refs= f['Data_Full/Sweep_Stim_Amp'][()]
            stim_dur_refs= f['Data_Full/Sweep_Stim_Dur'][()]
            stim_type_refs = f['Data_Full/Sweep_Stim_Type'][()]

            stim_on_list  = read_cell_numeric_vectors(f, stim_on_refs)
            stim_amp_list = read_cell_numeric_vectors(f, stim_amp_refs)
            stim_dur_list = read_cell_numeric_vectors(f, stim_dur_refs)
            stim_type_list = read_cell_strings_from_refs(f, stim_type_refs)

            # Rewards
            reward_refs = f['Data_Full/Sweep_Reward_Times'][()]
            reward_list = read_cell_numeric_vectors(f, reward_refs)

            # Step 4: Sweep data loaded
            time.sleep(1)
            tqdm.write("Loading sweep signal data (3/3) ...")
            pbar.update(1)

            # Behavior matrix (n_trials x 5)
            behav_refs = f['Data_Full/Sweep_Behavior'][()]
            behav_list = []
            for r in np.ravel(behav_refs):
                mat = np.array(f[r][()]).squeeze()
                if mat.size == 0:
                    behav_list.append(np.empty((0,5)))
                else:
                    if mat.ndim == 1:
                        if mat.size % 5 == 0:
                            mat = mat.reshape((-1,5))
                        else:
                            mat = np.expand_dims(mat, 0) if mat.size == 5 else np.empty((0,5))
                    behav_list.append(mat)

            # Sampling rates
            sr_vm_cm = 20000
            sr_wa    = 200

            # Step 5: Sweep data loaded
            time.sleep(1)
            tqdm.write("Loading sweep behavior data ...")
            pbar.update(1)


    ##############################################################
    ################# Convert data to NWB format #################
    ##############################################################

    # Treatment of mouse selection
    if choice_mouses is None:
        choice_mouses = np.unique(mouses_name)
    else:
        missing = [name for name in choice_mouses if name not in np.unique(mouses_name)]
        if missing:
            raise ValueError(f"Mouse name(s) not found in csv file: {missing}")
        
    # Get the indices of the chosen mice
    index_mouses_choices = np.concatenate([np.where(mouses_name == mouse)[0] for mouse in choice_mouses if mouse in mouses_name])

    # Get unique cell IDs and their corresponding indices 
    unique_ids, _ = np.unique(Cell_ID, return_index=True)
    cell = [np.arange(len(Cell_ID))[Cell_ID == unique_id] for unique_id in unique_ids]


    print("Converting data to NWB format for mouse:")

    # Get the cells corresponding to the chosen mice
    choice_cells = [x for x in unique_ids if any(x.startswith(y) for y in np.unique(choice_mouses))]
    print(list(choice_cells))

     # List of failures: (cell_id, error_message)
    failures = []  

    # Itèration over the indices of the choice mouses to collect data in the dataframe
    def to_data_frame(cell_id):
        """
        Converts raw data for a given cell_id into a structured pandas Series.

        Parameters
        ----------
        cell_id : str
            Unique identifier of the cell.

        Returns
        -------
        pandas.Series
            Row containing metadata, session info, sweeps, and trials.
        """
        for one_cell in cell: # The loop finds the corresponding cell for the given cell_id
            if one_cell[0] in index_mouses_choices and cell_id == Cell_ID[one_cell[0]]:
                name = mouses_name[one_cell[0]]
                sex = Mouse_Sex[one_cell[0]]
                session_type = sessiontypes[one_cell[0]]
                #behaviortype = " & ".join(np.unique(Sweep_type[one_cell]))
                if session_type not in ["Trained", "D1", "Naive"]:
                    raise ValueError(f"Unknown session type: {session_type}. Expected 'Trained', 'D1', or 'Naive'.") 
                behaviortype = "Whisker rewarded (WR+)" if (session_type == "Trained" or session_type == "D1") else "No Task"  
                start_time_hhmmss = sweep_start_times[one_cell[0]].strftime("%H%M%S") if not pd.isnull(sweep_start_times[one_cell[0]]) else None
                start_time = sweep_start_times[one_cell[0]] if not pd.isnull(sweep_start_times[one_cell[0]]) else None

                row = subject_session_selection[subject_session_selection['Mouse Name'] == name]
                date_str = str(row['Birth date'].iloc[0]).strip() 
                birth_date = date_str.replace("/", ".")
                date_session_str = str(row['Start date'].iloc[0]).strip()
                date_session_str = date_session_str.replace("/", ".")
                Weight_Session = float(row['Weight Experimet day'].iloc[0])
                ref_weight = float(row['Weight of Refence'].iloc[0]) 
                Ear_tag = str(row['Ear tag'].iloc[0]).strip()
                Created_on = str(row['Created on'].iloc[0]).strip()
                task = str(row['Behavioral task'].iloc[0]).strip()
                counter = int(row['Session Number'].iloc[0])
                start_date = str(row['Start date'].iloc[0]).strip()
                start_date = start_date.replace("/", ".")
                end_date = str(row['End date'].iloc[0]).strip()
                end_date = end_date.replace("/", ".")
                age = (datetime.strptime(start_date, "%d.%m.%Y") - datetime.strptime(birth_date, "%d.%m.%Y")).days


                session_date = start_date.replace(".", "")[4:] + end_date.replace(".", "")[2:4] + end_date.replace(".", "")[0:2]

                sweeps = []
                for one_sweep in one_cell:
                    t0 = sweep_start_times[one_sweep] if not pd.isnull(sweep_start_times[one_sweep]) else None

                    vm      = vm_list[one_sweep]   #if one_sweep < len(vm_list) else np.array([])
                    cur     = cur_list[one_sweep]  #if one_sweep < len(cur_list) else np.array([])
                    wa      = wa_list[one_sweep]   #if one_sweep < len(wa_list) else np.array([])
                    ap_rel  = ap_list[one_sweep]   #if one_sweep < len(ap_list) else np.array([])
                    stim_on = stim_on_list[one_sweep] #if one_sweep < len(stim_on_list) else np.array([])
                    stim_amp= stim_amp_list[one_sweep] #if one_sweep < len(stim_amp_list) else np.array([])
                    stim_dur= stim_dur_list[one_sweep] #if one_sweep < len(stim_dur_list) else np.array([])
                    reward_rel = reward_list[one_sweep] #if one_sweep < len(reward_list) else np.array([])
                    behav   = behav_list[one_sweep].T #if one_sweep < len(behav_list) else np.empty((0,5))
                    stim_type = stim_type_list[one_sweep] #if one_sweep < len(stim_type_list) else None

                    # durations
                    dur_vm = float(vm.size / sr_vm_cm) if sr_vm_cm and vm.size>0 else None
                    #dur_wa = float(wa.size / sr_wa)    if sr_wa and wa.size>0 else None

                    # absolute times
                    ap_abs = safe_abs_times(safe_list(ap_rel), t0) if t0 is not None else None
                    stim_on_abs  = safe_abs_times(safe_list(stim_on), t0)  if t0 is not None else None
                    reward_abs   = safe_abs_times(safe_list(reward_rel), t0) if t0 is not None else None

                    # Trials -> liste de dicts (col1=ts rel, col2=stim?, col3=lick, col4=reward, col5=response)
                    trials = []
                    duration_trial = 2.0
                    if behav.size > 0:
                        ts_rel   = behav[:,0].astype(float)
                        has_stim = behav[:,1].astype(bool)
                        lick     = behav[:,2].astype(bool)
                        reward_b = behav[:,3].astype(bool)
                        response = behav[:,4].astype(int)
                        
                        ts_abs = [t0 + timedelta(seconds=float(x)) for x in ts_rel] if t0 is not None else [None]*len(ts_rel)

                        stim_type = [stim_type] if isinstance(stim_type, str) else ([] if stim_type is None else safe_list(stim_type))

                        if len(stim_type) != 0: # if not Free Whisking
                            stim_on = safe_list(stim_on)
                            #print(stim_on)
                            #print(has_stim[has_stim > 0])
                            stim_on_abs = stim_on_abs
                            stim_amp = safe_list(stim_amp)
                            stim_dur = safe_list(stim_dur)
                            if len(has_stim[has_stim > 0]) != len(np.array(stim_on)[np.array(stim_on) > 0]): 
                                raise ValueError(f"Mismatch in lengths: has_stim ({len(has_stim[has_stim > 0])}) vs stim_on ({len(safe_list(stim_on))}) for sweep {one_sweep} and mouse {name}")
                            if stim_on_abs is None : 
                                if  len(stim_on) != len(stim_amp) or len(stim_on) != len(stim_dur) or len(stim_on) != len(stim_type):
                                    raise ValueError(f"Stimulus data length mismatch: {len(stim_on)} vs {len(stim_on_abs)}, {len(stim_amp)}, {len(stim_dur)}, {len(stim_type)} for sweep {one_sweep} and mouse {name}")
                            else :
                                if len(stim_on) != len(stim_on_abs) or len(stim_on) != len(stim_amp) or len(stim_on) != len(stim_dur) or len(stim_on) != len(stim_type):
                                    raise ValueError(f"Stimulus data length mismatch: {len(stim_on)} vs {len(stim_on_abs)}, {len(stim_amp)}, {len(stim_dur)}, {len(stim_type)} for sweep {one_sweep} and mouse {name}")

                        ind_stim = 0
                        for k in range(len(ts_rel)):
                            if bool(has_stim[k]) == False:
                                trials.append({
                                "trial_index": int(k),
                                "time_rel_s": float(ts_rel[k]),
                                "time_abs": (ts_abs[k]- start_time).total_seconds(),
                                "lick": bool(lick[k]),
                                "reward": bool(reward_b[k]),
                                "reward_time": reward_in_trial(reward_abs, ts_abs[k], ts_abs[k] + timedelta(seconds=2), bool(reward_b[k]),start_time),
                                "response": int(response[k]),
                                'has_stim': bool(has_stim[k]),
                                'amplitude': 0,
                                'duration_s': 0,
                                'type': "n.a",
                                })
                            elif bool(has_stim[k]) == True:
                                trials.append({
                                "trial_index": int(k),
                                "time_rel_s": float(ts_rel[k]),
                                "time_abs": (ts_abs[k]- start_time).total_seconds(),
                                "lick": bool(lick[k]),
                                "reward": bool(reward_b[k]),
                                "reward_time": reward_in_trial(reward_abs, ts_abs[k], ts_abs[k] + timedelta(seconds=2), bool(reward_b[k]), start_time),
                                "response": int(response[k]),
                                'has_stim': bool(has_stim[k]),
                                'stim_time_rel_s': stim_on[ind_stim] ,
                                'stim_time_abs': (stim_on_abs[ind_stim]- start_time).total_seconds() if stim_on_abs is not None else None,
                                'amplitude': stim_amp[ind_stim],
                                'duration_s': stim_dur[ind_stim],
                                'type': stim_type[ind_stim]

                                })
                                ind_stim += 1
                                
                    if (ind_stim == 0 and len(stim_type) != 0):
                        raise ValueError(f"Stimulus index mismatch: {ind_stim} vs {len(stim_on)-1} for sweep {one_sweep} and mouse {name}")
                    elif (ind_stim > 0 and (ind_stim != len(stim_type))):
                        raise ValueError(f"Stimulus index mismatch: {ind_stim} vs {len(stim_type)-1} for sweep {one_sweep} and mouse {name}")

                    sweep_dict = {
                        "Sweep Index": int(one_sweep)-int(one_cell[0]),
                        "Sweep Start Time": ((t0) - start_time).total_seconds(),
                        "Sweep Stop Time": ((t0 + timedelta(seconds=float(dur_vm))) - start_time).total_seconds(),
                        "Sweep Type": str(Sweep_type[one_sweep]),
                        #'behav': behav,

                        # Signaux
                        "membrane_potential": {
                            "sampling_rate_Hz": int(sr_vm_cm),
                            #"n_samples": int(vm.size),
                            #"duration_s": dur_vm,
                            "data": vm.tolist() if vm.size>0 else [],
                            "target area": "mPFC",
                            "Type of neurone": "NaN",
                            "Cell_ID" : Cell_ID[one_cell[0]],
                            "Cell_Depth (um)": Cell_Depth[one_sweep]
                        },
                        "current_monitor": {
                            "sampling_rate_Hz": int(sr_vm_cm),
                            #"n_samples": int(cur.size),
                            #"duration_s": float(cur.size/sr_vm_cm) if cur.size>0 else None,
                            "data": cur.tolist() if cur.size>0 else []
                        },
                        "whisker_angle": {
                            "sampling_rate_Hz": int(sr_wa),
                            #"n_samples": int(wa.size),
                            #"duration_s": dur_wa,
                            "data": wa.tolist() if wa.size>0 else []
                        },

                        # Spikes
                        "ap_times": {
                            "relative_s": safe_list(ap_rel),
                            "absolute": [(t - start_time).total_seconds() if t is not None else None for t in ap_abs] if ap_abs else []
                        },
                        
                        # Rewards
                        "reward_times": {
                            "relative_s": safe_list(reward_rel),
                            "absolute": [(t - start_time).total_seconds() if t is not None else None for t in reward_abs] if reward_abs else []
                        },

                        "lick": {
                            "sampling_rate_Hz": 20000, 
                            "data": safe_list(lick_list[one_sweep]) if lick_list[one_sweep].size > 0 else [],
                            "power": safe_list(lick_power_list[one_sweep]) if lick_power_list[one_sweep].size > 0 else [],
                            "threshold": float(lick_threshold[one_sweep]) if not np.isnan(lick_threshold[one_sweep]) else None
                        },

                        # Trials
                        "trials": trials
                    }

                    sweeps.append(sweep_dict)



                # Create a new row for the session
                new_row = {
                    "Mouse Name": name,
                    "Cell_ID": Cell_ID[one_cell[0]],
                    "Ear tag": Ear_tag,
                    "Start date (dd.mm.yy)": start_date,
                    "End date": end_date,
                    "Sex_bin": sex,
                    "strain":  "C57BL/6JRj",
                    "mutations": "WT",
                    "Birth date": birth_date,
                    "licence": "VD-1628.6",
                    #"DG": "",
                    #"ExpEnd": "",
                    "Created on": Created_on, 
                    "Session": Cell_ID[one_cell[0]] + "_" + session_date,
                    "counter": counter,
                    "Session Date (yyymmdd)": session_date,
                    "Start Time (hhmmss)": start_time_hhmmss,
                    "Behavior Type": behaviortype,
                    "Session Type": session_type,
                    "task": task,
                    "sweeps": sweeps,
                    "Mouse Age (d)": age,
                    "Weight of Reference": ref_weight,
                    "Weight Session": Weight_Session 
                }

                # Append the new row to the DataFrame
                row = pd.Series(new_row)
                return row
    
    # Loop through each unique cell ID (each representing a dataset)          
    bar = tqdm(total=len(choice_cells), desc="Processing ")
    for cell_id in choice_cells:
        try:
            csv_data_row = to_data_frame(cell_id)
            bar.set_postfix_str(str(csv_data_row["Mouse Name"]))
            bar.update(1)

            # Create config file for the NWB conversion
            output_path, _ = converters.Initiation_nwb.files_to_config(
                subject_info=csv_data_row,
                output_folder=output_folder
            )

            # Create the initial NWB file 
            nwb_file = converters.Initiation_nwb.create_nwb_file_an(config_file=output_path)

            # ⏸️ Add interval data (like trials, epochs)
            converters.intervals_to_nwb.add_intervals_container(
                nwb_file=nwb_file,
                csv_data_row=csv_data_row
            )

            # 🧪 Add acquisition signals and unit data (spikes, etc.)
            converters.acquisition_and_unit_to_nwb.add_to_nwb_acquisition_and_units_containers(
                nwb_file=nwb_file,
                csv_data_row=csv_data_row
            )

            # ⚙️ Add behavior-related data (stimulations, responses, etc.)
            converters.behavior_to_nwb.add_behavior_container(
                nwb_file=nwb_file,
                csv_data_row=csv_data_row
            )

            # 🔎 Validate and save NWB file
            # Decide output folder based on the task type
            if csv_data_row["task"] == "WDT":
                output_folder_save = os.path.join(output_folder, "WDT")
            else:
                output_folder_save = os.path.join(output_folder, "No Task")
            os.makedirs(output_folder_save, exist_ok=True)

            # Save NWB file
            nwb_path = converters.nwb_saving.save_nwb_file(
                nwb_file=nwb_file,
                output_folder=output_folder_save
            )

            # Validate the NWB file using PyNWB validation
            with NWBHDF5IO(nwb_path, 'r') as io:
                nwb_errors = validate(io=io)

            # If validation errors occur, delete the invalid NWB file
            if nwb_errors:
                os.remove(nwb_path)
                raise RuntimeError("NWB validation failed: " + "; ".join(map(str, nwb_errors)))

            # Delete temporary config file
            if os.path.exists(output_path):
                os.remove(output_path)

            del csv_data_row
            gc.collect()

        except Exception as e:
            # Log the failure and move to the next cell
            failures.append((cell_id, str(e)))
            continue

    # Show failed conversions
    if len(failures) > 0:
        print(f"⚠️ Conversion completed except for:")
        for i, (id, error) in enumerate(failures):
            print(f"    - {id}: {error}")

    bar.close()

    # Clean up any leftover config files (.yaml)
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
    parser.add_argument("--choice_mouses", nargs='+', default=None, help="Mouse name(s) to process (e.g., LB010)")

    args = parser.parse_args()

    convert_data_to_nwb_PB(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        choice_mouses=args.choice_mouses
    )


    