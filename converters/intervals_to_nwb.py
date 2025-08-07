

import numpy as np
import h5py
from pynwb.epoch import TimeIntervals   

###############################################################
# Functions for converting intervals to NWB format for AN sessions
###############################################################

def add_intervals_container_Rewarded(nwb_file, csv_data_row):
    """
    Add detailed trial information to the NWBFile 
    """

    sweep_data = csv_data_row["sweeps"]
    trial_onsets = np.array([])
    stim_indices = np.array([])
    stim_amps = np.array([])
    response_data = np.array([])
    Sweep_IDs = np.array([])
    Sweep_Start_time = np.array([])
    Sweep_Stop_time = np.array([])
    sweep_type = np.array([])
    for One_sweep in sweep_data:
        print(type(One_sweep))
        print(One_sweep.keys())
        for One_trial in One_sweep["trials"]:
            print(type(One_trial))
            print(One_trial.keys())
            trial_onsets = np.append(trial_onsets, One_trial["time_abs"].second)
            stim_indices = np.append(stim_indices, One_trial["has_stim"])
            stim_amps = np.append(stim_amps, One_trial["amplitude"])
            response_data = np.append(response_data, One_trial["response"])
            Sweep_IDs = np.append(Sweep_IDs, One_sweep["Sweep Index"])
            Sweep_Start_time = np.append(Sweep_Start_time, One_sweep["Sweep Start Time"].second)
            Sweep_Stop_time = np.append(Sweep_Stop_time, One_sweep["Sweep Stop Time"].second)
            sweep_type = np.append(sweep_type, One_sweep["Sweep Type"])

    trial_onsets = trial_onsets.astype(np.float64)
    stim_indices = stim_indices.astype(np.int64)
    stim_amps = stim_amps.astype(np.int64)

    response_labels = np.full(response_data.shape, '', dtype=object)
    response_labels[response_data == 1] = "hit"
    response_labels[response_data == 0] = "miss"
    response_labels[response_data == 2] = "CR"
    response_labels[response_data == 3] = "FA"

    Sweep_IDs = Sweep_IDs.astype(np.int64)
    Sweep_Start_time = Sweep_Start_time.astype(np.float64)
    Sweep_Stop_time = Sweep_Stop_time.astype(np.float64)
    sweep_type = sweep_type.astype(str)

    # --- Define new trial columns ---
    new_columns = {
        'trial_type': 'Stimulus Whisker vs no stimulation trial',
        'whisker_stim': '1 if whisker stimulus delivered, else 0',
        'whisker_stim_amplitude': 'Amplitude of whisker stimulus',
        'Sweep_ID': 'ID of the sweep where the trial occurred',
        'Sweep_Start_time': 'Start time of the sweep',
        'Sweep_Stop_time': 'Stop time of the sweep',
        #'reward_available': 'Whether reward could be earned (1 = yes)',
        #'response_window_start_time': 'Start of response window',
        'ResponseType': 'Trial outcome label (Hit, Miss, etc.)',
        #'lick_time': 'Timestamps of licks in trial',
        'Sweep_ID': 'Sweep ID where trial occurred',
    }

    # --- Add columns before inserting trials ---
    if nwb_file.trials is None:
        # This creates an empty trial table
        for col, desc in new_columns.items():
            nwb_file.add_trial_column(name=col, description=desc)

    else:
        # Add only missing columns if table already exists
        for col, desc in new_columns.items():
            if col not in nwb_file.trials.colnames:
                nwb_file.add_trial_column(name=col, description=desc)

    # --- Add trials ---
    for i in range(len(trial_onsets)):
        nwb_file.add_trial(
            start_time=float(trial_onsets[i]),
            stop_time=float(trial_onsets[i]) + 1.0,

            trial_type=sweep_type[i],
            whisker_stim=int(stim_indices[i]),
            whisker_stim_amplitude=float(stim_amps[i]),
            Sweep_ID=int(Sweep_IDs[i]),
            Sweep_Start_time=float(Sweep_Start_time[i]),
            Sweep_Stop_time=float(Sweep_Stop_time[i]),
            #reward_available=1,
            #response_window_start_time=float(reaction_abs[i]),
            ResponseType=response_labels[i],
            #lick_time=lick_time_per_trial[i]

        )

