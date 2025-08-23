
from pynwb.epoch import TimeIntervals   
import numpy as np

###############################################################
# Functions for converting intervals to NWB format 
###############################################################

def add_intervals_container(nwb_file,csv_data_row) -> None:
    """
    Populate `nwb_file.trials` from one pandas row.

    Args:
        nwb_file (pynwb.file.NWBFile): Target NWB file to which trials are added.
        csv_data_row (pandas.Series): Row containing trial data.

    Returns:
        None
    """

    # Extract trial information from csv_data_row
    Session_Type = str(csv_data_row.get("Session Type", ""))
    Session_Number = str(csv_data_row['counter'])
    sweep_data = csv_data_row["sweeps"]
    trial_onsets = np.array([])
    trial_onsets_relative = np.array([])
    whisker_stim = np.array([])
    perf = np.array([])
    whisker_stim_amplitude = np.array([])
    whisker_stim_time = np.array([])
    whisker_stim_time_relative = np.array([])
    Reward_flag = np.array([])
    Reward_time = np.array([])
    stim_type = np.array([])
    Sweep_IDs = np.array([])
    Sweep_Start_time = np.array([])
    Sweep_Stop_time = np.array([])
    trial_type = np.array([])
    lick_flag = np.array([])
    lick_threshold = np.array([])
    reward_available =  1 if csv_data_row["task"] == "WDT" else 0
    response_window = 2.0

    # each sweep is a dictionary that contains trial information
    for One_sweep in sweep_data:
        for One_trial in One_sweep["trials"]:
            trial_onsets = np.append(trial_onsets, One_trial["time_abs"])
            trial_onsets_relative = np.append(trial_onsets_relative, One_trial["time_rel_s"])
            whisker_stim = np.append(whisker_stim, One_trial["has_stim"])
            whisker_stim_amplitude = np.append(whisker_stim_amplitude, One_trial["amplitude"])
            perf = np.append(perf, One_trial["response"])
            Sweep_IDs = np.append(Sweep_IDs, One_sweep["Sweep Index"])
            Sweep_Start_time = np.append(Sweep_Start_time, One_sweep["Sweep Start Time"])
            Sweep_Stop_time = np.append(Sweep_Stop_time, One_sweep["Sweep Stop Time"])
            trial_type = np.append(trial_type, One_sweep["Sweep Type"])
            whisker_stim_time = np.append(whisker_stim_time, One_trial["stim_time_abs"] if One_trial["has_stim"] else np.nan)
            whisker_stim_time_relative = np.append(whisker_stim_time_relative, One_trial["stim_time_rel_s"] if One_trial["has_stim"] else np.nan)
            Reward_flag = np.append(Reward_flag, 1 if One_trial["reward"] else 0)
            Reward_time = np.append(Reward_time, One_trial["reward_time"] if One_trial["reward"] else np.nan)
            lick_flag = np.append(lick_flag, 1 if One_trial["lick"] else 0)
            lick_threshold = np.append(lick_threshold, One_sweep["lick"]["threshold"] if One_sweep.get("lick", {}).get("threshold", None) is not None else np.nan)
            stim_type = np.append(stim_type, One_trial["type"] if One_trial["type"] != "n.a" else np.nan)

    # Finalize data types
    trial_onsets = trial_onsets.astype(np.float64)
    trial_onsets_relative = trial_onsets_relative.astype(np.float64)
    whisker_stim = whisker_stim.astype(np.int64)
    whisker_stim_amplitude = whisker_stim_amplitude.astype(np.int64)
    perf = perf.astype(np.int64)
    whisker_stim_time = whisker_stim_time.astype(np.float64)
    whisker_stim_time_relative = whisker_stim_time_relative.astype(np.float64)
    Reward_flag = Reward_flag.astype(np.int64)
    Reward_time = Reward_time.astype(np.float64)
    Sweep_IDs = Sweep_IDs.astype(np.int64)
    Sweep_Start_time = Sweep_Start_time.astype(np.float64)
    trial_type = trial_type.astype(str)
    lick_flag = lick_flag.astype(np.int64)
    lick_threshold = lick_threshold.astype(np.float64)
    Sweep_Stop_time = Sweep_Stop_time.astype(np.float64)
    stim_type = stim_type.astype(str)


    # Define new trial columns
    new_columns = {
        'trial_time_relative': 'Relative time of the trial onset (s) relative to the corresponding sweep start time',
        'Sweep_ID': 'Unique identifier for each sweep',
        'Sweep_Start_time': 'Start time of the sweep',
        'Sweep_Stop_time': 'Stop time of the sweep',
        'trial_type': 'stimulus Whisker vs no stimulus trial',
        'whisker_stim': '1 if whisker stimulus delivered, else 0',
        'perf': '0= whisker miss; 1= whisker hit ; 2= correct rejection ; 3= false alarm',
        "whisker_stim_amplitude": "Amplitude of the whisker stimulation between 0 and 5",
        "whisker_stim_duration": "Duration of the whisker stimulation (ms)",
        "whisker_stim_time": "Time of whisker stimulation (s) relative to the session start time",
        "whisker_stim_time_relative": "Relative time of the whisker stimulation (s) relative to the corresponding sweep start time",
        "no_stim" : "1 if no stimulus delivered, else 0",
        "no_stim_time": "trial start time for no_stim=1 (catch trial) else NaN",
        "Reward_flag": "1 if reward else 0",
        "reward_available": "1 if reward available, else 0",
        "Reward_time": "Time of reward delivery (s) relative to the session start time",
        "response_window_start_time": "Start of response window",
        "response_window_stop_time": "Stop of response window",
        "lick_flag": "1 if lick detected, else 0",
        "lick_threshold": "Threshold for lick detection",
        "whisker_stim_type": "Type of stimulus presented",
        "Session_Type": "Type of session (e.g. trained, D1, Naive)",
        "Session_Number": "Unique identifier for each session",
    }

    # Add columns before inserting trials 
    for col, desc in new_columns.items():
        if (nwb_file.trials is None) or (col not in nwb_file.trials.colnames):
            nwb_file.add_trial_column(name=col, description=desc)

    # Add trials
    for i in range(len(trial_onsets)):
        nwb_file.add_trial(
            start_time=float(trial_onsets[i]),
            stop_time=float(trial_onsets[i]) + response_window,
            trial_time_relative=trial_onsets_relative[i],
            Sweep_ID=Sweep_IDs[i],
            Sweep_Start_time=Sweep_Start_time[i],
            Sweep_Stop_time=Sweep_Stop_time[i],
            trial_type='whisker_trial' if whisker_stim[i] else 'no_whisker_trial',
            whisker_stim=whisker_stim[i],
            perf=perf[i],
            whisker_stim_time=whisker_stim_time[i],
            whisker_stim_time_relative=whisker_stim_time_relative[i],
            whisker_stim_amplitude=whisker_stim_amplitude[i],
            whisker_stim_type=stim_type[i],
            whisker_stim_duration=str("1 (ms)"),
            no_stim= 1 if whisker_stim[i] == 0 else 0,
            no_stim_time= float(trial_onsets[i]) if whisker_stim[i] == 0 else np.nan,
            Reward_flag=1 if Reward_flag[i] else 0,
            Reward_time=Reward_time[i],
            reward_available=reward_available,
            response_window_start_time=float(trial_onsets[i]) + 0.05,
            response_window_stop_time=float(trial_onsets[i]) + 1,
            lick_flag=lick_flag[i],
            lick_threshold=lick_threshold[i],
            Session_Type=Session_Type,
            Session_Number=Session_Number,
        )

