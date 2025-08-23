from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries
from pynwb.base import TimeSeries
import numpy as np


################################################################
# Functions for adding behavior container to NWB file
################################################################

def add_behavior_container(nwb_file,csv_data_row):
    """
    Adds a 'behavior' container to the NWB file from the loaded .mat data.

   Args:
       nwb_file (pynwb.file.NWBFile): Target NWB file to which behavior data is added.
       csv_data_row (pandas.Series): Row containing behavior data fields.
       
    return: None
    """
    # Extract behavior data 
    sweep_data = csv_data_row["sweeps"]
    trial_onsets = np.array([])
    whisker_stim = np.array([])
    perf = np.array([])
    whisker_stim_amplitude = np.array([])
    whisker_stim_time = np.array([])
    reward_available =  1 if csv_data_row["task"] == "WDT" else 0
    Sweep_Start_time = np.array([])
    Sweep_Stop_time = np.array([])
    lick_flag = np.array([])
    reward_onset = np.array([])
    whisker_angle = list()
    PiezoLickSignal = list()
    
    for One_sweep in sweep_data:
        Sweep_Start_time = np.append(Sweep_Start_time, One_sweep["Sweep Start Time"])
        Sweep_Stop_time = np.append(Sweep_Stop_time, One_sweep["Sweep Stop Time"])
        whisker_angle.append((One_sweep["whisker_angle"]["data"], One_sweep["Sweep Start Time"]))
        PiezoLickSignal.append((One_sweep["lick"]["data"], One_sweep["Sweep Start Time"]))
        for One_trial in One_sweep["trials"]:
            trial_onsets = np.append(trial_onsets, One_trial["time_abs"])
            whisker_stim = np.append(whisker_stim, One_trial["has_stim"])
            whisker_stim_amplitude = np.append(whisker_stim_amplitude, One_trial["amplitude"])
            perf = np.append(perf, One_trial["response"])
            whisker_stim_time = np.append(whisker_stim_time, One_trial["stim_time_abs"] if One_trial["has_stim"] else np.nan)
            reward_onset = np.append(reward_onset, One_trial["reward_time"] if One_trial["reward"] else np.nan)
            lick_flag = np.append(lick_flag, 1 if One_trial["lick"] else 0)

    # Finalize data types
    trial_onsets = trial_onsets.astype(np.float64)
    whisker_stim = whisker_stim.astype(np.int64)
    whisker_stim_amplitude = whisker_stim_amplitude.astype(np.int64)
    perf = perf.astype(np.int64)
    whisker_stim_time = whisker_stim_time.astype(np.float64)
    reward_onset = reward_onset.astype(np.float64)
    Sweep_Start_time = Sweep_Start_time.astype(np.float64)
    lick_flag = lick_flag.astype(np.int64)
    Sweep_Stop_time = Sweep_Stop_time.astype(np.float64)


    # Created behavior processing module
    bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    ###############################################
    ### Add behavioral events                   ###
    ###############################################

    # Create a BehavioralEvents container
    behavior_events = BehavioralEvents(name='BehavioralEvents')
    bhv_module.add_data_interface(behavior_events)


    # --- TRIAL ONSETS ---
    ts_trial = TimeSeries(
        name='TrialOnsets',
        data=np.ones_like(trial_onsets),
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps marking the onset of each trial.',
        comments='time start of each trial',
        rate = None,
    )
    behavior_events.add_timeseries(ts_trial)

    """
    # --- ReactionTimes  ---
    lick_time1 = lick_time[lick_time > 0]
    ts_reaction = TimeSeries(
        name='ReactionTimes',
        data=np.ones_like(lick_time1),
        unit='n.a.',
        timestamps=lick_time1,
        description='Timestamps of response-time defined as lick-onset occurring after trial onset.',
        comments='reaction time from PiezoLickSignal',
        rate = None,
    )
    behavior_events.add_timeseries(ts_reaction)
    """

    # --- STIMULATION FLAGS (stim et flag) ---    

    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=whisker_stim_amplitude,
        timestamps=trial_onsets,
        unit='code',
        description='Timestamps marking the amplitude of whisker stimulation for each trial',
        comments='Whisker stimulation amplitudes are encoded as integers: 0 = no stimulus (Catch trial), 1 = deflection of the C2 whisker, and higher values indicate increasing stimulation amplitudes.',
        rate = None,
    )
    behavior_events.add_timeseries(ts_stim_flags)
    

    # ---- "ResponseType" ------

    response_labels_ts = TimeSeries(
        name='ResponseType',
        data=perf,
        unit='code',
        timestamps=trial_onsets,
        description = "Response type for each trial",
        comments='trial responses: 0 = MISS, 1 = HIT, 2 = CR (Correct Rejection), 3 = FA (False Alarm), 4 = Unlabeled (no assigned response).',

    )

    behavior_events.add_timeseries(response_labels_ts)


    # ---- "Whisker_hit_trial" ------
    ts_whisker_hit = TimeSeries(
        name='whisker_hit_trial',
        data=(perf == 1).astype(int), 
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps for whisker_hit_trial',
        comments='time of each whisker_hit_trial event.',
        rate=None,
    )
    behavior_events.add_timeseries(ts_whisker_hit)

    # --- whisker_miss_trial ----
    ts_whisker_miss = TimeSeries(
        name='whisker_miss_trial',
        data=(perf == 0).astype(int), 
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps for whisker_miss_trial',
        comments='time of each whisker_miss_trial event.',
        rate=None,
    )
    behavior_events.add_timeseries(ts_whisker_miss)

    # ---- correct_rejection_trial ----
    ts_correct_rejection = TimeSeries(
        name='correct_rejection_trial',
        data=(perf == 2).astype(int),  
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps for correct_rejection_trial',
        comments='time of each correct_rejection_trial event.',
        rate=None,
    )
    behavior_events.add_timeseries(ts_correct_rejection)

    # ---- false_alarm_trial ----
    ts_false_alarm = TimeSeries(
        name='false_alarm_trial',
        data=(perf == 3).astype(int),  
        unit='n.a.',
        timestamps=trial_onsets,
        description='Timestamps for false_alarm_trial',
        comments='time of each false_alarm_trial event.',
        rate=None,
    )
    behavior_events.add_timeseries(ts_false_alarm)

    if reward_available:
        # --- reward_onset ---
        ts_reward_onset = TimeSeries(
            name='reward_onset',
            data=np.ones_like(reward_onset[reward_onset > 0]),
            timestamps=reward_onset[reward_onset > 0],
            unit='n.a.',
            description = "Timestamps for reward-times",
            comments = "time of each reward delivery event.",
        )
        behavior_events.add_timeseries(ts_reward_onset)

    # --- Sweep start time ---
    ts_sweep_start = TimeSeries(
        name='sweep_start',
        data=np.ones_like(Sweep_Start_time),
        timestamps=Sweep_Start_time,
        unit='n.a.',
        description = "Timestamps for sweep start times",
        comments = "time of each sweep start event.",
    )
    behavior_events.add_timeseries(ts_sweep_start)

    # --- Sweep stop time ---
    ts_sweep_stop = TimeSeries(
        name='sweep_stop',
        data=np.ones_like(Sweep_Stop_time),
        timestamps=Sweep_Stop_time,
        unit='n.a.',
        description = "Timestamps for sweep stop times",
        comments = "time of each sweep stop event.",
    )
    behavior_events.add_timeseries(ts_sweep_stop)

    #########################################################
    ### Add continuous traces  ###
    #########################################################
    
    # Create BehavioralTimeSeries container
    bts = bhv_module.data_interfaces.get('BehavioralTimeSeries')
    if bts is None:
        bts = BehavioralTimeSeries(name='BehavioralTimeSeries')
        bhv_module.add(bts)


    # Add continuous traces (whisker_angle, PiezoLickSignal)
    for index, i in enumerate([whisker_angle, PiezoLickSignal]):
        all_data = []
        all_timestamps = []
        for data, sweep_start in i:
            if len(data) > 0:
                data = np.array(data)
                n_samples = len(data)
                sampling_rate = 200.0 if index == 0 else 20000.0
                sweep_time = np.arange(n_samples) / sampling_rate + sweep_start
                all_data.append(data)
                all_timestamps.append(sweep_time)
            elif len(data) == 0 or data is None:
                continue
        if len(all_data) == 0 or len(all_timestamps) == 0:
            continue
        all_data = np.concatenate(all_data)
        all_timestamps = np.concatenate(all_timestamps)
 

        whisker_ts = TimeSeries(
            name="whisker_angle" if index == 0 else "piezo_lick_signal",
            data=all_data,
            unit='a.u' if index == 0 else 'V',
            timestamps=all_timestamps,
            description = "Whisker angle trace across aligned video_onsets." if index == 0 else "Lick signal over time (V, Sampling rate = 20 000 Hz)",
            comments = "the whisker angle is defined as the angle between the whisker shaft and the midline of the head." if index == 0 else "PiezoLickSignal is the continuous electrical signal recorded from the piezo film attached to the water spout to detect when the mouse contacts the water spout with its tongue."
        )
        bts.add_timeseries(whisker_ts)


    return None

