import numpy as np
from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries
from pynwb import TimeSeries
import pandas as pd


################################################################
# Functions for adding behavior container to NWB file
################################################################

def add_behavior_container(nwb_file,csv_data_row):
    """
    Adds a 'behavior' container to the NWB file from the loaded .mat data.

   Args:
       nwb_file (pynwb.file.NWBFile): Target NWB file to which behavior data is added.
       csv_data_row (pandas.Series | Mapping): Row containing behavior data fields.
       Rewarded (bool): if the mouse has a rewarded task.
       
    return: None
    """



    # --- Extract behavior data ---

    sweep_data = csv_data_row["sweeps"]
    trial_onsets = np.array([])
    trial_onsets_relative = np.array([])
    whisker_stim = np.array([])
    perf = np.array([])
    whisker_stim_amplitude = np.array([])
    whisker_stim_time = np.array([])
    whisker_stim_time_relative = np.array([])
    reward_available = np.array([])
    stim_type = np.array([])
    Sweep_IDs = np.array([])
    Sweep_Start_time = np.array([])
    Sweep_Stop_time = np.array([])
    lick_flag = np.array([])

    response_window = 2.0
    """
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
            whisker_stim_time = np.append(whisker_stim_time, One_trial["stim_time_abs"] if One_trial["has_stim"] else np.nan)
            whisker_stim_time_relative = np.append(whisker_stim_time_relative, One_trial["stim_time_rel_s"] if One_trial["has_stim"] else np.nan)
            reward_available = np.append(reward_available, 1 if One_trial["reward"] else 0)
            lick_flag = np.append(lick_flag, 1 if One_trial["lick"] else 0)
            stim_type = np.append(stim_type, One_trial["type"])


    trial_onsets = trial_onsets.astype(np.float64)
    whisker_stim = whisker_stim.astype(np.int64)
    whisker_stim_amplitude = whisker_stim_amplitude.astype(np.int64)
    perf = perf.astype(np.int64)
    whisker_stim_time = whisker_stim_time.astype(np.float64)
    whisker_stim_time_relative = whisker_stim_time_relative.astype(np.float64)
    reward_available = reward_available.astype(np.int64)
    Sweep_IDs = Sweep_IDs.astype(np.int64)
    Sweep_Start_time = Sweep_Start_time.astype(np.float64)
    lick_flag = lick_flag.astype(np.int64)
    Sweep_Stop_time = Sweep_Stop_time.astype(np.float64)
    stim_type = stim_type.astype(str)
    """


    # To supress
    lick_time = np.asarray(list(map(float, csv_data_row["lick_time"].split(";"))))
    PiezoLickSignal = np.asarray(list(map(float, csv_data_row["PiezoLickSignal"].split(";"))))
    reward_onset = np.asarray(list(map(float, csv_data_row["reward_onset"].split(";"))))

    # 1. Created behavior processing module
    bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    ###############################################
    ### Add behavioral events                    ###
    ###############################################


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


    # --- STIMULATION FLAGS (stim et flag) ---    

    ts_stim_flags = TimeSeries(
        name='StimFlags',
        data=whisker_stim_amplitude[whisker_stim_amplitude > 0],
        timestamps=whisker_stim_time[whisker_stim_time > 0],
        unit='n.a.',
        description='Timestamps marking the amplitude of whisker stimulation for each trial',
        comments='Whisker stimulation amplitudes are encoded as integers: 0 = no stimulus (Catch trial), 1 = deflection of the C2 whisker.',
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


    # --- reward_onset ---
    ts_reward_onset = TimeSeries(
        name='reward_onset',
        data=np.ones_like(reward_onset),
        timestamps=reward_onset,
        unit='n.a.',
        description = "Timestamps for reward-times",
        comments = "time of each reward delivery event.",
    )
    behavior_events.add_timeseries(ts_reward_onset)
    #########################################################
    ### Add continuous traces  ###
    #########################################################
    bts = bhv_module.data_interfaces.get('BehavioralTimeSeries')
    if bts is None:
        bts = BehavioralTimeSeries(name='BehavioralTimeSeries')
        bhv_module.add(bts)

    if pd.notna(csv_data_row["EMG"]):
        EMG = list(map(float, csv_data_row["EMG"].split(";")))
    else:
        EMG = None

    # ---------- EMG ----------
    RATE = 2000.0
    UNIT = "V"

    if EMG is not None :
        es_emg = TimeSeries(
            name="ElectricalSeries_EMG",
            data=EMG,
            starting_time=0.0,
            rate=RATE,
            unit=UNIT,
            description="EMG recorded differentially from 2 electrodes, resulting in a single EMG signal",
            comments = "2000 Hz, in V."
        )
        bts.add_timeseries(es_emg)

    es_PiezoLickSignal = TimeSeries(
    name="ElectricalSeries_PiezoLickSignal",
    data=PiezoLickSignal,
    starting_time=0.0,
    rate=RATE,
    unit=UNIT,
    description="Lick signal over time (V, Sampling rate = 2000 Hz)",
    comments="PiezoLickSignal is the continuous electrical signal recorded from the piezo film attached to the water spout to detect when the mouse contacts the water spout with its tongue."
    )
    bts.add_timeseries(es_PiezoLickSignal)

    return None

