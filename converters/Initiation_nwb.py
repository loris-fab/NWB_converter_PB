from datetime import datetime, timedelta
from typing import List, Optional
from dateutil.tz import tzlocal
from pynwb.file import Subject
from scipy.io import loadmat
from pynwb import NWBFile
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import yaml
import time
import os
import re

#############################################################################
# Function that creates the nwb file object using all metadata
#############################################################################

def create_nwb_file_an(config_file):
    """
    Create an NWBFile from a YAML configuration.

    Args:
        config_file (str): Absolute path to a YAML file containing
            "subject_metadata" and "session_metadata" sections.

    Returns:
        pynwb.file.NWBFile: The in-memory NWB file object.
        None: If the YAML cannot be read or required fields are missing.

    Raises:
        ValueError: If NWB file creation fails after parsing the config.
    """

    try:
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)
    except:
        print("Issue while reading the config file.")
        return

    subject_data_yaml = config['subject_metadata']
    session_data_yaml = config['session_metadata']

    # Subject info
    keys_kwargs_subject = ['age', 'age__reference', 'description', 'genotype', 'sex', 'species', 'subject_id',
                           'weight', 'date_of_birth', 'strain']
    kwargs_subject = dict()
    for key in keys_kwargs_subject:
        kwargs_subject[key] = subject_data_yaml.get(key)
        if kwargs_subject[key] is not None:
            kwargs_subject[key] = str(kwargs_subject[key])
    if 'date_of_birth' in kwargs_subject and kwargs_subject['date_of_birth'] != "na":
        date_of_birth = datetime.strptime(kwargs_subject['date_of_birth'], '%m/%d/%Y')
        date_of_birth = date_of_birth.replace(tzinfo=tzlocal())
        kwargs_subject['date_of_birth'] = date_of_birth
    else:
        kwargs_subject.pop('date_of_birth', None)
    subject = Subject(**kwargs_subject)

    # Session info
    keys_kwargs_nwb_file = ['session_description', 'identifier', 'session_id', 'session_start_time',
                            'experimenter', 'experiment_description', 'institution', 'keywords',
                            'notes', 'pharmacology', 'protocol', 'related_publications',
                            'source_script', 'source_script_file_name',  'surgery', 'virus',
                            'stimulus_notes', 'slices', 'lab']

    kwargs_nwb_file = dict()
    for key in keys_kwargs_nwb_file:
        kwargs_nwb_file[key] = session_data_yaml.get(key)
        if kwargs_nwb_file[key] is not None:
            if not isinstance(kwargs_nwb_file[key], list):
                kwargs_nwb_file[key] = str(kwargs_nwb_file[key])
    if 'session_description' not in kwargs_nwb_file:
        print('session_description is needed in the config file.')
        return
    if 'identifier' not in kwargs_nwb_file:
        print('identifier is needed in the config file.')
        return
    if 'session_start_time' not in kwargs_nwb_file:
        print('session_start_time is needed in the config file.')
        return
    else:

        if kwargs_nwb_file['session_start_time'][-2:] == '60': # handle non leap seconds
            kwargs_nwb_file['session_start_time'] = kwargs_nwb_file['session_start_time'][0:-2] + '59'
        session_start_time = datetime.strptime(kwargs_nwb_file['session_start_time'], '%Y%m%d %H%M%S')
        session_start_time = session_start_time.replace(tzinfo=tzlocal())
        kwargs_nwb_file['session_start_time'] = session_start_time
    if 'session_id' not in kwargs_nwb_file:
        kwargs_nwb_file['session_id'] = kwargs_nwb_file['identifier']


    # Create NWB file object
    kwargs_nwb_file['subject'] = subject
    kwargs_nwb_file['file_create_date'] = datetime.now(tzlocal())

    nwb_file = NWBFile(**kwargs_nwb_file)
    if nwb_file is None:
        raise ValueError("❌ NWB file creation failed. Please check the provided metadata in the config file.")

    return nwb_file


#############################################################################
# Function that creates the config file for the NWB conversion
#############################################################################


def files_to_config(subject_info,output_folder="data"):
    """
    Build a session/subject NWB config from one CSV row and save it as YAML.

    Args:
        csv_data_row (pandas.Series or Mapping): Row with fields such as
            "Mouse Name", "Session", "Session Date (yyymmdd)", "Start Time (hhmmss)",
            "Behavior Type", etc.
        output_folder (str or pathlib.Path): Folder where the YAML file is saved.

    Returns:
        tuple[str, dict]: (output_path to the written YAML file, in-memory config dict).
    """
    related_publications = 'n.a (no publication yet)'
    

    mouse = str(subject_info['Mouse Name'])
    session_name = subject_info['Session']


    ###  Session metadata extraction  ###

    ### behavioral type
    behavior_type = str(subject_info.get("Behavior Type", "Unknown").strip())

    ### Experiment_description
    Session_Type = subject_info.get("Session Type", "")
    if pd.isna(Session_Type) or str(Session_Type).strip().lower() in ["", "nan"]:
        Session_Type = "Naive"
    if Session_Type == "Trained" or Session_Type == "D1":
        Session_Type = int(1)
        session_description = "ephys " + behavior_type + " mouse: the mouse was rewarded with a drop of water if it licked within 1 s following a whisker stimulus (go trials) but not in the absence of the stimulus (no-go trials). Membrane potential recording was performed in the medial prefrontal cortex using patch-clamp whole-cell recording with glass pipette (4-7 MOhms). WDT session = " + str(subject_info['counter'])
    elif Session_Type == "Naive" :
        Session_Type = int(0)
        session_description = "ephys " + behavior_type + " mouse: the mouse was habituated to sit still while head-restrained. Membrane potential recording was performed in the medial prefrontal cortex using patch-clamp whole-cell recording with glass pipette (4-7 MOhms) while single-whisker stimuli were delivered at random times."


    ref_weight = subject_info.get("Weight of Reference", "")
    if pd.isna(ref_weight) or str(ref_weight).strip().lower() in ["", "nan"]:
        ref_weight = 'na'
    else:
        try:
            ref_weight = float(ref_weight)
        except Exception:
            ref_weight = 'na'


    experiment_description = {
    'reference_weight': ref_weight,
    "session_type": "ephys_session " + str(subject_info.get("Session Type", "")),
    "behavior Task": behavior_type,
    'wh_reward': 1 if Session_Type == 1 else 0,
    #'aud_reward': ?,
    'reward_proba': 1 if Session_Type == 1 else 0,
    'wh_stim_amps': '5',
    "Session Number" : str(subject_info['counter']),
    #'lick_threshold': ?,
    #'no_stim_weight': ?,
    #'wh_stim_weight': ?,
    #'aud_stim_weight': ?,
    #'camera_flag': ?,
    #'camera_freq': ?,
    #'camera_exposure_time': camera_exposure_time,
    #'each_video_duration': ?,
    #'camera_start_delay': ?,
    #'artifact_window': ?,
    'licence': str(subject_info.get("licence", "")).strip(),
    'ear tag': str(subject_info.get("Ear tag", "")).strip(),
    #"Software and Algorithms": "?",
    'Ambient noise': '80 dB',
    }
    ### Experimenter
    experimenter = subject_info["User (user_userName)"]
   
    ### Session_id, identifier, institution, keywords
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology", "patch-clamp"] 

    ### Session start time
    session_start_time = str(subject_info["Session Date (yyymmdd)"])+" " + str(subject_info["Start Time (hhmmss)"])

    ###  Subject metadata extraction  ###

    ### Birth date and age calculation
    if subject_info["Birth date"] != "Unknown":
        birth_date = pd.to_datetime(subject_info["Birth date"], dayfirst=True).strftime('%m/%d/%Y')
    else:
        birth_date = 'na'

    age = subject_info["Mouse Age (d)"] 
    if age == "Unknown":
        age = 'na'
    else:
        age = f"P{age}D"


    ### Genotype 
    genotype = subject_info.get("mutations", "")
    if pd.isna(genotype) or str(genotype).strip().lower() in ["", "nan"]:
        genotype = "WT"
    genotype = str(genotype).strip()


    ### weight
    weight = subject_info.get("Weight Session", "")
    if pd.isna(weight) or str(weight).strip().lower() in ["", "nan"]:
        weight = 'na'
    else:
        try:
            weight = float(weight)
        except Exception:
            weight = 'na'

    ### Behavioral metadata extraction 
    camera_flag = 1

    # Construct the output YAML path
    config = {
        'session_metadata': {
            'experiment_description' : experiment_description,
            'experimenter': experimenter,
            'identifier': identifier,
            'institution': "Ecole Polytechnique Federale de Lausanne",
            'keywords': keywords,
            'lab' : "Laboratory of Sensory Processing",
            'notes': "Single-neuron membrane potential recording in the medial prefrontal cortex of awake behaving mice.",
            'pharmacology': 'na',
            'protocol': 'na',
            'related_publications': related_publications,
            'session_description': session_description ,
            'session_id': session_id,
            'session_start_time': session_start_time,
            'slices': "na", 
            'source_script': 'na',
            'source_script_file_name': 'na',
            'stimulus_notes': 'na',
            'surgery': 'na',
            'virus': 'na',

        },
        'subject_metadata': {
            'age': age,
            'age__reference': 'birth',
            'date_of_birth': birth_date,
            'description': mouse,
            'genotype': genotype,
            'sex': subject_info.get("Sex_bin", "").upper().strip(),
            'species': "Mus musculus",
            'strain': str(subject_info.get("strain", "").strip()),
            'subject_id': mouse,
            'weight': weight,

        },
    }

    # save config
    output_path = os.path.join(output_folder, f"{session_name}_config.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return output_path, config


#############################################################################
# Function that creates the csv file for the NWB conversion
#############################################################################

def files_to_dataframe(mat_file, choice_mouses,dataframe_subject):
    """
    Load a preprocessed MATLAB .mat file and subject metadata into a unified pandas DataFrame.

    Args:
        mat_file (str | Path): Path to the .mat file containing "Data_Full/*" groups.
        choice_mouses (list[str] | None): List of mouse names to include. If None, all are included.
        dataframe_subject (pd.DataFrame): Subject/session metadata table with required columns
            (e.g. 'Mouse Name', 'Birth date', 'Ear tag', 'Weight of Reference', etc.).

    Returns:
        pd.DataFrame: One row per mouse/session with metadata and a 'sweeps' column.
            Each 'sweeps' entry is a list of dicts containing:
            - Sweep metadata (index, start/stop time, type)
            - Signals (membrane potential, current monitor, whisker angle, lick, etc.)
            - Spike, reward, and trial information.
    """

    columns = ['Mouse Name', 'User (user_userName)', 'Cell_ID', 'Ear tag',
        'Start date (dd.mm.yy)', 'End date', 'Sex_bin', 'strain', 'mutations',
        'Birth date', 'licence', 'DG', 'ExpEnd', 'Created on', 'Session',
        'Session Date (yyymmdd)', 'Start Time (hhmmss)', 'Behavior Type',
        'Session Type','sweeps', 'Opto Session', 'Mouse Age (d)', 'Weight of Reference',
        'Weight Session', "task", "counter"]
    csv_data = pd.DataFrame(columns=columns)
    csv_data.columns = csv_data.columns.str.strip()
    ##dataframe_subject.columns = dataframe_subject.columns.str.strip()
    #print(dataframe_subject.columns)

    def reward_in_trial(
        reward_times: List[datetime],
        trial_start: datetime,
        trial_stop: datetime,
        has_reward: bool,
        start_time: datetime
    ) -> Optional[datetime]:
        """
        Return the reward datetime if it falls within [trial_start, trial_stop].

        Parameters
        ----------
        reward_times : list of datetime
            Reward timestamps.
        trial_start : datetime
            Start time of the trial.
        trial_stop : datetime
            Stop time of the trial.
        has_reward : bool
            Whether the trial should have a reward.

        Returns
        -------
        float | None
            The reward time in seconds if exactly one is found, None if has_reward=False.

        Raises
        ------
        ValueError
            If has_reward=True but no reward or multiple rewards are found in the interval.
        """
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



    def to_text(x):
        a = np.array(x)
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode('utf-8', errors='ignore')
        if isinstance(x, (str, np.str_)):
            return str(x)

        if a.dtype.kind in ('i','u'):
            return ''.join(chr(int(v)) for v in a.ravel() if int(v) > 0)

        if a.dtype.kind == 'S':
            return b''.join(a.ravel().tolist()).decode('utf-8', errors='ignore')

        if isinstance(x, (np.integer, int)):
            return chr(int(x))

        return str(x)

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
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            if x.size == 0 or (x.shape == (2,) and np.array_equal(x, [0, 0])):
                return []
        arr = np.atleast_1d(x)
        return arr.tolist()

    def _decode_chars(x):
        a = np.array(x)
        return ''.join(chr(int(v)) for v in a.ravel() if int(v) > 0)

    def read_cell_strings(f, path):
        refs = f[path][()]
        return np.array([_decode_chars(f[r][()]) for r in np.ravel(refs)])

    def read_birth_dates(f, path):
        refs = f[path][()]
        out = []
        for r in np.ravel(refs):
            #vals = [int(f[r2][()][()]) for r2 in np.ravel(f[r][()])]
            vals = [int(f[r2][()].item()) for r2 in np.ravel(f[r][()])]
            # format dd.mm.yy 
            out.append(f"{vals[2]:02d}.{vals[1]:02d}.{vals[0]}")
        return np.array(out)

    def read_sweep_start_times(f, path):
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
        """Décoder string MATLAB depuis divers formats (bytes, char-codes, nested)."""
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

    with tqdm(total=5) as pbar:

        with h5py.File(mat_file, 'r') as f:


            Cell_ID        = read_cell_strings(f, 'Data_Full/Cell_ID')
            mouses_name    = np.array([s.split('_')[0] for s in Cell_ID])
            #birth_dates    = read_birth_dates(f, 'Data_Full/Mouse_DateOfBirth')
            Mouse_Sex      = read_cell_strings(f, 'Data_Full/Mouse_Sex')
            sessiontypes   = read_cell_strings(f, 'Data_Full/Session_Type')
            Sweep_type  = read_cell_strings(f, 'Data_Full/Sweep_Type')
            Cell_Depth = np.array(f['Data_Full/Cell_Depth'][()]).squeeze()

            

            # Step 1: Mouse metadata loaded
            time.sleep(1)
            tqdm.write("Loading mouse metadata ...")
            pbar.update(1)

            # --- SWEEPS ----
            # Start times
            sweep_start_times = read_sweep_start_times(f, 'Data_Full/Sweep_StartTime')

            # Signaux
            mp_refs   = f['Data_Full/Sweep_MembranePotential'][()]
            cm_refs   = f['Data_Full/Sweep_CurrentMonitor'][()]
            wa_refs   = f['Data_Full/Sweep_WhiskerAngle'][()]

            # Step 2: Signal data loaded
            time.sleep(1)
            tqdm.write("Loading sweep signal data (1/3) ...")
            pbar.update(1)

            lick_refs   = f['Data_Full/Sweep_Lick'][()]
            lick_power_refs = f['Data_Full/Sweep_Lick_Power'][()]
            lick_threshold  = np.array(f['Data_Full/Sweep_Lick_Threshold'][()]).squeeze() 

            vm_list   = read_cell_numeric_vectors(f, mp_refs)  # 487 éléments, np.array par sweep
            cur_list  = read_cell_numeric_vectors(f, cm_refs)
            wa_list   = read_cell_numeric_vectors(f, wa_refs)
            lick_list        = read_cell_numeric_vectors(f, lick_refs)
            lick_power_list  = read_cell_numeric_vectors(f, lick_power_refs)

            # Step 3: Signal data loaded
            time.sleep(1)
            tqdm.write("Loading sweep signal data (2/3) ...")
            pbar.update(1)

            # AP times
            ap_refs   = f['Data_Full/Sweep_AP_Times'][()]
            ap_list   = read_cell_numeric_vectors(f, ap_refs)

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
            sr_vm_cm = 20000
            sr_wa    = 200

            # Step 5: Sweep data loaded
            time.sleep(1)
            tqdm.write("Loading sweep behavior data ...")
            pbar.update(1)

    # Verify if choice_mouses is correct
    if choice_mouses is None:
        choice_mouses = np.unique(mouses_name)
    else:
        missing = [name for name in choice_mouses if name not in np.unique(mouses_name)]
        if missing:
            raise ValueError(f"Mouse name(s) not found in csv file: {missing}")
        
    index_mouses_choices1 = [np.where(mouses_name == mouse)[0] for mouse in choice_mouses if mouse in mouses_name]
    index_mouses_choices = np.concatenate(index_mouses_choices1)


    # Indices Session (cell_ID) for NWB
    unique_ids, _ = np.unique(Cell_ID, return_index=True)
    cell = [np.arange(len(Cell_ID))[Cell_ID == unique_id] for unique_id in unique_ids]
    # Itèration over the indices of the choice mouses to collect data in the dataframe
    for one_cell in tqdm(cell, total=len(index_mouses_choices1), desc="Creating a unified DataFrame : Adding Mouses Cells ..."):
        if one_cell[0] in index_mouses_choices:
            name = mouses_name[one_cell[0]]
            sex = Mouse_Sex[one_cell[0]]
            session_type = sessiontypes[one_cell[0]]
            #behaviortype = " & ".join(np.unique(Sweep_type[one_cell]))
            if session_type not in ["Trained", "D1", "Naive"]:
                raise ValueError(f"Unknown session type: {session_type}. Expected 'Trained', 'D1', or 'Naive'.") 
            behaviortype = "Whisker rewarded (WR+)" if (session_type == "Trained" or session_type == "D1") else "No Task"  
            start_time_hhmmss = sweep_start_times[one_cell[0]].strftime("%H%M%S") if not pd.isnull(sweep_start_times[one_cell[0]]) else None
            start_time = sweep_start_times[one_cell[0]] if not pd.isnull(sweep_start_times[one_cell[0]]) else None

            row = dataframe_subject[dataframe_subject['Mouse Name'] == name]
            date_str = str(row['Birth date'].iloc[0]).strip() 
            birth_date = date_str.replace("/", ".")
            date_session_str = str(row['Start date'].iloc[0]).strip()
            date_session_str = date_session_str.replace("/", ".")
            Weight_Session = float(row['Weight Experimet day'].iloc[0])
            ref_weight = float(row['Weight of Refence'].iloc[0]) 
            Ear_tag = str(row['Ear tag'].iloc[0]).strip()
            Created_on = str(row['Created on'].iloc[0]).strip()
            user = str(row['User (user_userName)'].iloc[0]).strip()
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
                    "Sweep Index": int(one_sweep),
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
                "User (user_userName)": user,
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
            csv_data.loc[len(csv_data)] = new_row
    return csv_data

def remove_rows_by_mouse_name(df, mouse_name):
    return df[df["Mouse Name"] != mouse_name]

def remove_nwb_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.nwb'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path}: {e}")