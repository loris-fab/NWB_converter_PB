from datetime import datetime, timedelta
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
    Create an NWB file object using all metadata containing in YAML file

    Args:
        config_file (str): Absolute path to YAML file containing the subject experimental metadata

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
    related_publications = 'n.a (no publication yet)'
    

    mouse = str(subject_info['Mouse Name'])
    date = subject_info['Session Date (yyymmdd)']
    session_name = subject_info['Session']


    ###  Session metadata extraction  ###

    ### Experiment_description
    #date_experience = pd.to_datetime(date, format='%Y%m%d')

    ref_weight = subject_info.get("Weight of Reference", "")
    if pd.isna(ref_weight) or str(ref_weight).strip().lower() in ["", "nan"]:
        ref_weight = 'na'
    else:
        try:
            ref_weight = float(ref_weight)
        except Exception:
            ref_weight = 'na'

    experiment_description = {
    #'reference_weight': ref_weight,
    #'wh_reward': ?,
    #'aud_reward': ?,
    #'reward_proba': ?,
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
    'licence': str(subject_info.get("licence", "")).strip() + " (All procedures were approved by the Swiss Federal Veterinary Office)",
    #'ear tag': str(subject_info.get("Ear tag", "")).strip(),
    #"Software and Algorithms": "?",
    }
    ### Experimenter
    experimenter = subject_info["User (user_userName)"]
   
    ### Session_id, identifier, institution, keywords
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology", "patch clamp"] 

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
    #age = f"P{age}D"


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

    ### behavioral type
    behavior_type = str(subject_info.get("Behavior Type", "Unknown").strip())

    """
    if behavior_type == "Detection Task":
        session_description = "ephys " + behavior_type + ": For the detection task, trials with whisker stimulation (Stimulus trials) or those without whisker stimulation (Catch trials) were started without any preceding cues, at random inter-trial intervals ranging from 6 to 12 s. Catch trials were randomly interleaved with Stimulus trials, with 50% probability of all trials. If the mouse licked in the 3-4 s no-lick window preceding the time when the trial was supposed to occur, then the trial was aborted. Catch trials were present from the first day of training. Mice were rewarded only if they licked the water spout within a 1 s response window following the whisker stimulation (Hit)."
        stimulus_notes = "Whisker stimulation was applied to the C2 region to evoke sensory responses."

    elif behavior_type == "Neutral Exposition":
        session_description = "ephys " + behavior_type + ": For the neutral exposure task, mice were trained to collect the reward by licking the water spout with an intertrial interval ranging from 6 to 12 s and after a no-lick period of 3-4 s, similar to the detection task. At random times the same 1 ms whisker stimulus was delivered to the C2 whisker with an inter stimulus interval ranging from 6 to 12 s and a probability of 50%. The whisker stimulus was not correlated to the delivery of the reward, therefore, no association between the stimulus and the delivery of the reward could be made. In this behavioral paradigm, mice were exposed to the whisker stimulus during 7-10 days."
        stimulus_notes = "Whisker stimulation was applied to the C2 region to evoke sensory responses."

    else:
        raise ValueError(f"Unknown behavior type: {behavior_type}")
    """

    # Construct the output YAML path
    config = {
        'session_metadata': {
            'experiment_description' : experiment_description,
            'experimenter': experimenter,
            'identifier': identifier,
            'institution': "Ecole Polytechnique Federale de Lausanne",
            'keywords': keywords,
            'lab' : "Laboratory of Sensory Processing",
            'notes': 'na',
            'pharmacology': 'na',
            'protocol': 'na',
            'related_publications': related_publications,
            'session_description': "ephys " + behavior_type,
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

def files_to_dataframe(mat_file, choice_mouses):

    columns = ['Mouse Name', 'User (user_userName)', 'Cell_ID', 'Ear tag',
        'Start date (dd.mm.yy)', 'End date', 'Sex_bin', 'strain', 'mutations',
        'Birth date', 'licence', 'DG', 'ExpEnd', 'Created on', 'Session',
        'Session Date (yyymmdd)', 'Start Time (hhmmss)', 'Behavior Type',
        'Session Type','sweeps', 'Opto Session', 'Mouse Age (d)', 'Weight of Reference',
        'Weight Session']
    csv_data = pd.DataFrame(columns=columns)
    csv_data.columns = csv_data.columns.str.strip()

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

    with tqdm(total=4) as pbar:

        with h5py.File(mat_file, 'r') as f:

            # Step 1: Mouse metadata loaded
            time.sleep(1)
            tqdm.write("Loading mouse metadata ...")
            pbar.update(1)


            Cell_ID        = read_cell_strings(f, 'Data_Full/Cell_ID')
            mouses_name    = np.array([s.split('_')[0] for s in Cell_ID])
            birth_dates    = read_birth_dates(f, 'Data_Full/Mouse_DateOfBirth')
            Mouse_Sex      = read_cell_strings(f, 'Data_Full/Mouse_Sex')
            sessiontypes   = read_cell_strings(f, 'Data_Full/Session_Type')
            behaviortypes  = read_cell_strings(f, 'Data_Full/Sweep_Type')


            # --- SWEEPS ----
            # Start times
            sweep_start_times = read_sweep_start_times(f, 'Data_Full/Sweep_StartTime')

            # Signaux
            mp_refs   = f['Data_Full/Sweep_MembranePotential'][()]
            cm_refs   = f['Data_Full/Sweep_CurrentMonitor'][()]
            wa_refs   = f['Data_Full/Sweep_WhiskerAngle'][()]

            vm_list   = read_cell_numeric_vectors(f, mp_refs)  # 487 éléments, np.array par sweep
            cur_list  = read_cell_numeric_vectors(f, cm_refs)
            wa_list   = read_cell_numeric_vectors(f, wa_refs)

            # Step 2: Signal data loaded
            time.sleep(1)
            tqdm.write("Loading sweep signal data ...")
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

            # Step 3: Sweep data loaded
            time.sleep(1)
            tqdm.write("Loading sweep behavior data ...")
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
            user = name[:2]
            birth_date = birth_dates[one_cell[0]]
            sex = Mouse_Sex[one_cell[0]]
            session_type = sessiontypes[one_cell[0]]
            behaviortype = " & ".join(np.unique(behaviortypes[one_cell]))
            start_date = sweep_start_times[one_cell[0]].strftime("%d.%m.%y") if not pd.isnull(sweep_start_times[one_cell[0]]) else None
            session_date = sweep_start_times[one_cell[0]].strftime("%Y%m%d") if not pd.isnull(sweep_start_times[one_cell[0]]) else None
            start_time_hhmmss = sweep_start_times[one_cell[0]].strftime("%H%M%S") if not pd.isnull(sweep_start_times[one_cell[0]]) else None
            end_date = sweep_start_times[one_cell[-1]].strftime("%d.%m.%y") if not pd.isnull(sweep_start_times[one_cell[-1]]) else None

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
                ap_abs = safe_abs_times(ap_rel, t0) if t0 is not None else None
                stim_on_abs  = safe_abs_times(stim_on, t0)  if t0 is not None else None
                reward_abs   = safe_abs_times(reward_rel, t0) if t0 is not None else None

                # Trials -> liste de dicts (col1=ts rel, col2=stim?, col3=lick, col4=reward, col5=response)
                trials = []
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
                            "time_abs": ts_abs[k],
                            "lick": bool(lick[k]),
                            "reward": bool(reward_b[k]),
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
                            "time_abs": ts_abs[k],
                            "lick": bool(lick[k]),
                            "reward": bool(reward_b[k]),
                            "response": int(response[k]),
                            'has_stim': bool(has_stim[k]),
                            'onset_time_rel_s': stim_on[ind_stim] ,
                            'onset_time_abs': stim_on_abs[ind_stim] if stim_on_abs is not None else None,
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
                    "Sweep Start Time": t0,
                    "Sweep Stop Time": t0 + timedelta(seconds=float(dur_vm)),
                    "Sweep Type": str(behaviortypes[one_sweep]),
                    'behav': behav,

                    # Signaux
                    "membrane_potential": {
                        "sampling_rate_Hz": int(sr_vm_cm),
                        #"n_samples": int(vm.size),
                        #"duration_s": dur_vm,
                        "data": vm.tolist() if vm.size>0 else []
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
                        "absolute": ap_abs
                    },
                    
                    # Rewards
                    "reward_times": {
                        "relative_s": safe_list(reward_rel),
                        "absolute": reward_abs
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
                "Ear tag": "Unknown", # UNKNOWN
                "Start date (dd.mm.yy)": start_date,
                "End date": end_date,
                "Sex_bin": sex,
                "strain":  "C57BL/6JRj",
                "mutations": "WT",
                "Birth date": birth_date,
                "licence": "VD-1628",
                #"DG": "",
                #"ExpEnd": "",
                "Created on": "Unknown", # UNKNOWN
                "Session": Cell_ID[one_cell[0]] + "_" + session_date,
                "Session Date (yyymmdd)": session_date,
                "Start Time (hhmmss)": start_time_hhmmss,
                "Behavior Type": behaviortype,
                "Session Type": session_type,
                "sweeps": sweeps,
                "Mouse Age (d)": "Unknown", # UNKNOWN
                "Weight of Reference": "Unknown", # UNKNOWN
                "Weight Session": "Unknown" # UNKNOWN
            }

            # Append the new row to the DataFrame
            csv_data = pd.concat([csv_data, pd.DataFrame([new_row])], ignore_index=True)
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