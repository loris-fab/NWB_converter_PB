from datetime import datetime
import pandas as pd
import numpy as np
import os
import yaml
from dateutil.tz import tzlocal
from pynwb import NWBFile
from pynwb.file import Subject
from scipy.io import loadmat
import re
import h5py

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


def files_to_config(csv_data_row,output_folder="data"):
    related_publications = 'Le Merre P, Esmaeili V, Charrière E, Galan K, Salin PA, Petersen CCH, Crochet S. Reward-Based Learning Drives Rapid Sensory Signals in Medial Prefrontal Cortex and Dorsal Hippocampus Necessary for Goal-Directed Behavior. Neuron. 2018 Jan 3;97(1):83-91.e5. doi: 10.1016/j.neuron.2017.11.031. Epub 2017 Dec 14. PMID: 29249287; PMCID: PMC5766832.'
    

    mouse = csv_data_row['Mouse Name']
    date = csv_data_row['Session Date (yyymmdd)']
    session_name = csv_data_row['Session']

    subject_info = csv_data_row

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
    "Software and Algorithms": "Labview, Klusta, MATLAB R2015b",
    }
    ### Experimenter
    experimenter = "Pierre Le Merre"
   
    ### Session_id, identifier, institution, keywords
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology"] 

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
    if behavior_type == "Detection Task":
        session_description = "ephys " + behavior_type + ": For the detection task, trials with whisker stimulation (Stimulus trials) or those without whisker stimulation (Catch trials) were started without any preceding cues, at random inter-trial intervals ranging from 6 to 12 s. Catch trials were randomly interleaved with Stimulus trials, with 50% probability of all trials. If the mouse licked in the 3-4 s no-lick window preceding the time when the trial was supposed to occur, then the trial was aborted. Catch trials were present from the first day of training. Mice were rewarded only if they licked the water spout within a 1 s response window following the whisker stimulation (Hit)."
        stimulus_notes = "Whisker stimulation was applied to the C2 region to evoke sensory responses."

    elif behavior_type == "Neutral Exposition":
        session_description = "ephys " + behavior_type + ": For the neutral exposure task, mice were trained to collect the reward by licking the water spout with an intertrial interval ranging from 6 to 12 s and after a no-lick period of 3-4 s, similar to the detection task. At random times the same 1 ms whisker stimulus was delivered to the C2 whisker with an inter stimulus interval ranging from 6 to 12 s and a probability of 50%. The whisker stimulus was not correlated to the delivery of the reward, therefore, no association between the stimulus and the delivery of the reward could be made. In this behavioral paradigm, mice were exposed to the whisker stimulus during 7-10 days."
        stimulus_notes = "Whisker stimulation was applied to the C2 region to evoke sensory responses."

    else:
        raise ValueError(f"Unknown behavior type: {behavior_type}")
    
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
            'session_description': session_description,
            'session_id': session_id,
            'session_start_time': session_start_time,
            'slices': "na", 
            'source_script': 'na',
            'source_script_file_name': 'na',
            'stimulus_notes': stimulus_notes,
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
            'strain': subject_info.get("strain", "").strip(),
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

    
    csv_data = pd.DataFrame()
    csv_data.columns = ['Mouse Name', 'User (user_userName)', 'Cell_ID', 'Ear tag',
       'Start date (dd.mm.yy)', 'End date', 'Sex_bin', 'strain', 'mutations',
       'Birth date', 'licence', 'DG', 'ExpEnd', 'Created on', 'Session',
       'Session Date (yyymmdd)', 'Start Time (hhmmss)', 'Behavior Type',
       'Session Type', 'Opto Session', 'Mouse Age (d)', 'Weight of Reference',
       'Weight Session']
    csv_data.columns.str.strip(inplace=True)

    with h5py.File(mat_file, 'r') as f:
        print(list(f.keys())) 
        General_data = f['Data_Full']
        print("Contenu de 'Data_Full' :", list(General_data.keys()))


    mouse_name = General_data["LFP_Data"][0][0][0][0][0][0]

    
    # information general
    _, id_session_indices = np.unique(General_data["LFP_Data"][0][0][4], return_index=True)
    print(f"Number of sessions: {len(id_session_indices)}")
    print(f"Number of files in PLALL: {len(sorted(os.listdir(PLALL)))}")
    mouse_name = str(General_data["LFP_Data"][0][0][0][0][0][0])
    strain = str(General_data["LFP_Data"][0][0][1][0][0][0])
    sex = str(General_data["LFP_Data"][0][0][2][0][0][0])
    raw_birth = General_data["LFP_Data"][0][0][3][0][0][0][0]
    birth_date = "Unknown" if raw_birth is None or str(raw_birth).strip() in ["", "[]"] else str(raw_birth)

    # informations sessions
    # Iterate over each file in PLALL 

    def extract_number(file_name):
        if file_name.endswith('.mat'):
            match = re.search(r'D(\d+)', file_name)
            if match:
                return int(match.group(1))
            raise ValueError(f"No number found in a .mat file: {file_name}")
        return -1

    i = -1

    for file_name in sorted(os.listdir(PLALL), key=extract_number):
        file_path = os.path.join(PLALL, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.mat'):
            i += 1
            pli = loadmat(file_path)
            # Start date
            ## dd
            dd = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][2][0][0])
            if len(dd) == 1:
                dd = "0" + dd
            ## mm
            mm = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][1][0][0])
            if len(mm) == 1:
                mm = "0" + mm
            ## yy
            yy = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][0][0][0])

            start_date = dd + "." + mm + "." + yy
            start_date_2 = yy + mm + dd
            End_date = start_date

            session = mouse_name + "_" + start_date_2

            # Start time (hhmmss)
            ## hh
            hh = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][3][0][0])
            if len(hh) == 1:
                hh = "0" + hh
            ## mm
            mm = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][4][0][0])
            if len(mm) == 1:
                mm = "0" + mm
            ## ss
            ss = str(General_data["LFP_Data"][0][0][5][id_session_indices[i]][5][0][0])
            if len(str(int(float(ss)))) == 1:
                ss = "0" + str(int(float(ss)))
            else:
                ss = str(int(float(ss)))
                
            start_time = hh + mm +  ss

            # Behavior type
            behaviortype = str(General_data["LFP_Data"][0][0][6][id_session_indices[i]][0][0])
            if behaviortype == "DT":
                behaviortype = "Detection Task"
            elif behaviortype == "X":
                behaviortype = "Neutral Exposition"
            else:
                raise ValueError(f"Unknown behavior type: {behaviortype}")

            # Stim_times
            stim_onset= np.asarray(pli["Stim_times"][0][0][1][0])/1000
            # Catch_times
            catch_onset = np.asarray(pli["Catch_times"][0][0][0][0])/1000
            # trial_onset
            all_onsets = np.concatenate([stim_onset, catch_onset])
            all_onsets_sorted = np.sort(all_onsets)


            #EMG
            if "EMG" in pli.keys():
                EMG = np.asarray(pli["EMG"][0][0][1]).flatten()
            else:
                EMG = np.nan
                
            # PtA
            if "PtA" in pli.keys():
                PtA= np.asarray(pli["PtA"][0][0][1]).flatten()
            else:
                PtA = np.nan

            # dCA1
            if "dCA1" in pli.keys():
                dCA1= np.asarray(pli["dCA1"][0][0][1]).flatten()
            else:
                dCA1 = np.nan

            # mPFC
            if "mPFC" in pli.keys():
                mPFC= np.asarray(pli["mPFC"][0][0][1]).flatten()
            else:
                mPFC = np.nan

            # wM1
            if "wM1" in pli.keys():
                wM1= np.asarray(pli["wM1"][0][0][1]).flatten()
            else:
                wM1 = np.nan

            # wS1
            if "wS1" in pli.keys():
                wS1= np.asarray(pli["wS1"][0][0][1]).flatten()
            else:
                wS1 = np.nan

            # wS2
            if "wS2" in pli.keys():
                wS2= np.asarray(pli["wS2"][0][0][1]).flatten()
            else:
                wS2 = np.nan


            if "antM1" in pli.keys():
                antM1 = np.asarray(pli["antM1"][0][0][1]).flatten()
            else:
                antM1 = np.nan

            if "EEG" in pli.keys():
                EEG = np.asarray(pli["EEG"][0][0][1]).flatten()
            else:
                EEG = np.nan

            # Create a new row for the session
            new_row = {
                "Mouse Name": mouse_name,
                "User (user_userName)": "PL",
                "Ear tag": "Unknown",
                "Start date (dd.mm.yy)": start_date,
                "End date": End_date,
                "Sex_bin": sex,
                "strain": strain,
                "mutations": "",
                "Birth date": birth_date,
                "licence": "1628",
                "DG": "",
                "ExpEnd": "",
                "Created on": "Unknown",
                "Session": session,
                "Session Date (yyymmdd)": start_date_2,
                "Start Time (hhmmss)": start_time,
                "Behavior Type": behaviortype,
                "Session Type": "Whisker Rewarded",
                "Mouse Age (d)": "Unknown",
                "Weight of Reference": "Unknown",
                "Weight Session": "Unknown",
                "Trial_onset" : ';'.join(map(str, all_onsets_sorted)),
                "stim_onset": ';'.join(map(str, stim_onset)),
                "catch_onset": ';'.join(map(str, catch_onset)),
                "Responses_times": ';'.join(map(str, np.asarray(pli["Valve_times"][0][0][1][0])/1000)),
                "EMG": ';'.join(map(str, EMG)) if not (isinstance(EMG, float) and np.isnan(EMG)) else np.nan,
                "PtA": ';'.join(map(str, PtA)) if not (isinstance(PtA, float) and np.isnan(PtA)) else np.nan,
                "dCA1": ';'.join(map(str, dCA1)) if not (isinstance(dCA1, float) and np.isnan(dCA1)) else np.nan,
                "mPFC": ';'.join(map(str, mPFC)) if not (isinstance(mPFC, float) and np.isnan(mPFC)) else np.nan,
                "wM1": ';'.join(map(str, wM1)) if not (isinstance(wM1, float) and np.isnan(wM1)) else np.nan,
                "wS1": ';'.join(map(str, wS1)) if not (isinstance(wS1, float) and np.isnan(wS1)) else np.nan,
                "wS2": ';'.join(map(str, wS2)) if not (isinstance(wS2, float) and np.isnan(wS2)) else np.nan,
                "antM1": ';'.join(map(str, antM1)) if not (isinstance(antM1, float) and np.isnan(antM1)) else np.nan,
                "EEG": ';'.join(map(str, EEG)) if not (isinstance(EEG, float) and np.isnan(EEG)) else np.nan
            }

            # Append the new row to the DataFrame
            csv_data = pd.concat([csv_data, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Processing session file: {file_name}")
    # Save the updated DataFrame to the CSV file
    csv_data.to_csv(csv_file, sep=';', index=False)
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