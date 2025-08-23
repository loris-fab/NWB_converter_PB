from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.file import Subject
from scipy.io import loadmat
from pynwb import NWBFile
import pandas as pd
import yaml
import os

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
    Build a session/subject NWB config from one pandas row and save it as YAML.

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


    ###  Session metadata extraction 
    ### behavioral type
    behavior_type = str(subject_info.get("Behavior Type", "Unknown").strip())

    ### Experiment_description
    Session_Type = subject_info.get("Session Type", "")
    if pd.isna(Session_Type) or str(Session_Type).strip().lower() in ["", "nan"]:
        Session_Type = "Naive"
    if Session_Type == "Trained" or Session_Type == "D1":
        reward_proba_wh = int(1)
        session_description = "ephys " + behavior_type + " mouse: the mouse was rewarded with a drop of water if it licked within 1 s following a whisker stimulus (go trials) but not in the absence of the stimulus (no-go trials). Membrane potential recording was performed in the medial prefrontal cortex using patch-clamp whole-cell recording with glass pipette (4-7 MOhms). WDT session = " + str(subject_info['counter'])
    elif Session_Type == "Naive" :
        reward_proba_wh = int(0)
        session_description = "ephys " + behavior_type + " mouse: the mouse was habituated to sit still while head-restrained. Membrane potential recording was performed in the medial prefrontal cortex using patch-clamp whole-cell recording with glass pipette (4-7 MOhms) while single-whisker stimuli were delivered at random times."

    ### Reference weight
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
    "session_type": "ephys_session " + Session_Type,
    "behavior Task": behavior_type,
    'wh_reward': 1 if reward_proba_wh == 1 else 0,
    'reward_proba': 1 if reward_proba_wh == 1 else 0,
    'wh_stim_amps': '[0:5]',
    "Session Number" : str(subject_info['counter']),
    'licence': str(subject_info.get("licence", "")).strip(),
    'ear tag': str(subject_info.get("Ear tag", "")).strip(),
    'Ambient noise': '80 dB',
    }

    ## Session metadata extraction
    ### Experimenter, Session_id, identifier, institution, keywords, Session start time
    experimenter = "Lila Banterle"
    session_id = subject_info["Session"].strip() 
    identifier = session_id + "_" + str(subject_info["Start Time (hhmmss)"])
    keywords = ["neurophysiology", "behaviour", "mouse", "electrophysiology", "patch-clamp"] 
    session_start_time = str(subject_info["Session Date (yyymmdd)"])+ " " + str(subject_info["Start Time (hhmmss)"])



    ###  Subject metadata extraction 
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
