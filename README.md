
# 🧪 NWB Conversion Pipeline for LB Sessions

This project provides a conversion pipeline for **single-neuron membrane potential recordings** from unpublished experiments into the standard **Neurodata Without Borders (NWB)** format.

The dataset includes:

* **Whisker rewarded (WR+)** sessions, where mice received a drop of water if they licked within 1 s after a whisker stimulus (go trials), but not in no-go trials.
* **No-Task** sessions, where head-fixed mice received random whisker stimuli without any behavioral task.

Recordings were obtained in the medial prefrontal cortex using **whole-cell patch-clamp** with glass pipettes (4–7 MΩ), measuring membrane potential from a single neuron. Trials are organized as sweeps, with both **recorded Vm** and **injected currents** stored.


## ⚙️ Features
* Reads `.mat` files containing raw data and `.csv` files containing subject metadata
* Converts to NWB structure including:
  * General metadata (subject, session…)
  * Time intervals (e.g., trials/sweeps…)
  * Behavioral data (licks, rewards, whisker stimuli…)
  * Intracellular ephys data (membrane potential, injected current)
* Validates the NWB file after conversion


## 📁 Project Structure

```
NWB_converter_LB
│
├── converters/
│   ├── acquisition_and_unit_to_nwb.py
│   ├── behavior_to_nwb.py
│   ├── general_to_nwb.py
│   ├── intervals_to_nwb.py
│   ├── Initiation_nwb.py
│   └── nwb_saving.py
├── Subject_Session_Selection.csv
├── README.md
├── convert_data_to_nwb_LB.py  ← Main conversion script
```


## 💻 Work Environment

Follow the environment setup instructions provided in [LSENS-Lab-Immersion repository](https://github.com/loris-fab/LSENS-Lab-Immersion.git), and include the link to it.



## 🧩 How to use

Please find below the key information

* `input_folder` → directory containing the `.mat` file "mPFC_Preprocessed",
* `output_folder` → directory where you want the NWB files to be saved.
* `choice_mouses` → lets you specify one or more mouse names to process.

### Commande in the terminal
Run the following command in the terminal, replacing the arguments :

```bash
python convert_data_to_nwb_PB.py input_folder output_folder --choice_mouses LB010 LB011 (...)
```
*Options:*

* `--choice_mouses` : One or more mouse/session names to convert (default: all sessions), separated by spaces `(e.g., --choice_mouses LB010 LB011).`

*for exemple in window:*
```bash
python convert_to_nwb_for_LB.py \
"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Sylvain_Crochet/DATA_REPOSITORY/Banterle_mPFC_Vm_2019/mPFC_Preprocessed.mat" \
"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/LB/NWB_files" \
--choice_mouses LB010
```


### Run inside a Jupyter Notebook

You can also call the conversion function directly inside a Jupyter Notebook without using the command line.
Simply import `convert_data_to_nwb_LB` from your script and call it with the correct arguments:

*for exemple in window:*
```python
from convert_to_nwb_for_LB import convert_data_to_nwb_LB

convert_data_to_nwb_LB(
    input_folder="//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Sylvain_Crochet/DATA_REPOSITORY/Banterle_mPFC_Vm_2019/mPFC_Preprocessed.mat",
    output_folder="//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/LB/NWB_files",
    choice_mouses=["LB010"]  # you can pass one or multiple mouse names
)
```

*Options:*
* `choice_mouses` : One or more mouse/session names to convert (default: all sessions). Use a Python list for multiple mice (e.g. `["LB010", "LB011"]`).


### Outcome
If everything runs correctly, you should see an output similar to this:

```bash
**************************************************************************
-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_
📥 Collecting data from .mat file:  /Users/lorisfabbro/Documents/MATLAB/PB/mPFC_Preprocessed.mat
Loading finished.: 100%|██████████| 5/5 [01:43<00:00, 20.73s/it]                  
Converting data to NWB format for mouse:
['LB010_S1_R1', 'LB010_S1_R3']
Conversion to NWB is finished: 100%|██████████| 2/2 [00:03<00:00,  1.77s/it]**************************************************************************
```

## ✍️ Author

Project developed as part of a student project focused on organizing and converting unpublished neuroscience data into the NWB standard.
Main code by **@loris-fab**

For any questions related to the code, please contact: [loris.fabbro@epfl.ch](mailto:loris.fabbro@epfl.ch)

