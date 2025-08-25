
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

Run the following command in the terminal, replacing:

* `input_folder` with the directory containing the `.mat` file "mPFC_Preprocessed",
* `output_folder` with the directory where you want the NWB files to be saved.

`--choice_mouses` lets you specify one or more mouse names to process, separated by spaces (e.g., `--choice_mouses LB010 LB011`).

```bash
python convert_data_to_nwb_PB.py input_folder output_folder --choice_mouses LB010 LB011 (...)
```
### *Options:*

* `--choice_mouses` : One or more mouse/session names to convert (default: all sessions)

for exemple in window:
```bash
python convert_to_nwb_for_LB.py \
"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Sylvain_Crochet/DATA_REPOSITORY/Banterle_mPFC_Vm_2019/mPFC_Preprocessed.mat" \
"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Loris_Fabbro/LB/NWB_files" \
--choice_mouses LB010
```

If everything runs correctly, you should see an output similar to this:

```bash
**************************************************************************
-_-_-_-_-_-_-_-_-_-_-_-_-_-_- NWB conversion _-_-_-_-_-_-_-_-_-_-_-_-_-_-_
📥 Collecting data from .mat file: \\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Sylvain_Crochet\DATA_REPOSITORY\Banterle_mPFC_Vm_2019\mPFC_Preprocessed.mat
 20%|██        | 1/5 [00:01<00:06,  1.63s/it]Loading mouse metadata ...
 40%|████      | 2/5 [00:02<00:03,  1.32s/it]Loading sweep signal data (1/3) ...
 60%|██████    | 3/5 [02:02<01:51, 55.61s/it]Loading sweep signal data (2/3) ...
 80%|████████  | 4/5 [02:04<00:34, 34.39s/it]Loading sweep signal data (3/3) ...
100%|██████████| 5/5 [02:05<00:00, 25.19s/it]
Loading sweep behavior data ...                   
Converting data to NWB format for mouse: 
['LB010_S1_R1', 'LB010_S1_R3']
Processing : 100%|██████████| 2/2 [00:09<00:00,  4.87s/it, LB010]
**************************************************************************
```

## ✍️ Author

Project developed as part of a student project focused on organizing and converting unpublished neuroscience data into the NWB standard.
Main code by **@loris-fab**

For any questions related to the code, please contact: [loris.fabbro@epfl.ch](mailto:loris.fabbro@epfl.ch)

