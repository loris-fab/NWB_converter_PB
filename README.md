
# 🧪 NWB Conversion Pipeline for PB Sessions

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
NWB_converter_PB
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
├── convert_data_to_nwb_PB.py  ← Main conversion script
```


## 💻 Work Environment

Follow the environment setup instructions provided in [LSENS-Lab-Immersion repository](https://github.com/loris-fab/LSENS-Lab-Immersion.git), and include the link to it.

---

## 🧩 How to use

Run the following command in the terminal, replacing `output_folder` with the directory where you want the NWB file to be saved.
`--mouses_name` lets you specify one or more mouse names to process, separated by spaces (e.g., `--mouses_name LB010 LB011`).

```bash
python convert_data_to_nwb_PB.py output_folder --mouses_name LB010 LB011 (...)
```

*Options:*
* `--mouses_name` : Name of the mouse/session to convert (default: all sessions)

If everything runs correctly, you should see an output similar to this:

```bash
**************************************************************************
                  
**************************************************************************
```

## ✍️ Author

Project developed as part of a student project focused on organizing and converting unpublished neuroscience data into the NWB standard.
Main code by **@loris-fab**

For any questions related to the code, please contact: [loris.fabbro@epfl.ch](mailto:loris.fabbro@epfl.ch)

