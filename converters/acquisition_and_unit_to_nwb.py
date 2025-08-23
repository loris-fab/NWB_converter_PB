from pynwb.icephys import CurrentClampSeries, CurrentClampStimulusSeries
from typing import Dict, Any, List
from pynwb import NWBFile
import numpy as np


def add_to_nwb_acquisition_and_units_containers(nwb_file, csv_data_row):
    """
    Add sweeps from a single cell into an NWB file.
    Each sweep is added as an intracellular recording, then grouped into
    a SimultaneousRecording (best practice minimal level for NWB).

    parameters
    ----------
    nwb_file : NWBFile
        The NWB file to which the data will be added.
    csv_data_row : dict
        A dictionary containing the data for a single cell.
    Returns
    -------
    None
    """

    # ------------------------ helpers -------------------------------
    def _clean(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        if isinstance(val, str) and val.strip().lower() in {"", "nan", "none"}:
            return None
        return val

    def _ensure_units_columns(nwb: NWBFile):
        cols = {
            "cell_id": "Cell identifier from CSV",
            "target_area": "Recording area/region",
            "cell_depth_um": "Cell depth in µm",
            "neuron_type": "Neuron type (if known)",
        }
        for name, desc in cols.items():
            if nwb.units is None or name not in nwb.units.colnames:
                nwb.add_unit_column(name, desc)

    def _safe_meta(val, as_float: bool = False):
        if val is None:
            return np.nan if as_float else ""
        return float(val) if as_float else val
    # ---------------------------------------------------------------

    # Devices (created once if missing)
    devices_info = {
        "Amplifier Vm": "Differential extracellular amplifier for membrane potential recording – Multiclamp 700B (Molecular Devices).",
        "Digitizer": "Wavesurfer (https://wavesurfer.janelia.org/) + National Instrument cards (https://www.ni.com).",
        "Patch-clamp Microelectrodes": (
            "Whole-cell pipettes had resistances of 4-7 MΩ and were filled with a solution containing (in mM): 135 potassium gluconate, 4 KCl, 10 HEPES, 10 phosphocreatine, 4 MgATP, 0.3 Na3GTP (adjusted to pH 7.3 with KOH), and 2mg/ml biocytin."
        ),
    }
    for name, desc in devices_info.items():
        if name not in nwb_file.devices:
            nwb_file.create_device(name=name, description=desc)

    # Cell metadata 
    sweeps = csv_data_row.get("sweeps", [])
    if len(sweeps) == 0:
        raise ValueError("No sweeps found in csv_data_row. Cannot proceed with NWB conversion.")

    mp0 = sweeps[0].get("membrane_potential", {})
    cell_id = mp0.get("Cell_ID")
    target_area = mp0.get("target area")
    neuron_type = mp0.get("Type of neurone")
    depth_um = _clean(mp0.get("Cell_Depth (um)"))

    ## Create electrode
    elec_name = f"elec_{cell_id}"
    if elec_name in nwb_file.icephys_electrodes:
        electrode = nwb_file.icephys_electrodes[elec_name]
    else:
        electrode = nwb_file.create_icephys_electrode(
            name=elec_name,
            description="Whole-cell patch. Pipette 4–7 MΩ. DC current-clamp.",
            device=nwb_file.devices["Amplifier Vm"],
            location=target_area,
            filtering="Bevel filter 10 kHz; DC current-clamp",
        )

    # Add sweeps 
    spike_times_all: List[float] = []

    for idx, sweep in enumerate(sweeps):
        start_time = float(sweep.get("Sweep Start Time", 0.0))
        mp = sweep.get("membrane_potential", {}) or {}
        cm = sweep.get("current_monitor", None)
        ap = sweep.get("ap_times", {}) or {}

        sr_mp = float(mp.get("sampling_rate_Hz", np.nan))
        if not np.isfinite(sr_mp):
            raise ValueError(f"Sweep {idx}: missing sampling_rate_Hz for membrane_potential")

        # Response
        mp_data = np.asarray(mp.get("data", []), dtype=np.float32)
        resp = CurrentClampSeries(
            name=f"Membrane_potential_sw{idx:03d}",
            data=mp_data,
            unit="volts",
            rate=sr_mp,
            starting_time=start_time,
            electrode=electrode,
            comments="sampling rate 20 000 Hz, in V.",
            description="Patch-clamp membrane potential recorded from one neuron with a glass electrode of 4-7 MOhms (DC current-clamp recording, bevel filter 10 000 Hz)",
            sweep_number=np.uint(idx)
        )

        # Stimulus
        stim = None
        if cm is not None:
            sr_cm = float(cm.get("sampling_rate_Hz", sr_mp))
            cm_data = np.asarray(cm.get("data", []), dtype=np.float32)
            stim = CurrentClampStimulusSeries(
                name=f"Current_Monitor_sw{idx:03d}",
                data=cm_data,
                unit="amperes",
                rate=sr_cm,
                starting_time=start_time,
                electrode=electrode,
                comments="sampling rate 20 000 Hz, in A.",
                description="Current injected through the recording pipette",
                sweep_number=np.uint(idx)
            )

        # Add intracellular recording
        ir_index = nwb_file.add_intracellular_recording(
            electrode=electrode,
            stimulus=stim,
            response=resp,
        )

        # Group into SimultaneousRecording (best practice minimal level)
        nwb_file.add_icephys_simultaneous_recording([ir_index])

        # Spikes
        abs_spikes = ap.get("absolute", [])
        if abs_spikes:
            spike_times_all.extend([float(t) for t in abs_spikes])

    # Add spikes to nwb.units 
    if spike_times_all:
        _ensure_units_columns(nwb_file)
        nwb_file.add_unit(
            spike_times=np.array(spike_times_all, dtype=float),
            cell_id=cell_id,
            target_area=target_area,
            cell_depth_um=_safe_meta(depth_um, as_float=True),
            neuron_type=neuron_type,
        )
