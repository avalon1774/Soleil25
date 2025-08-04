
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Dict, Optional, Any

from scipy.signal import savgol_filter
import numpy as np
import os
import h5py
from dotenv import load_dotenv
import importlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit.models import PolynomialModel, GaussianModel, LinearModel, LorentzianModel, VoigtModel
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import logging
import warnings
__Analysis__ = "Analysis"

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__Analysis__)


def load_scans (sample_number: int,
                scans = dict['str', list],
                roi_id: Optional[int] = " ",
                kind: str = 'Amptek',
                smooth: Optional[int] = 1) -> Optional[pd.Series]:
    """ load specified scans from the h5 files that combine all scans of a sample. return df with all the data interpolated onto the same energy axis if needed"""

    path = f"Sample_{sample_number:04d}.h5"
    if not Path(path).exists():
        logger.error(f"Sample {sample_number:04d} does not exist")
        return None
    dfs = []
    with h5py.File(path, 'r') as f:
        sample_name = f["metadata"]["name"][()].decode("utf-8")

        for scan_number in scans[str(sample_number)]:
            scan_key = f"{scan_number:04d}"
            if scan_key not in f["scans"]:
                logger.error(f"Scan {scan_number:04d} does not exist in sample {sample_number:04d}")
                continue
            scan_grp = f["scans"][scan_key]


            if kind == 'Amptek':
                if "Amptek_XAS" not in scan_grp:
                    logger.error(f"Amptek_XAS not found in scan {scan_number:04d} of sample {sample_number:04d}")
                    continue
                data = scan_grp["Amptek_XAS"]["spectrum"][:]
                df = pd.DataFrame(data, columns=["incident_energy", "intensity"])
                dfs.append(df)


    energy_ax = dfs[0]["incident_energy"]
    summed_intensity = np.zeros_like(energy_ax)

    for df in dfs:
        interpolated_intensity = np.interp(energy_ax, df["incident_energy"], df["intensity"])
        summed_intensity += interpolated_intensity


    if smooth and smooth > 1:
        smoothed = pd.Series(summed_intensity).rolling(window=smooth, center=True).mean().to_numpy()
        if smooth % 2 == 0:
            smooth += 1
        order = min (3, smooth - 1)  # Ensure polyorder is less than window_length
        smoothed = savgol_filter(summed_intensity, window_length=smooth, polyorder=order)#rolling mean for smoothing
    else:
        smoothed = summed_intensity

    df = pd.Series(data=smoothed, index=pd.Series(energy_ax, name="energy [eV]"), name=sample_name + ": Sample " + str(sample_number))
    return df


def nomalize_xas(df: pd.Series) -> pd.Series:
    #first find pre-edge line and subtract from entire spectrum
    pre_edge = df[df.index < 2465]
    line = np.polyfit(pre_edge.index, pre_edge.values, 1)
    baseline = np.polyval(line, df.index)
    pre_edge_normalized = df - baseline

    #then fit a function to last part of spectra, evaluate it at e0 (max derivative) and normalize wit that

    post_edge = df[df.index > 2475]
    line = np.polyfit(post_edge.index, post_edge.values, 1)
    max_der= np.argmax(np.gradient(pre_edge_normalized))
    e0 = df.index[max_der]
    norm_factor = np.polyval(line, e0)
    normalized = pre_edge_normalized / norm_factor



    norm_df = pd.Series(data=normalized, index=pd.Series(df.index, name="energy [eV]"), name = df.name)

    return  norm_df



def plot_xas(samples: List[int], scans: Dict[str, List[int]], kind: str = 'Amptek', cmap_name='viridis'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    axin1 = ax.inset_axes([0.05, 0.55, 0.28, 0.35])
    cmap = cm.get_cmap(cmap_name, len(samples))
    norm = mcolors.Normalize(vmin=0, vmax=len(samples)-1)




    for i, sample in enumerate(samples):
        df = load_scans(sample, scans, kind=kind, smooth=6)
        df = nomalize_xas(df)
        data = df.to_numpy()
        if df is None:
            continue

        color = cmap(norm(i))

        ax.plot(df.index, data, label=df.name, color=color)
        axin1.plot(df.index, np.gradient(data), color=color)

    ax.set_xlabel("Incident Energy (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Amptek XAS Spectra")
    ax.legend(loc = 'lower right')
    ax.grid(True, alpha=0.3)

    axin1.set_xlim(2466, 2476)
    axin1.set_title("Derivatives")



samples = [1,6,7,2,4,5]
scans = {'1': [8],
        '6': [12],
        '7': [6,],
        '2': [6,8],
        '4': [6,7],
         '5': [6],}

plot_xas(samples, scans)

