
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
                df = pd.DataFrame(data, columns=["incident_energy", "intensity","intensity_2", "I01", "I02"]),
                dfs.append(df)


    energy_ax = dfs[0][0]["incident_energy"]
    summed_intensity = np.zeros_like(energy_ax)

    for df in dfs:
        interpolated_intensity = np.interp(energy_ax, df[0]["incident_energy"], df[0]["intensity"])
        summed_intensity += interpolated_intensity


    if smooth and smooth > 1:
        smoothed = pd.Series(summed_intensity).rolling(window=smooth, center=True).mean().to_numpy()
        if smooth % 2 == 0:
            smooth += 1
        order = min (3, smooth - 1)  # Ensure polyorder is less than window_length
        smoothed = savgol_filter(summed_intensity, window_length=smooth, polyorder=order)#rolling mean for smoothing
    else:
        smoothed = summed_intensity

    df = pd.Series(data=smoothed, index=pd.Series(energy_ax, name="energy [eV]"), name=sample_name + ": Sample " + str(sample_number) + f"({scan_number})")
    return df


def nomalize_xas(df: pd.Series, flat = False, plot = False) -> pd.Series:
    #first find pre-edge line and subtract from entire spectrum
    if plot:
        fig,ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.plot(df.index,df.values, color = 'black', label = "data")


    pre_edge = df[df.index < 2465]
    pre_line = np.polyfit(pre_edge.index, pre_edge.values, 1)
    pre_baseline = np.polyval(pre_line, df.index)
    pre_edge_normalized = df - pre_baseline
    if plot:
        ax.plot(df.index,pre_baseline, color = 'red', ls = '--', label = "pre-edge")

    #then fit a function to last part of spectra, evaluate it at e0 (max derivative) and normalize wit that

    post_edge = df[df.index > 2485]
    post_line = np.polyfit(post_edge.index, post_edge.values, 1)
    post_baseline = np.polyval(post_line, df.index)
    max_der= np.argmax(np.gradient(pre_edge_normalized))
    e0 = df.index[max_der]
    edge_step = np.polyval(post_line, e0) - np.polyval(pre_line, e0)
    normalized = pre_edge_normalized/edge_step

    norm_df = pd.Series(data=normalized, index=pd.Series(df.index, name="energy [eV]"), name=df.name)
    if plot:
        ax.plot(df.index,post_baseline, color = 'red', ls = '--', label = "post-edge")
        ax.axvline(e0, color='green', ls=':', label='E0')

    #flatten by fitting a quadratic function to the end and subtract that from the espectrum above e0

    if flat:
        post_edge = norm_df[norm_df.index > 2480] #maybe set both limits, idk why it doesn't normalize properly
        quadratic = np.polyfit(post_edge.index, post_edge.values, 1)
        baseline = np.polyval(quadratic, norm_df.index)


        flat = (norm_df - baseline) + baseline[max_der]
        flat.iloc[:max_der] = normalized.iloc[:max_der]


        if plot:
            ax.plot(df.index, (baseline*edge_step) , color='blue', ls='--', label="quad")
            ax.plot(post_edge.index,post_edge.values, linewidth = 10, alpha = 0.3)
        return  flat

    else:
        return norm_df


def plot_xas(samples: List[int], scans: Dict[str, List[int]], kind: str = 'Amptek', cmap_name='viridis'):
    cmap_name = 'managua'
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    axin1 = ax.inset_axes([0.05, 0.55, 0.28, 0.35])

    #cmap = cm.get_cmap(cmap_name, len(samples))
    #norm = mcolors.Normalize(vmin=0, vmax=len(samples) - 1)

    for i, sample in enumerate(samples):
        temp_dict = {}
        cmap = cm.get_cmap(cmap_name, len(scans[str(sample)]))
        norm = mcolors.Normalize(vmin=0, vmax=len(scans[str(sample)]) - 1)

        for j, scan in enumerate(scans[str(sample)]):
            temp_dict[str(sample)] = [scan]
            df = load_scans(sample, temp_dict, kind=kind, smooth=10)
            df = nomalize_xas(df, plot=False, flat = True)
            data = df.to_numpy()
            if df is None:
                continue

            color = cmap(norm(j))


            ax.plot(df.index, data/data[-1] + 0.3*j, label=df.name, color=color)
            axin1.plot(df.index, np.gradient(data), color=color)

    ax.set_xlabel("Incident Energy (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Amptek XAS Spectra")
    #ax.legend(loc = 'lower right')
    ax.grid(True, alpha=0.3)

    battery_lines = [2471.1, 2473.4,2482.5]
    for line in battery_lines:
        ax.axvline(line, color='red', ls=':', alpha = 0.7, linewidth = 1)

    axin1.set_xlim(2466, 2476)
    axin1.set_title("Derivatives")


def plot_xas_2D(samples: List[int], scans: Dict[str, List[int]], kind: str = 'Amptek', cmap_name='viridis'):
    cmap_name = 'managua'
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    #axin1 = ax.inset_axes([0.05, 0.55, 0.28, 0.35])

    all_xas = []


    for i, sample in enumerate(samples):
        temp_dict = {}
        cmap = cm.get_cmap(cmap_name, len(scans[str(sample)]))
        norm = mcolors.Normalize(vmin=0, vmax=len(scans[str(sample)]) - 1)

        time = []
        for j, scan in enumerate(scans[str(sample)]):
            temp_dict[str(sample)] = [scan]
            df = load_scans(sample, temp_dict, kind=kind, smooth=10)
            df = nomalize_xas(df, plot=False, flat = False)
            time.append(j*0.75)
            data = df.to_numpy()
            if df is None:
                continue

            color = cmap(norm(j))

            all_xas.append(np.gradient(data))
            #all_xas.append(data)

            #ax.plot(df.index, data, label=df.name, color=color)
            #axin1.plot(df.index, np.gradient(data), color=color)
        im = ax.pcolormesh(df.index, time, all_xas, cmap = 'managua',shading='nearest')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Incident Energy (eV)")
        ax.set_ylabel("Time (h)") #not exact
        #ax.set_title("Amptek XAS Spectra")
        #ax.legend(loc = 'lower right')
        ax.grid(True, alpha=0.3)

        ax.set_xlim(2465, 2476)
        ax.set_title("Derivatives")






samples = [23]
scans = {'23': np.arange(14,136,5)}








plot_xas_2D(samples, scans, )
