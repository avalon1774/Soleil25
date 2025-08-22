
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

            if kind == 'Pilatus':
                roi = 'ROI2'
                data = scan_grp["ROIs"][roi]["XAS_clean"][:]
                df = pd.DataFrame(data, columns=["incident_energy", "intensity"]),
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


def plot_xas(samples: List[int], scans: Dict[str, List[int]], kind: str = 'Amptek', gauss = True, cmap_name='viridis'):
    cmap_name = 'managua'


    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    if gauss:
        fig2, axs2 = plt.subplots(3, 1, figsize=(7, 7), sharex=True)


    #axin1 = ax.inset_axes([0.05, 0.55, 0.28, 0.35])

    cmap = cm.get_cmap(cmap_name, len(samples))
    norm = mcolors.Normalize(vmin=0, vmax=len(samples) - 1)

    for i, sample in enumerate(samples):
        temp_dict = {}
        #cmap = cm.get_cmap(cmap_name, len(scans[str(sample)]))
        #norm = mcolors.Normalize(vmin=0, vmax=len(scans[str(sample)]) - 1)

        for j, scan in enumerate(scans[str(sample)]):
            temp_dict[str(sample)] = [scan]
            df = load_scans(sample, temp_dict, kind=kind, smooth=15)
            df = nomalize_xas(df, plot=False, flat = True)
            data = df.to_numpy()
            if df is None:
                continue

            color = cmap(norm(i))

            col = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
            #ax.plot(df.index, data/data[-1] + 0.3*i, label=df.name, color=color)
            ax.plot(df.index, np.gradient(data) + 0.1 * i, label=df.name, color=color)
            #axin1.plot(df.index, np.gradient(data), color=color)

            #find the position of first peak in the derivative:
            max_der = np.argmax(np.gradient(data))
            e0 = df.index[max_der]



            if gauss:
                #fit N gaussians to the first part of the derivative and plot them iin ax_2

                gauss_params, amp = fit_N_gaus(3, df.index, np.gradient(data), plot=True)
                markers = ['o','d','s']
                energies = [2469.2, 2471.0, 2472.0]
                for n in range(3):

                    ax2 = axs2[n]
                    ax2.scatter(i, amp[f'g{n}_amplitude'], label=f'Gaussian {i+1}',color=color, marker=markers[n])



                    #ax2.text(gauss_params[f'g{j}_center'], f'{gauss_params[f"g{j}_center"]:.2f}', fontsize=8, color=color)
    if gauss:
        for n in range(3):
            ax2 = axs2[n]
            ax2.set_ylabel("Intensity of peak (a.u)")
            ax2.text(0.02, 0.95, f"{energies[n]} eV",
                     transform=ax2.transAxes,
                     fontsize=8,
                     va="top", ha="left")
            ax2.grid(True, alpha=0.3)


    ax.set_xlabel("Incident Energy (eV)")
    ax.set_ylabel("Intensity (counts)")

    ax.set_title("Derivative of Pilatus XAS Spectra")
    #ax.legend(loc = 'lower right')
    ax.grid(True, alpha=0.3)
    ax2.set_xlabel("Time [h]")



    battery_lines = [2471.1, 2473.4,2482.5]
    for line in battery_lines:
        ax.axvline(line, color='red', ls=':', alpha = 0.7, linewidth = 1)

    #axin1.set_xlim(2466, 2476)
    ax.set_xlim(2460, 2490)
    #axin1.set_title("Derivatives")


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
            df = load_scans(sample, temp_dict, kind=kind, smooth=20)
            df = nomalize_xas(df, plot=False, flat = False)
            time.append(j*0.75)
            data = df.to_numpy()
            if df is None:
                continue

            color = cmap(norm(i))

            all_xas.append(np.gradient(data))
            #all_xas.append(data)

            #ax.plot(df.index, data, label=df.name, color=color)
            #axin1.plot(df.index, np.gradient(data), color=color)
        im = ax.pcolormesh(df.index, time, all_xas, cmap = 'managua',shading='nearest')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Incident Energy (eV)")
        ax.set_ylabel("Sample") #not exact
        #ax.set_title("Amptek XAS Spectra")
        #ax.legend(loc = 'lower right')
        ax.grid(True, alpha=0.3)

        ax.set_xlim(2465, 2476)
        ax.set_title("Derivatives")


def fit_N_gaus(N, x, y, plot = True):
    "fit N gaussian fuction to the derivative of data"
    models = []
    for i in range(N):
        g = GaussianModel(prefix=f'g{i}_')
        models.append(g)
    model = models[0]
    for m in models[1:]:
        model += m
    params = model.make_params()
    gauss_start_values = [2469.0, 2470.3, 2472.0]

    for i in range(N):
        params[f'g{i}_center'].set(value=gauss_start_values[i], min=gauss_start_values[i]-0.2, max=gauss_start_values[i]+0.2)
        params[f'g{i}_sigma'].set(value=0.1, min=0.0001)
        params[f'g{i}_amplitude'].set(value=1, min=0)
    result = model.fit(y, params, x=x)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.plot(x, y, label='Data', color='black')
        ax.plot(x, result.best_fit, label='Fit', color='red')
        for i in range(N):
            ax.plot(x, result.eval_components(x=x)[f'g{i}_'], label=f'Gaussian {i+1}')
        ax.legend()
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

    #return only position and amplitudes of peaks
    return {f'g{i}_center': result.params[f'g{i}_center'].value for i in range(N)}, {f'g{i}_amplitude': result.params[f'g{i}_amplitude'].value for i in range(N)}



#samples = [23]
#scans = {'23': np.arange(14,136,5)}


samples = [21,1,6,7,2,4,5,8,9,10]
#samples = [1,6,7,2,4,5,]
#scans = {'1': [8],         '6': [12],         '7': [6],         '2': [6],         '4': [7],         '5': [6],}

scans = {'21':[65], '1': [7],            '6': [8],           '7': [4],           '2': [7],           '4': [8],           '5': [4], '8':[2], '9':[2], '10': [6]}




plot_xas(samples, scans, kind='Pilatus')
