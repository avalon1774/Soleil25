from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from tkinter.ttk import Label
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

from scipy.stats import alpha

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__Analysis__)


def load_scans(sample: int,
                scans: list,
                roi_id: Optional[int] = "ROI2",
                kind: str = 'XES_combined',
                smooth: Optional[int] = 1) -> Optional[pd.Series]:
    """ load specified scans from the h5 files that combine all scans of a sample. return df with all the data interpolated onto the same energy axis if needed"""



    path = f"Sample_{int(sample):04d}.h5"
    if not Path(path).exists():
        logger.error(f"Sample {int(sample):04d} does not exist")
        return None
    dfs = []
    with h5py.File(path, 'r') as f:
        sample_name = f["metadata"]["name"][()].decode("utf-8")

        for scan_number in scans:
            scan_key = f"{scan_number:04d}"
            if scan_key not in f["scans"]:
                logger.error(f"Scan {scan_number:04d} does not exist in sample {sample_number:04d}")
                continue
            scan_grp = f["scans"][scan_key]


            if kind == 'XES_combined':
                if "ROIs" not in scan_grp:
                    logger.error(f"ROIs not found in scan {scan_number:04d} of sample {sample_number:04d}")
                    continue

                roi_names = list(scan_grp["ROIs"])
                roi_names = [roi_id]

                for roi in roi_names:
                    excitation_energies = scan_grp["ROIs"][roi]["XES_slices"]["slice_excitation_energies"][:]
                    data = scan_grp["ROIs"][roi]["XES_slices"]["XES_slices"][:]
                    columns = ["emission_energy"] + excitation_energies.tolist()

                    df = pd.DataFrame(data, columns=columns),
                    dfs.append(df)

                    df = pd.DataFrame(data=data[:, 1:], index=pd.Series(data[:,0], name="energy [eV]"),columns=excitation_energies)
                    #df.attrs["name"] = sample_name + ": Sample " + str(sample) + f"({scan_number})"
                    df.attrs["name"] = sample_name
    return df


samples = {'1': [7],
           '6': [8],
           '7': [4],
           '2': [7],
           '4': [8],
           '5': [4],}

samples = {'1': [7],            '6': [8],           '7': [4],           '2': [7],           '4': [8],           '5': [4], '8':[2], '9':[2], '10': [6]}
#samples = {'23' : np.arange(9000,9020).tolist(),}


energy_groups = {}

for sample in list(samples.keys()):

#for scan in samples['23']:
#    df = load_scans('23', [scan])

    df = load_scans(sample,samples[sample])

    plot_sample = False
    if plot_sample:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        df.plot(ax=ax)
        ax.set_title(df.attrs["name"])
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

    for col_idx, col_name in enumerate(df.columns):
        x = df.index
        y = df.iloc[:, col_idx]
        label = (df.attrs["name"])

        if col_name not in energy_groups:
            energy_groups[col_name] = []

        energy_groups[col_name].append((x, y, label))





def remove_background(x,y,energy=None, plot = False):
    "fit a combination of linear background and a gaussian to the data and subtract the linear function from the data"
    mask = ((x > 2445) & (x < 2450)) | ((x > 2470) & (x < 2480)) #roi2
    #mask = ((x > 2435) & (x < 2450)) | ((x > 2470) & (x < 2480)) #roi1

    x_fit = x[mask]
    y_fit = y[mask]
    gaussian_model = GaussianModel(prefix='g_')
    linear_model = LinearModel(prefix='l_')
    model = gaussian_model + linear_model
    params = model.make_params()

    params['g_amplitude'].set(value=np.max(y), min=0)
    params['g_center'].set(value=energy)
    params['g_sigma'].set(value=0.5)
    result = model.fit(y_fit, params, x=x_fit)


    elastic_mask = (x > energy - 2) & (x < energy + 2)

    line = result.eval_components(x=x)['l_']
    gauss = result.eval_components(x=x)['g_']

    y_corrected = y - line
    #y_masked = y_corrected.copy()
    #y_masked[elastic_mask] = gauss[elastic_mask]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.plot(x,y, color = 'black', label='Original Data')
        ax.plot(x, line, color='red', label='Linear Background')
        ax.plot(x, gauss, color='blue', label='Gaussian Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)

    return x, y_corrected



def normalize_intensity(x,y,limits=(2455, 2468)):
    low_limit = limits[0]
    upper_limit = limits[1]
    mask = (x > low_limit) & (x < upper_limit)
    x_fit = x[mask]
    y_fit = y[mask]
    area = np.trapezoid(y_fit, x=x_fit)
    max_height = np.max(y_fit)
    return x,y/area


def scale_to_elastic(x, y, energy):
    if energy < max(x):
        mask = (x> energy - 2) & (x < energy + 2)
        x_mask = x[mask]
        y_mask = y[mask]
        area = np.trapezoid(y_mask, x=x_mask)
        return x, y/area

    else:
        return x,y

def smooth_data(x, y, window_length=5, polyorder=2):
    """Smooth the data using a Savitzky-Golay filter."""
    if len(y) < window_length:
        logger.warning("Data length is shorter than window length. Skipping smoothing.")
        return x, y
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return x, y_smooth


fig, axs = plt.subplots(len(energy_groups.items()), 2, figsize=(8, 10), sharex='col')

limits = [(2465,2470),(2457,2467),(2457, 2469),(2457, 2469.3),(2457,2472),(2457, 2475)] #ex situ 6
limits = [(2458,2467),(2454,2469),(2457,2471),(2457, 2472),(2457, 2474),(2457,2472),(2457, 2475)]
y_limits = [23, 0.65, 0.5, 0.5, 0.5,0.5]

#limits = [(2460,2468),(2457,2470.5),(2457, 2475),(2457, 2475),]
#y_limits = [1, 1, 1.2, 1, 2,2]#battery

custom_labels = []

for i, (col_index, group) in enumerate(energy_groups.items()):
    cmap = cm.get_cmap('managua', len(group))
    ax = axs[i,0]
    ax2 = axs[i,1]

    ref_x = None
    ref_y = None
    diff = []
    labels = []


    for ind,(x, y, label) in enumerate(group):
        x, y = smooth_data(x, y, window_length=3)
        x,y = remove_background(x, y, energy = col_index, plot = False)
        norm_lower_limit = limits[i][0]
        norm_upper_limit = limits[i][1]
        x, y = normalize_intensity(x, y, limits=(norm_lower_limit, norm_upper_limit)) #limit should be vared for each energy group
        #x,y = scale_to_elastic(x, y, energy=col_index)
        color = cmap(ind)



        if ind == 0:
            ref_x = x.copy()
            ref_y = y.copy()

        ax.plot(x, y+(0.05*ind), label=label, color = color, linewidth=1)

        if ind == 0:
            diff.append(0.0)
        else:
            y_shifted = np.interp(ref_x,x,y) #if axis by any chance don't match
            mask = (ref_x > norm_lower_limit) & (ref_x < norm_upper_limit)
            area = np.trapezoid(np.abs(y_shifted[mask]-ref_y[mask]), x = ref_x[mask])
            diff.append(area)

        #ax2.plot(ind*0.75, diff[-1], marker='o',color=color)
        ax2.plot(ind, diff[-1], marker='o', color=color)

        labels.append(label)


    ax.axvline(norm_lower_limit, color='red', ls=':', alpha = 0.7, linewidth = 1)
    ax.axvline(norm_upper_limit, color='red', ls=':', alpha = 0.7, linewidth = 1)
    ax.set_yticklabels([])
    ax.set_xlim(2450,2476)
    ax.set_ylim(-0.05,y_limits[i])
    ax.text(0.02, 0.95, f"{col_index} eV",
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="left")


    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)

    ax2. set_ylabel("IAD")
    ax2.grid(True,alpha=0.3)


#labels[0] = 'S8'
ax.set_xlabel("Energy [eV]")
ax2.set_xlabel("Sample")
ax2.set_xlabel("Sample")
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels)
fig.suptitle('1st cycle: ROI 2')
plt.tight_layout()
#.savefig('(extrapolate 10) Battery 3: ROI 2', dpi=300)
#ax.legend()

plt.show()




#load references

