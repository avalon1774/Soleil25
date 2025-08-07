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


def load_scans(sample: int,
                scans: list,
                roi_id: Optional[int] = " ",
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
                for roi in roi_names:
                    excitation_energies = scan_grp["ROIs"][roi]["XES_slices"]["slice_excitation_energies"][:]
                    data = scan_grp["ROIs"][roi]["XES_slices"]["XES_slices"][:]
                    columns = ["emission_energy"] + excitation_energies.tolist()

                    df = pd.DataFrame(data, columns=columns),
                    dfs.append(df)

                    df = pd.DataFrame(data=data[:, 1:], index=pd.Series(data[:,0], name="energy [eV]"),columns=excitation_energies)
                    df.attrs["name"] = sample_name + ": Sample " + str(sample) + f"({scan_number})"
    return df


samples = {'1': [7],
           '6': [8],
           '7': [4],
           '2': [7],
           '4': [8],
           '5': [4], }


energy_groups = {}

for sample in list(samples.keys()):

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





def remove_background(x,y,energy=None):
    "fit a combination of linear background and a gaussian to the data and subtract the linear function from the data"
    mask = ((x > 2445) & (x < 2455)) | ((x > 2475) & (x < 2480))
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



    line = result.eval_components(x=x)['l_']
    gauss = result.eval_components(x=x)['g_']

    y_corrected = y - line
    return x, y_corrected



def normalize_intensity(x,y,):
    mask = (x > 2455) & (x < 2468)
    x_fit = x[mask]
    y_fit = y[mask]
    area = np.trapezoid(y_fit, x=x_fit)
    return x,y /area




for col_index, group in energy_groups.items():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    for x, y, label in group:
        x,y = remove_background(x, y)
        x, y = normalize_intensity(x, y)
        ax.plot(x, y, label=label)
    ax.set_title(f"Energy Group {col_index}")
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

