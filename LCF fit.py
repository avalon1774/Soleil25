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
from scipy.optimize import nnls
__Analysis__ = "Analysis"

from scipy.stats import alpha

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__Analysis__)



def load_scans(sample: int, scans: list, roi_id, kind= 'XES_combined'):
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

                #roi_names = list(scan_grp["ROIs"])
                #roi_names = [roi_id]  #for now look at only one roi. if more are needed, load them seperately
                #if more than one roi, then combine them by adding them up acoss the shared x axis:
                roi_names = roi_id
                if len(roi_id) > 1:
                    data = []

                    for roi in roi_names:
                        excitation_energies = scan_grp["ROIs"][roi]["XES_slices"]["slice_excitation_energies"][:]
                        data_roi = scan_grp["ROIs"][roi]["XES_slices"]["XES_slices"][:]
                        data.append(data_roi)


                    dx = data[0][1,0] - data[0][0,0]
                    x_reference = np.arange(max(arr[:,0][0] for arr in data), min(arr[:,0][-1] for arr in data), step = dx) #coomon part of the x axis for all roias

                    #extrapolate to the common x axis and combine ROI1 + ROI2 for all column
                    combined_data = []
                    for i in range(len(data[0][0])):
                        combined_column = np.zeros_like(x_reference)
                        for roi_data in data:
                            y_interpolated = np.interp(x_reference, roi_data[:,0], roi_data[:,i])
                            combined_column += y_interpolated
                        combined_data.append(combined_column)

                    data = np.column_stack([x_reference] + combined_data[1:])



                    columns = ["emission_energy"] + excitation_energies.tolist()

                    df = pd.DataFrame(data, columns=columns),
                    dfs.append(df)

                    df = pd.DataFrame(data=data[:, 1:], index=pd.Series(data[:,0], name="energy [eV]"),columns=excitation_energies)
                    #df.attrs["name"] = sample_name + ": Sample " + str(sample) + f"({scan_number})"
                    df.attrs["name"] = sample_name


                else:
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

def load_by_energy(samples, plot_sample: bool = False, ):
    """ Load all scans and group them by energy. requires them to have the same enegies"""
    energy_groups = {}
    roi_id = ['ROI2' ]  # can be a list of ROIs to load, if more than one is needed
    for sample in list(samples.keys()):

        if len(samples[sample])>1:
            for scan in samples[sample]:
                df = load_scans(sample, [scan], roi_id = roi_id)
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

        else:

            df = load_scans(sample, samples[sample],roi_id = roi_id)

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

    return energy_groups


samples = {'1': [7],
           '6': [8],
           '7': [4],
           '2': [7],
           '4': [8],
           '5': [4],}

#samples = {'23' : np.arange(9000,9020).tolist(),}

references = {'1': [7],
           '5': [4],}

#references = {'23': [9001,9006]}

energy_groups = load_by_energy(samples, plot_sample=False,)
reference_groups = load_by_energy(references, plot_sample=False)

def remove_background(x,y,energy=None, plot = False):
    "fit a combination of linear background and a gaussian to the data and subtract the linear function from the data"
    mask = ((x > 2445) & (x < 2450)) | ((x > 2470) & (x < 2580)) #roi2
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
        ax.plot(x, y_corrected, color='purple', label='Removed data')
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

def smooth_data(x, y, window_length=6, polyorder=2):
    """Smooth the data using a Savitzky-Golay filter."""
    if len(y) < window_length:
        logger.warning("Data length is shorter than window length. Skipping smoothing.")
        return x, y
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return x, y_smooth


def preprocess(x, y, energy, limits: Optional[tuple] = None):
    #x,y = smooth_data(x, y)
    x,y = remove_background(x,y,energy, plot=False)
    x,y = normalize_intensity(x,y, limits=limits)

    return x,y

    """remove background, normalize to the area under the XES peak and smooth the data"""



def LCF (energy_groups, reference_groups, plot: bool = True):
    """
    Perform Linear Combination Fitting (LCF) on the provided energy groups using the reference groups.
    each group has a key which is energy. then a number of tuples with (x, y, label) where x is the energy, y is the intensity and label is the sample name, all at that exact incident energy.
    """
    results = {}
    limits = [(2460, 2467), (2457, 2471), (2457, 2472), (2457, 2474),(2457, 2472), (2457, 2474)]
    y_limits = [23, 0.65, 0.5, 0.5, 0.5, 0.5,0.5, 0.5]

    fig, axs = plt.subplots(len(energy_groups),2, figsize=(7, 10), sharex='col')

    for i,(energy, data) in enumerate(energy_groups.items()):
        labels = []
        #go over energy, first check if references contain that energy
        if energy not in reference_groups:
            logger.warning(f"No reference group found for energy {energy}. Skipping LCF.")
            continue

        ref_data = reference_groups[energy]
        proc_ref = []

        #fig1,axs = plt.subplots(1, 1, figsize=(16, 10), sharex='col')

        ax1 = axs[i, 0] if len(energy_groups) > 1 else axs[0]
        ax2 = axs[i, 1] if len(energy_groups) > 1 else axs[1]

        for (xr,yr,label) in ref_data:

            x_pr,y_pr = (preprocess(xr,yr,energy, limits = limits[i]) if preprocess else (xr,yr))
            proc_ref.append((np.asarray(x_pr), np.asarray(y_pr), label))
            #axs.plot(x_pr,y_pr, label = label)
            #axs.axvline(x = limits[i][0], color = 'red', ls = '--')
            #axs.axvline(x=limits[i][1], color='red', ls='--')

        #coose a reference x axis and combine all references to that one

        R_labels = []
        R_columns = []
        x_axis = proc_ref[0][0]

        for (xr,yr,label) in proc_ref:
            R_labels.append(label)
            y_inter = np.interp(x_axis, xr,yr)
            R_columns.append(y_inter)

        R = np.column_stack(R_columns)

        #now for each columnin an energy group, perform LC with the above defined references. all should be interpolated to the same energy axis



        for j,(x,y,label) in enumerate(data):
            labels.append(label)
            x_pr, y_pr = (preprocess(x, y, energy, limits=limits[i]) if preprocess else (x, y))
            y_inter = np.interp(x_axis, x_pr, y_pr)
            mask = (x_axis > limits[i][0]) & (x_axis < limits[i][1])


            coef = nnls(R[mask], y_inter[mask])[0]
            s = coef.sum()
            fracs = coef / s if s != 0 else coef
            resid = np.linalg.norm(y_inter[mask] - R[mask] @ coef)  #fit error

            vertical_step = 0.1


            ref_colors = plt.get_cmap('managua', len(R_columns))
            for k,r in enumerate(R_columns):
                ax1.plot(x_axis, r*fracs[k] + (j * vertical_step), color=ref_colors(k), linewidth = 1, alpha = 0.7)
            ax1.plot(x_axis, y_inter + (j * vertical_step), label=label, color='black', linewidth=1)


            for l,lab in enumerate(R_labels):
                ax2.plot(j*0.75,fracs[l], color = ref_colors(l), marker='o')
                ax2.errorbar(j*0.75, fracs[l], yerr= resid, color=ref_colors(l), capsize=3, elinewidth=1, markeredgewidth=1)


        ax1.grid(True, alpha=0.3)
        ax1.axvline(x = limits[i][0], color = 'red', ls=':', alpha = 0.7, linewidth = 1)
        ax1.axvline(x=limits[i][1], color = 'red', ls=':', alpha = 0.7, linewidth = 1)
        ax1.set_ylabel("Intensity")
        ax1.text(0.02, 0.95, f"{energy} eV",
                transform=ax1.transAxes,
                fontsize=8,
                va="top", ha="left")

        ax2.set_ylabel("LCF Coefficients")
        ax2.grid(True, alpha=0.3)

    ax1.set_xlabel("Energy [eV]")
    #ax2.set_xlabel("Sample")
    ax2.set_xlabel("Time [h]")  # not exact, just for visualization
    labels[0] = 'S8'
    #ax2.set_xticks(range(len(labels)))
    #ax2.set_xticklabels(labels)
    #fig.suptitle('1st cycle: ROI 2')
    fig.suptitle('Battery (2.65 mg S: ROI 2)')
    plt.tight_layout()




    return results




LCF(energy_groups, reference_groups, plot=False)
