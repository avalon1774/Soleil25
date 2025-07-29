from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import os
import h5py
from dotenv import load_dotenv
import importlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit.models import PolynomialModel, GaussianModel, LinearModel, LorentzianModel, VoigtModel
import pandas as pd
import warnings

from cache_utils import get_scan_dir, get_sample_dir

warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0.*")   #idk where this error comes from but now I don't see it anymore so eh...

import cache_utils
importlib.reload(cache_utils)

get_calibration_path = cache_utils.get_calibration_path
save_pickle = cache_utils.save_pickle
clean_sample_cache = cache_utils.clean_sample_cache
load_pickle = cache_utils.load_pickle
get_spectrum_path = cache_utils.get_spectrum_path
get_plot_path = cache_utils.get_plot_path

import logging


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sample")

     # If you have logs from galaxies module

def read_nxs_file(filename):
    """
    Reads a .nxs file and returns the requested data as a dictionary. Silently ignores missing data.
    Parameters:
    filename (str): The path to the .nxs file relative to the path given in the DATA_PATH variable.

    Returns:
    dict: A dictionary containing the rquested data from the .nxs file. At the moment, the following keys may be available:
        - 'energies': The energies of the incident photons.
        - 'images': The images taken during the scan.
        - 'position': The position of the sample during the scan.
        - 'DIODE': The DIODE signal during the scan.
        - 'I01': The I01 signal during the scan.
        - 'SDD': The SDD signal during the scan.
        - 'Amptek': The Amptek signal during the scan.
        - 'exposure_time': The exposure time for each image.
        - 'start_time': The start time of the scan in wall clock.
        - 'scan_command': The command used to initiate the scan.
    """
    load_dotenv('/home/ava/Documents/2025/GALAXIES25/Analysis/soleil25a/.env')  # Adjust this path to where your .env file is
    data_path = os.environ.get('DATA_PATH')

    fullpath = os.path.join(data_path, filename)
    logger.info("reading file %s", filename)

    to_read_arrays = {
        'energies': '/root_spyc_config1d_RIXS_0001/GALAXIES/pilatus/Energy_Real',
        'images': '/root_spyc_config1d_RIXS_0001/scan_data/image_image',
        'position': '/root_spyc_config1d_RIXS_0001/GALAXIES/scan_record/MotorTrj1',
        'DIODE': '/root_spyc_config1d_RIXS_0001/GALAXIES/scan_record/RIXS_DIODE',
        'I01': '/root_spyc_config1d_RIXS_0001/GALAXIES/scan_record/QBPM_C08_sum',
        'SDD': '/root_spyc_config1d_RIXS_0001/scan_data/xspchannel00',
        'Amptek': '/root_spyc_config1d_RIXS_0001/scan_data/xspchannel01',
        'exposure_time': '/root_spyc_config1d_RIXS_0001/GALAXIES/i07-c-cx2-dt-pilatus.2/exposure_time',
        'sample_zs': '/root_spyc_config1d_RIXS_0001/GALAXIES/i07-c-cx2-ex-sample_zs/sample_zs',
    }

    to_read_scalars = {
        'start_time': '/root_spyc_config1d_RIXS_0001/start_time',
        'end_time': '/root_spyc_config1d_RIXS_0001/end_time',
        'scan_command': '/root_spyc_config1d_RIXS_0001/GALAXIES/scan_record/scan',
    }

    with h5py.File(fullpath, 'r') as f:
        out = {}
        for key, path in to_read_arrays.items():
            if path in f:
                out[key] = f[path][:]
        for key, path in to_read_scalars.items():
            if path in f:
                out[key] = f[path][()]
                if isinstance(out[key], bytes):
                    out[key] = out[key].decode('utf-8', errors='ignore')

    return out


def save_xas_from_RIXS(scan: 'RIXSMap', roi_id, energy, rw_xas, cleaned_xas):
    df = pd.DataFrame({
        'incident_energy': energy,
        'raw_xas': rw_xas,
        'cleaned_xas': cleaned_xas})
    path = get_spectrum_path(scan, roi_id, kind='pilatus_xas')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved XAS from sample {scan.filename} into {path}")

def save_amptek(scan: 'XASScan',roi, energy, intensity):
    df = pd.DataFrame({
        'emited_energy': energy,
        'intensity': intensity})


    path = get_spectrum_path(scan, roi, kind='amptek_xas')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved XAS from sample {scan.filename}  into {path}")

def save_xes_from_RIXS(scan: 'RIXSMap', roi_id, emitted_energy, slices: np.ndarray, energies: list):
    'save all requested slices (at energies) into one file'
    df = pd.DataFrame(slices.T, columns = [f"{e:.2f} eV" for e in energies])
    df.insert(0, 'emision energy', emitted_energy)
    # Add emitted energy as the first column
    path = get_spectrum_path(scan, roi_id, kind='pilatus_xes')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved XES slices from sample {scan.filename} into {path}")

def save_plot(scan: 'BaseScan', fig: plt.Figure, descriptor: str = "plot"):
    """save figure as png in the scan directory"""
    path = get_plot_path(scan,descriptor)
    fig.savefig(path,dpi=300)
    #plt.close(fig)
    logger.info(f"Saved plot from sample {scan.filename} into {path}")



@dataclass
class BaseScan:
    number: int
    filename: str
    #sample: 'Sample'

    data: Optional[Dict] = None
    type: str = None
    energy: Optional[np.ndarray] = None
    electrode_id: Optional[int] = None
    ROIs: Optional[np.ndarray] = None
    _preloaded_data: Optional[Dict] = None
    source_scans: Optional[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if self._preloaded_data is not None:
            self.data = self._preloaded_data
            self._preloaded_data = None
            self.define_energy()
        else:
            self.load_data()

    def print_info(self):
        print(f"{self.type} no. {self.number}: {self.filename}")

    def load_data(self):
        if not self.data:
            try:
                self.data = read_nxs_file(self.filename)
                logger.info("Successfully loaded data from %s", self.filename)
            except Exception as e:
                logger.error("Failed to load data from %s: %s", self.filename, str(e))
                self.data = None
    def define_energy(self,energy: Optional[np.ndarray] = None):
        if energy is not None:
            self.energy = energy
            logger.info("energy set for sample %i", self.number)
        else:
            try:
                self.energy = self.data['energies']
            except:
                logger.error("no energy data found for sample %i", self.number)
    def detect_type(self):

        if not self.data:
            self.data = read_nxs_file(self.filename)
        scan_command = self.data.get('scan_command', '')
        no_of_images = len(self.data.get('images', []))

        if 'sample_' in scan_command:
            self.scan_type = 'sample alignement'
        elif 'Amptek' in self.data:
            self.scan_type = 'XAS Scan'
        elif 'images' in self.data and no_of_images > 1:
            self.scan_type = 'RIXS map'
        elif 'images' in self.data and no_of_images == 1:
            self.scan_type = 'XES Scan'
        else:
            self.scan_type = 'Unknown'

        return self.scan_type


    def auto_detect_ROI(self, threshold: float = 0.4, Plot = True, min_region_width: int=5):
        self.ROIs = {}
        ROIs_found = []

        pilatus_image = self.data['images']
        summed_pilatus = np.sum(pilatus_image, axis=0)

        profile = np.sum(summed_pilatus, axis=0)
        threshold = np.max(profile) * threshold
        above_threshold = profile > threshold

        transitions = np.diff(above_threshold.astype(int))
        rising_edges = np.where(transitions == 1)[0] + 1
        falling_edges = np.where(transitions == -1)[0] + 1

        if len(rising_edges) == 0 or len(falling_edges) == 0:
            return []
        if above_threshold[0]:
            rising_edges = np.insert(rising_edges, 0, 0)
        if above_threshold[-1]:
            falling_edges = np.append(falling_edges, len(above_threshold))

        ROIs= []
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        E = np.arange(0, 195)

        for start, end in zip(rising_edges, falling_edges):
            if end - start >= min_region_width:
                roi_id = f"ROI{len(ROIs_found) + 1}"
                ROIs_found.append((roi_id, (start, end)))
                ROIs.append((start, end))
                #self.ROIs.append((start, end))


        if Plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), squeeze=False)
            ax = axs[0, 0]
            ax.set_title('Sum over all E - ' + self.filename.split('.')[0])
            im = ax.pcolormesh(summed_pilatus, shading='auto', vmax=3000, vmin=0)
            ax.set_ylim(195, 0)
            ax.set_xlabel('pixel')
            ax.set_ylabel('pixel')
            fig.colorbar(im, ax=ax)

            for (left, right), color in zip(ROIs, colors):
                ax = axs[0, 0]
                ax.axvline(left, color=color, linestyle='--')
                ax.axvline(right, color=color, linestyle='--')
                ax = axs[0, 1]
                ax.plot(E, np.sum(pilatus_image[:, :, left:right], axis=(0, 2)), color=color,
                        label='ROI ' + str(left) + ' - ' + str(right) + '')
                ax.set_title(f'Summed XES spectra')
                ax.grid(visible = True, alpha = 0.3)
                ax.set_xlabel('Energy (px)')
                ax.set_ylabel('Summed intensity')
                ax.legend()
            #plt.show()
            save_plot(self, fig, descriptor=f"{self.filename}_calibration_summary")

        for roi_id, roi_tuple in ROIs_found:
            self.ROIs[roi_id] = roi_tuple
        return self.ROIs

    def add_roi(self, start: int, end: int):
        if not hasattr(self, "ROIs") or self.ROIs is None:
            self.ROIs = {}
        roi_id = f"ROI{len(self.ROIs) + 1}"
        self.ROIs[roi_id] = (start, end)
        logger.info(f"Added manual ROI {roi_id}: ({start}, {end})")

@dataclass
class XASScan(BaseScan):
    xas_data: Dict[str, Any] = field(default_factory=dict, init=False)
    def plot(self, save=False):
        """Plot the XAS spectrum.
        1. summed amptek images
        2. heatmap of the pilatus detector vs. incident energy
        3. XAS spectrum within ROI which is shown on the image 1
        return the XAS spectrum"""

        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])  # 2 rows, 2 columns


        position = self.data['position']
        Amptek = self.data['Amptek']
        roi = [220,240]
        roi_start = roi[0]
        roi_end = roi[1]

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(np.arange(100, 4000), np.sum(Amptek[:, 100:4000], axis=0), label='Amptek')
        ax1.set_xlim(100, 1000)
        ax1.grid(visible=True, alpha=0.3)
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Summed Counts')
        ax1.set_title(f"Summed amptek counts for {self.filename}")
        ax1.axvline(roi_start, color='red')
        ax1.axvline(roi_end, color='red')

        # Left bottom plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.pcolormesh(np.arange(4096), position, np.log(Amptek + 1))
        ax2.set_xlim(100, 1000)
        ax2.set_ylabel('Incident energy [eV]')
        ax2.set_xlabel('Pixel')
        ax2.set_title('Logscale of Amptek detector')
        ax2.axvline(roi_start, color='red')
        ax2.axvline(roi_end, color='red')

        # Right tall plot spanning both rows
        ax3 = fig.add_subplot(gs[:, 1])
        Amptek_roi = np.sum(Amptek[:, roi_start:roi_end], axis=1)
        ax3.plot(position, Amptek_roi, '-', color='black', label = 'Amptek')
        ax3.set_title("Amptek ROI projection")
        ax3.set_xlabel("Incident energy [eV]")
        ax3.set_ylabel("Summed ROI Counts")
        ax3.grid(visible=True, alpha=0.3)

        save_plot(self, fig, descriptor=f"{self.filename}_amptek_XAS")
        if save:
            save_amptek(self, roi, self.data['position'], Amptek_roi)


        plt.tight_layout()

        self.xas_data = {
            "incident_energy": self.data["position"],
            "intensity": Amptek_roi
        }

    def normalize_spectrum(self):
        # XAS-specific method
        pass

@dataclass
class RIXSMap(BaseScan):

    calibration_data: Dict[str, Any] = field(default_factory=dict, init=False)  # Store detailed fit results
    calibration_line: Optional[np.ndarray] = None  # Store slope and intercept
    xas_data:  Dict[str, Any] = field(default_factory=dict, init=False)
    xes_data:  Dict[str, Any] = field(default_factory=dict, init=False)


    def slice(self, absorption_energy: Optional[np.ndarray] = None, save = False):
        self.xes_slices = {}
        """returns the slices of the map at energies specified in absorption_energy list"""
        if absorption_energy is None:
            absorption_energy = [2460,2469.5,2471,2472,2481,2495] #default values

        pilatus_image = self.data['images']
        energy = self.data['energies']
        filename = self.filename

        for roi_id, roi in self.ROIs.items():
            slices = []
            #roi_id = f"roi_{roi[0]}_{roi[1]}"
            #roi_id = f"ROI{len(ROIs_found) + 1}"

            pixel_calibration = self.calibration_data[roi_id]['line']
            E2 = np.polyval(pixel_calibration, np.arange(0, 195))

            pilatus_sum = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)

            fig, axs = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)

            ax = axs[0, 0]
            ax.set_title(f"Sum ROI {roi[0]}-{roi[1]} of {filename}")
            im = ax.pcolormesh(E2, energy, pilatus_sum, shading='auto')
            ax.set_xlabel('Emitted Energy (eV)')
            ax.set_ylabel('Incident Energy (eV)')
            fig.colorbar(im, ax=ax)
            # plt.axis('square')
            ax.set_xlim(min(E2), max(E2))

            for line in absorption_energy:
                ax.axhline(line)

            ax = axs[0, 1]
            ax.set_title(f'Emission lines at incident energies')
            for target_E in absorption_energy:
                # Find closest index to the desired energy
                idx = (np.abs(energy - target_E)).argmin()
                emission_line = pilatus_sum[idx]
                ax.plot(E2, emission_line, label=f'{energy[idx]:.2f} eV')
                slices.append(emission_line)

            ax.set_xlabel('Emitted Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.grid(visible=True, alpha=0.3)
            ax.set_xlim(min(E2), max(E2))
            ax.legend()

            if hasattr(self, "source_scans") and self.source_scans:
                ax.text(0.01, 0.99, f"From scans: {', '.join(map(str, self.source_scans))}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='left',
                        color='gray'
                        )

            plt.tight_layout()
            #plt.show()
            save_plot(self, fig, descriptor=f"{roi_id}_XES")
            self.xes_slices[roi_id] = {
                "absorption_energies": absorption_energy,
                "emitted_energy": E2,
                "slices": {f"{E:.2f}": slices[n] for n, E in enumerate(absorption_energy)}
            }
            if save:
                save_xes_from_RIXS(self, roi_id, E2, np.array(slices), absorption_energy)



    def energy_calibration(self, vmax=None, plot=True, save = True):
        """calibrate energy axis by fitting the gaussians to the elastic peaks, that are isolated
        somewhere below the given linear line that should be around elastic peak
        TODO: save elastic peaks or, maybe fit parameters, for later removal
        returns: parameter of the elastic line
        """
        self.calibration_line = []

        pilatus_image = self.data['images']
        energy = self.data['energies']
        filename = self.filename


        for roi_id, roi in self.ROIs.items():
            result = None
            #roi_id = f"roi_{roi[0]}_{roi[1]}"
            #roi_id = f"ROI{len(ROIs_found) + 1}"

            cache_path = get_calibration_path(self, roi_id)

            if roi_id in self.calibration_data:
                logger.info(f"Calibration already exists for ROI: {roi_id}.")
                continue

            if cache_path.exists():
                logger.info(f"Loading calibration from cache for ROI: {roi_id}.")
                self.calibration_data[roi_id] = load_pickle(cache_path)
                continue

            else:
                logger.info(f"Calibrating energy for {roi_id}: {roi[0]}-{roi[1]}")


                pilatus_sum = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)

                line_ends = ((energy[0], energy[np.argmax(pilatus_sum[:,-1])]), (np.argmax(pilatus_sum[0]), 195))


                approx_line_ene2pix = np.polyfit(line_ends[0], line_ends[1], 1)
                pixel_axis = np.arange(0, 195)

                if plot:
                    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
                    ax = axs[0, 0]
                    ax.set_title(f"Sum ROI {roi[0]}-{roi[1]} of {filename}")
                    g1 = ax.pcolormesh(np.arange(0, 195), energy, pilatus_sum, vmax=vmax)
                    plt.colorbar(g1, ax=ax)
                    ax.set_xlabel("pixel")
                    ax.set_ylabel("Incident Energy [eV]")
                    ax.set_xlim(0, 195)
                    ax.plot(np.polyval(approx_line_ene2pix, energy), energy, color='red')

                fit_results = []
                for j, (e, data) in enumerate(zip(energy, pilatus_sum)):

                    x_max = int(np.polyval(approx_line_ene2pix, e))  # at which pixel are we looking for elastic peak a
                    if x_max > data.shape[0] - 10:  # stop when we run out of pixels (elastic runs out)
                        continue
                    # presumed peakmaks 5 px left and right from the
                    # mask = [x_max-5:x_max+5]

                    rng = 20
                    x = np.linspace(0, len(pixel_axis), len(pixel_axis))[x_max - rng + 5:x_max + rng]
                    y = data[x_max - rng + 5:x_max + rng]

                    background_order = 1   #background aroun elstic peak is estimated with a linear function... not true when close to the XES line
                    back_model = PolynomialModel(degree=background_order, prefix='bkg_')
                    gauss_model = GaussianModel(prefix='g_')
                    model = back_model + gauss_model
                    params = model.make_params()

                    for i in range(background_order + 1):
                        params[f'bkg_c{i}'].set(value=1)

                    params['g_amplitude'].set(value=np.max(y), min=0)
                    params['g_center'].set(value=x[np.argmax(y)])
                    params['g_sigma'].set(value=0.5)

                    result = model.fit(y, params, x=x)

                    fit_results.append({
                        'energy': e,
                        'g_amplitude': result.params['g_amplitude'].value,
                        'g_center': result.params['g_center'].value,
                        'g_fwhm': 2.3548 * result.params['g_sigma'].value,
                        'g_intensities': result.params['g_height'].value,
                        'bkg_c0': result.params['bkg_c0'].value,
                        'bkg_c1': result.params['bkg_c1'].value,
                    })

                    if plot and j % 25 == 0:
                        ax = axs[0, 1]
                        ax.plot(x, y, color='black')
                        # ax.plot(x_fit, result.best_fit, color = 'red', label = 'fit')
                        # ax.plot(x, result.eval_components(x=x)['bkg_'], '--', label='Background')

                        ax.plot(x, result.eval_components(x=x)['g_'], '--', label=f'Gaussian at {e:.2f} eV')
                        ax.axvline(x_max, color='gray', alpha=0.5, linestyle='--'
                                   )
                        ax.set_title(f'Fit to elastic peak')
                        ax.set_xlabel('energy (px)')
                        ax.set_ylabel(f'intesity')
                        ax.grid(visible=True, alpha=0.3)
                        ax.legend()

                fit_data = pd.DataFrame(fit_results)
                # print(fit_data)





                if len(fit_data) >= 3:
                    lin_model = LinearModel(prefix='lin_')
                    params = lin_model.make_params()
                    result = lin_model.fit(fit_data['energy'], params, x=fit_data['g_center'])

                    slope = result.params['lin_slope'].value
                    slope_err = result.params['lin_slope'].stderr or 1e-10
                    intercept = result.params['lin_intercept'].value
                    intercept_err = result.params['lin_intercept'].stderr or 1e-10

                    fit_text = (f"Slope = {slope:.4f} ± {slope_err:.4f}\n"
                                f"Intercept = {intercept:.4f} ± {intercept_err:.4f}")

                    line = np.array([result.params['lin_slope'].value, result.params['lin_intercept'].value])
                    fwhm_e = fit_data['g_fwhm'] * result.params['lin_slope'].value
                    mean = np.mean(fwhm_e)
                    fit_data['e_fwhm'] = fwhm_e

                elif len(fit_data) == 2:
                    x1, x2 = fit_data['g_center'].values
                    y1, y2 = fit_data['energy'].values
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1

                    fit_text = (f"Slope = {slope:.4f} (2-point)\n"
                                f"Intercept = {intercept:.4f} (2-point)")

                    logger.warning(f"Only 2 points available for calibration in ROI {roi_id} — using manual fit.")

                else:
                    logger.warning(f"Not enough data points for calibration in ROI {roi_id}. Skipping.")
                    continue

                if plot:
                    ax = axs[1, 0]
                    ax.set_title('Enegy calibration' + filename)
                    ax.scatter(fit_data['g_center'], fit_data['energy'], color='black', s=10)
                    if len(fit_data) >= 3:
                        ax.plot(fit_data['g_center'], result.best_fit, color='red', label='fit')
                    else:
                        x_vals = fit_data['g_center']
                        y_vals = slope * x_vals + intercept
                        ax.plot(x_vals, y_vals, color='red', label='manual 2-point fit')


                    ax.set_title(f'Elastic calibration {filename}')
                    ax.set_xlabel('Y pixels')
                    ax.grid(visible=True, alpha=0.3)
                    ax.set_ylabel(f'Energy (eV)')
                    ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    if len(fit_data) >= 3:
                        ax = axs[1, 1]
                        ax.set_title('Gauss FWHM' + filename)
                        ax.plot(calibrate_energy_ax(fit_data['g_center'], line),
                                fit_data['g_fwhm'] * result.params['lin_slope'].value, color='black', label='fit')

                        ax.set_title(f'Elastic peak width')
                        ax.grid(visible=True, alpha=0.3)
                        ax.axhline(y=mean, color='r', linestyle='--', alpha=0.7)
                        ax.text(0.05, 0.95, f"Mean FWHM = {mean:.4f}", transform=ax.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        ax.set_xlabel('Incident Energy (eV)')
                        ax.set_ylabel(f'FWHM (eV)')

                        ax = axs[2, 0]
                        ax.plot(fit_data['energy'], fit_data['g_intensities'], color='black')
                        ax.set_title('Elastic peak Intensity')
                        ax.set_xlabel('Incident Energy (eV)')
                        ax.set_ylabel('Intensity')
                        ax.grid(visible=True, alpha=0.3)

                        ax = axs[2, 1]

                        energy_shifts = np.array(np.polyval(line, fit_data['g_center']) - fit_data['energy'])

                        ax.plot(fit_data['energy'], energy_shifts, color='black')
                        ax.set_title('Energy Shift (Calibrated-True)')
                        ax.set_xlabel('Incident Energy (eV)')
                        ax.set_ylabel('Energy Shift (eV)')
                        ax.grid(visible=True, alpha=0.3)
                        # ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                        ax.axhline(y=np.mean(energy_shifts), color='r', linestyle='--', alpha=0.7)  # Add zero-line reference

                        plt.tight_layout()

                    save_plot(self, fig, descriptor=f"{roi_id}_calibration_summary")


            self.calibration_data[roi_id] = {
                'line': (np.array([slope, intercept])),
                'mean_fwhm': float(np.mean(fit_data['e_fwhm'])) if 'e_fwhm' in fit_data else None,
                'gaussians': fit_data,
                'roi_id': roi_id,
                'roi_pixels': roi
            }

            logger.info(f"Calibration for scan {self.number}. "
                   f"Slope: {slope:.4f}, "
                   f"Intercept: {intercept:.4f}")



            if save:
                path = get_calibration_path(self, roi_id)
                save_pickle(self.calibration_data[roi_id], path)


    def optimize_ROI(self, roi_range: tuple=(280,370), width: int=5, step: int = 5, plot = True):
        "check the region between roi_range in steps to find the optimum FWHM"

        fwhms = []
        centers = []
        results = []

        left_start, left_end = roi_range
        for offset in range (left_start, left_end - width, step):
            test_roi = (offset, offset + width)
            roi_id = f"roi_{test_roi[0]}_{test_roi[1]}"
            logger.info(f"Testing ROI {roi_id}")

            original_rois = self.ROIs
            self.ROIs = [test_roi]
            self.calibration_data.pop(roi_id, None)  # ensure clean
            self.energy_calibration(plot=False, save=False)

            result = self.calibration_data[roi_id]
            mean_fwhm = result["mean_fwhm"]
            fwhms.append(mean_fwhm)
            centers.append(offset + width / 2)
            results.append({
                "roi": test_roi,
                "center": offset + width / 2,
                "mean_fwhm": mean_fwhm
            })
            self.ROIs = original_rois
            # Restore original ROIs


        df = pd.DataFrame(results)

        if plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(df["center"], df["mean_fwhm"], marker="o", color = 'black')
            ax.set_xlabel("ROI center (pixels)")
            ax.set_ylabel("Mean FWHM (eV)")
            ax.set_title(f"Elastic FWHM vs ROI center with width {width} for Scan {self.number}")
            ax.grid(True, alpha=0.3)

            save_plot(self, fig, descriptor="optimize_ROI")

        return df




    def project_XAS(self, remove_elastic=False, save = False):
        self.xas_data = {}
        """ Projects the XAS from the RIXS map, removing elastic peaks if specified."""
        pilatus_image = self.data['images']
        filename = self.filename
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)

        for roi_id, roi in self.ROIs.items():
            if roi_id not in self.calibration_data:
                raise ValueError(f"No calibration data found for ROI {roi_id}")

            #roi_id = f"roi_{roi[0]}_{roi[1]}"
            pixel_calibration = self.calibration_data[roi_id]['line']
            energy = self.data['energies']

            xas_spectrum = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(1, 2)) #projection onto incident energy axi


            ax = axs[0, 0]
            ax.set_title(f'Projected XAS for ROIs')
            ax.plot(energy, xas_spectrum, label=f'Projected XAS for ROI {roi[0]}-{roi[1]}')
            ax.set_xlabel('Incident Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.grid(visible=True, alpha=0.3)
            ax.legend()


            if roi_id not in self.calibration_data:
                raise ValueError(f"No calibration data found for ROI {roi_id}")

            RIXS_map = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)
            energy = self.data['energies']

            pixel_calibration = self.calibration_data[roi_id]['line']
            pixel_axis = np.arange(0, 195)
            E2 = np.polyval(pixel_calibration, np.arange(0, 195))


            # replace elastic peak with a linear background determined before
            N = 15
            gaussians = self.calibration_data[roi_id]['gaussians']
            for index, row in gaussians.iterrows():
                c0 = row['bkg_c0']
                c1 = row['bkg_c1']
                peak = int(row['g_center'])

                #replace the elastic peak with a linear background in 10 px rnge around the peak
                if peak > 10 and peak < len(E2):
                    if peak < len(E2) - N:
                        RIXS_map[index][peak - 10:int(peak) + 10] = c0 + c1 * pixel_axis[peak - 10:peak + 10]
                    else:
                        RIXS_map[index][peak - 10:-1] = c0 + c1 * pixel_axis[peak - 10:-1]



            cleaned_xas_spectrum = np.sum(RIXS_map[:, :-N], axis=1)  # projection onto incident energy axis while omitting the last 5px

            ax = axs[0, 1]
            ax.grid(visible=True, alpha=0.3)
            ax.set_title(f'Projected cleaned XAS for ROIs')
            ax.plot(energy, cleaned_xas_spectrum, label=f'Projected XAS for ROI {roi[0]}-{roi[1]}')
            ax.set_xlabel('Incident Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.legend()

            save_plot(self, fig, descriptor=f"{self.filename}_XAS_projections")

            self.xas_data[roi_id] = {
                "raw": xas_spectrum,
                "clean": cleaned_xas_spectrum,
                "incident_energy": energy
            }
            if save:
                save_xas_from_RIXS(self, roi_id, energy, xas_spectrum, cleaned_xas_spectrum)

                #plot the rixs map without elastic peak
                # ax = axs[0, 0]
                # ax.set_title(f"Sum ROI {roi[0]}-{roi[1]} of {filename}")
                # im = ax.pcolormesh(E2[:-N], energy, RIXS_map[:, :-N], shading='auto')
                # ax.set_xlabel('Emitted Energy (eV)')
                # ax.set_ylabel('Incident Energy (eV)')
                # fig.colorbar(im, ax=ax)
                # # plt.axis('square')
                # ax.set_xlim(min(E2), max(E2))



def calibrate_energy_ax(data, line):
    data = np.array(data)
    k = line[0]
    n = line[1]
    return data*k + n




@dataclass
class XESScan(BaseScan):
    calibration: Dict[str, Any] = field(default_factory=dict, init=False)  # dictionary of roi and calibration data, e.g. {'roi_1_2': {'line': [slope, intercept], 'mean_fwhm': 0.5, 'gaussians': pd.DataFrame}} = None
    #need to take energy calibration from somewwhere.
    xes_data: Dict[str, Any] = field(default_factory=dict, init=False)
      # sum over all energies, if tere is only one then fine


    def energy_calibration_from_scan(self, RIXSscan: RIXSMap):
        #take energy calibration from the RIXS scan, also adopt same ROIs

        #self.calibration = RIXSscan.calibration_data
        #self.ROIs = RIXSscan.ROIs

        cache_dir = get_scan_dir(RIXSscan)

        if self.ROIs is None:
            self.ROIs = {}


        for file in cache_dir.glob('calibration_*.pkl'):
                result = load_pickle(file)
                roi_left, roi_right = result['roi_pixels']
                roi_id = result['roi_id']

                #roi_id = f"roi_{roi_left}_{roi_right}"
                self.calibration[roi_id] = result

                if self.ROIs is None:
                    self.ROIs = {}

                self.ROIs[roi_id] = (roi_left, roi_right)





    def plot(self, save=True):
        summed_pilatus = np.sum(self.data['images'], axis=0)
        if not self.calibration:
            logger.warning('no calibration assigned to scan %s', self.filename)

        fig, axs = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)

        ax = axs[0, 0]
        ax.set_title(f"Pilatus image of {self.filename.split('.')[0]} at energy: {self.energy[0]:.2f}")
        im = ax.pcolormesh(summed_pilatus, shading='auto', vmax=3000, vmin=0)
        ax.set_ylim(195, 0)
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')
        fig.colorbar(im, ax=ax)
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        for (roi_id, roi), color in zip(self.ROIs.items(), colors):
            #roi_id = f"roi_{roi[0]}_{roi[1]}"
            if roi_id not in self.calibration:
                logger.warning(f"No calibration data found for ROI {roi_id}. Skipping plot.")
                continue

            pixel_calibration = self.calibration[roi_id]['line']
            E2 = np.polyval(pixel_calibration, np.arange(0, 195))

            pilatus_image = self.data['images']
            energy = self.energy

            xes_spectrum = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(1, 2))
            ax = axs[0, 0]

            ax.axvline(roi[0], color=color, linestyle='--')
            ax.axvline(roi[1], color=color, linestyle='--')
            ax = axs[0, 1]
            XES = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(0, 2))
            ax.plot(E2, XES, color=color,
                    label='ROI ' + str(roi[0]) + ' - ' + str(roi[1]) + '')
            ax.set_title(f'XES spectra')
            ax.grid(visible=True, alpha=0.3)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.legend()

            save_xes_from_RIXS(self, roi_id, E2, XES, self.energy)
            if not hasattr(self, "xes_data"):
                self.xes_data = {}

            self.xes_data[roi_id] = {
                "incident_energy": self.energy,
                "emitted_energy": E2,
                "intensity": XES
            }

        save_plot(self, fig, descriptor=f"{self.filename}_XES")




@dataclass
class Sample:
    electrode_id: int
    name: str
    cycle_info: str
    scans: Dict[int, 'BaseScan'] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


    def _generate_filename(self, scan_number: int) -> str:
        return f"Electrode_{self.electrode_id:02d}_{scan_number:04d}.nxs"

    def add_scans(self, scan_numbers: np.ndarray, energy: Optional[np.ndarray]=None) -> None:
        for scan_number in scan_numbers:
            filename = self._generate_filename(scan_number)
            temp_scan = BaseScan(number=scan_number, filename=filename, electrode_id=self.electrode_id)
            if not temp_scan.data:
                logger.warning("Skipping %s due to load failure", filename)
                continue

            scan_type = temp_scan.detect_type()

            if scan_type == 'XAS Scan':
                scan = XASScan(number=scan_number, filename=filename, electrode_id=self.electrode_id, type=scan_type, _preloaded_data=temp_scan.data
)
            elif scan_type == 'RIXS map':
                scan = RIXSMap(number=scan_number, filename=filename, electrode_id=self.electrode_id,type=scan_type,_preloaded_data=temp_scan.data)
            elif scan_type == 'XES Scan':
                scan = XESScan(number=scan_number, filename=filename, electrode_id=self.electrode_id,type=scan_type,_preloaded_data=temp_scan.data)
            else:
                scan = BaseScan(number=scan_number, filename=filename, electrode_id=self.electrode_id,type=scan_type,_preloaded_data=temp_scan.data)

            if energy is not None:
                scan.data['energies'] = energy


            self.scans[scan_number] = scan

    def clear_data(self) -> None:
        for scan in self.scans.values():
            scan.clear_data()


    def Sample_summary(self):
        df = pd.DataFrame([{
            'No.': no,
            'Type': scan.type,
            'Start time': scan.data.get('start_time', [0]),
            'No energy': len(scan.energy) if scan.energy is not None else 0,
            'Exposure time': scan.data.get('exposure_time', [0])[0]/1000,
            'Scan Command': scan.data.get('scan_command'),
        } for no, scan in self.scans.items()])
        df = df.sort_values('No.')

        print(f"Sample {self.electrode_id}: {self.name}")
        print(f"Cycle: {self.cycle_info}")
        print(f"Number of scans: {len(self.scans)}")
        print(f"Metadata: {self.metadata}")

        print("\nScan Summary:")
        print(df.to_string(index=False))




    def combine_xes_scans(self, scan_numbers: List[int], tag: str = '') -> RIXSMap:
        "combine multiple XES scans into one RIXSMap object"

        xes_scans = [self.scans[n] for n in scan_numbers if n in self.scans and isinstance(self.scans[n], XESScan)]

        if not xes_scans:
            raise ValueError("No valid XESScans")

        base_number=9000
        while base_number in self.scans:
            base_number += 1

        images = np.stack([scan.data['images'][0] for scan in xes_scans], axis=0)
        energies = np.array([scan.energy[0] for scan in xes_scans])
        el_id = xes_scans[0].electrode_id

        rixs = RIXSMap(
            number=base_number,
            filename=f"synthetic_RIXS_{base_number}{'_' + tag if tag else ''}.nxs",
            #sample=self,
            electrode_id=el_id,
            type = 'sythetic RIXS',
            data={
                'images': images,
                'energies': energies,
                'scan_command': f"Combined XES scans {scan_numbers} with tag {tag}"
            }
        )

        rixs.source_scans = [scan.number for scan in xes_scans]
        self.scans[base_number] = rixs

        logger.info(f"[Synthetic] Combined {len(xes_scans)} XES scans into synthetic RIXS scan #{base_number}")
        return rixs

    def combine_xes_scans_by_time(self, scan_numbers: List[int], scan_times = List[float], tag: str = "XES_time_series"):
        """combine xes scans taken at the same energy into one object that can then be exported. Calibration will be dependant on RIXS scan
        result is a dictionary where each roi contains spectra from multiple scans labeled by timestamp and scan number"""
        xes_scans = [self.scans[n] for n in scan_numbers if n in self.scans and isinstance(self.scans[n], XESScan)]
        if len(xes_scans) != len(scan_times):
            raise ValueError("Mismatch between number of scans and scan_times")


        ref_scan = xes_scans[0]
        roi_ids = list(ref_scan.ROIs.keys())
        combined_data =  {}

        fig, axs = plt.subplots(len(roi_ids),2, figsize=(16, 16), squeeze=False)


        for n,roi_id in enumerate(roi_ids):
            # Get emitted_energy from first scan (they're assumed identical)

            emitted_energy = xes_scans[0].xes_data[roi_id]["emitted_energy"]
            data = {"emitted_energy": emitted_energy
            }
            integral_under_XES = []

            for scan, time in zip(xes_scans, scan_times):
                xes = scan.xes_data[roi_id]
                intensity = xes["intensity"]
                col_label = f"{scan.number} ({time:.1f} s)"
                data[col_label] = intensity

                ax = axs[n, 0]
                ax.set_title(f"emission for {roi_id}")
                ax.plot(emitted_energy, intensity, label = col_label)
                ax.set_xlabel('Emission energy [eV]')
                ax.set_ylabel('Intensity')
                ax.grid(visible=True, alpha=0.3)
                ax.legend()

                emin = 2455
                emax =  2475
                mask = (emitted_energy >= emin) & (emitted_energy <= emax)
                integrated_intensity = np.trapezoid(intensity[mask], emitted_energy[mask])
                integral_under_XES.append(integrated_intensity)

            ax.vline(emin, color='red', linestyle='--')
            ax.vline(emax, color='red', linestyle='--')
            ax = axs[n, 1]
            ax.set_title(f"integrated emission for {roi_id}")
            ax.plot(scan_times, integral_under_XES, color = 'black', marker='o')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Integrated intensity')

            # Convert to DataFrame
            df = pd.DataFrame(data)
            combined_data[roi_id] = df

        path = get_sample_dir(self)/f"{tag}"
        fig.savefig(path, dpi=300)

        return combined_data

    def plot_time_XES(self, scan_numbers: List[int], scan_times: List[float] ):
        """Plot the combined XES scans by time."""



def export_hd5(sample: 'Sample', filepath: Path, xes_series = None) -> None:
    with h5py.File(filepath, "w") as h5file:
        meta_grp = h5file.create_group("metadata")
        meta_grp.create_dataset("name", data=sample.name)
        meta_grp.create_dataset("electrode_id", data=sample.electrode_id)
        meta_grp.create_dataset("cycle_info", data=sample.cycle_info)

        scans_grp = h5file.create_group("scans")

        for scan_number, scan in sample.scans.items():
            scan_grp = scans_grp.create_group(f"{scan_number:04d}")
            scan_grp.attrs["type"] = scan.type or "Unknown"

            # =============== RIXSMap ===============
            if isinstance(scan, RIXSMap):
                if scan.ROIs:
                    rois_grp = scan_grp.create_group("ROIs")
                    for roi_id, roi_tuple in scan.ROIs.items():
                        roi_grp = rois_grp.create_group(roi_id)
                        roi_grp.attrs["pixels"] = roi_tuple
                        # Calibration
                        calib = scan.calibration_data.get(roi_id)
                        if calib:
                            cal_grp = roi_grp.create_group("calibration")
                            cal_grp.create_dataset("line", data=calib["line"])
                            if calib.get("mean_fwhm") is not None:
                                cal_grp.create_dataset("mean_fwhm", data=calib["mean_fwhm"])
                        # XAS
                        if roi_id in scan.xas_data:
                            xas = scan.xas_data[roi_id]
                            roi_grp.create_dataset("XAS_raw",data=np.stack([xas["incident_energy"], xas["raw"]], axis=1))
                            roi_grp.create_dataset("XAS_clean",data=np.stack([xas["incident_energy"], xas["clean"]], axis=1))
                        # XES
                        if roi_id in scan.xes_slices:
                            xes = scan.xes_slices[roi_id]
                            xes_grp = roi_grp.create_group("XES_slices")
                            for energy_label, slice_data in xes["slices"].items():
                                energy_value = float(energy_label)
                                spectrum = np.stack([xes["emitted_energy"], slice_data], axis=1)
                                xes_grp.create_dataset(f"{energy_label}", data=spectrum)

            # =============== XASScan ===============
            elif isinstance(scan, XASScan):
                xas = scan.xas_data
                xas_grp = scan_grp.create_group("Amptek_XAS")
                spectrum = np.stack([xas["incident_energy"], xas["intensity"]], axis=1)
                xas_grp.create_dataset("spectrum", data=spectrum)

            # =============== XESScan ===============
            elif isinstance(scan, XESScan):
                xes_grp = scan_grp.create_group("XES")
                for roi_id, roi in scan.ROIs.items():
                    xes = scan.xes_data[roi_id]
                    spectrum = np.stack([xes["emitted_energy"], xes["intensity"]], axis=1)
                    xes_grp.create_dataset(f"{roi_id}", data=spectrum)

        if xes_series:
            for idx, series_dict in enumerate(xes_series):
                series_grp = h5file.create_group(f"series_{idx}")
                for roi_id, df in series_dict.items():
                    roi_grp = series_grp.create_group(roi_id)
                    roi_grp.create_dataset("spectrum", data=df.to_numpy(dtype=np.float32))
                    roi_grp.create_dataset("spectrum_labels", data=np.array(df.columns.astype(str), dtype="S"))



sample = Sample(
    electrode_id=1,
    name="Electrode before cycling",
    cycle_info="1st cycle")




sample.add_scans([7])
sample.add_scans([8])
sample.scans[8].plot(save=True)

sample.scans[7].auto_detect_ROI(Plot=False)
#important: always do energy calibration first, then slice or plot XAS from map
sample.scans[7].energy_calibration(plot=True, save=True)
sample.add_scans([9])
sample.scans[9].energy_calibration_from_scan(sample.scans[7])
sample.scans[9].plot()
sample.scans[7].slice(save=True)
sample.scans[7].project_XAS(remove_elastic=True, save=True)

sample.add_scans([12,14])
sample.scans[12].energy_calibration_from_scan(sample.scans[7])
sample.scans[14].energy_calibration_from_scan(sample.scans[7])
sample.combine_xes_scans([14,12,9], tag='testE')
sample.scans[9000].auto_detect_ROI()
sample.scans[9000].energy_calibration()
sample.scans[9000].slice(sample.scans[9000].data['energies'], save = True)

plt.show()





