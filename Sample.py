from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lmfit.models import PolynomialModel, GaussianModel, LinearModel, LorentzianModel, VoigtModel
import pandas as pd
import logging


from galaxies import read_nxs_file
#from galaxies_ava import
import logging

""" Structure of data: 
    Sample class, cotains all info about the sample (name, cycle...), and it holds all the scans
    that are associated with its number. Each scan is a BaseScan which has some inharent functionality and . 
    according to the type, each one falls under one if subclasses
    XAS - line XAS scan
    EXS - one-shot XES scan at fixed energy
    RIXS map - a series of individual XES scans for each incident energy"""


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sample")

     # If you have logs from galaxies module


@dataclass
class BaseScan:
    number: int
    filename: str
    sample: 'Sample'

    data: Optional[Dict] = None
    type: str = None
    energy: Optional[np.ndarray] = None
    ROIs: Optional[np.ndarray] = None
    _preloaded_data: Optional[Dict] = None

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
        self.ROIs = []
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
        colors = ['red', 'green', 'blue', ]
        E = np.arange(0, 195)

        for start, end in zip(rising_edges, falling_edges):
            if end - start >= min_region_width:
                ROIs.append((start, end))
                self.ROIs.append((start, end))


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
                ax.set_xlabel('Energy (px)')
                ax.set_ylabel('Summed intensity')
                ax.legend()
            plt.show()

        return ROIs

@dataclass
class XASScan(BaseScan):
    def plot(self):
        # plot the scan and return it, to be used as comparison
        self.load_data()
        # Process Amptek data
        pass

    def normalize_spectrum(self):
        # XAS-specific method
        pass

@dataclass
class RIXSMap(BaseScan):

    calibration_data: Dict[str, Any] = field(default_factory=dict, init=False)  # Store detailed fit results
    calibration_line: Optional[np.ndarray] = None  # Store slope and intercept


    def slice(self, absorption_energy: Optional[np.ndarray] = None):
        """returns the slices of the map at energies specified in absorption_energy list"""
        if absorption_energy is None:
            absorption_energy = [2460,2469.5,2471,2472] #default values

        pilatus_image = self.data['images']
        energy = self.data['energies']
        filename = self.filename

        for roi in self.ROIs:

            roi_id = f"roi_{roi[0]}_{roi[1]}"
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
            ax.set_xlabel('Emitted Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.set_xlim(min(E2), max(E2))
            ax.legend()

            plt.tight_layout()
            plt.show()



    def energy_calibration(self, vmax=None, plot=True):
        """calibrate energy axis by fitting the gaussians to the elastic peaks, that are isolated
        somewhere below the given linear line that should be around elastic peak
        TODO: save elastic peaks or, maybe fit parameters, for later removal
        returns: parameter of the elastic line
        """
        self.calibration_line = []

        pilatus_image = self.data['images']
        energy = self.data['energies']
        filename = self.filename


        for roi in self.ROIs:
            roi_id = f"roi_{roi[0]}_{roi[1]}"

            if roi_id in self.calibration_data:
                logger.info(f"Calibration already exists for ROI: {roi_id}.")

                continue


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
                if x_max > data.shape[0] - 15:  # stop when we run out of pixels (elastic runs out)
                    continue
                # presumed peakmaks 5 px left and right from the
                # mask = [x_max-5:x_max+5]

                rng = 20
                x = np.linspace(0, len(pixel_axis), len(pixel_axis))[x_max - rng + 5:x_max + rng]
                y = data[x_max - rng + 5:x_max + rng]

                background_order = 1
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
                    #        'bkg_c0': result.params['bkg_c0'].value,
                    #        'bkg_c1': result.params['bkg_c1'].value,
                })

                if plot and j % 50 == 0:
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

            lin_model = LinearModel(prefix='lin_')
            params = lin_model.make_params()
            result = lin_model.fit(fit_data['energy'], params, x=fit_data['g_center'])

            slope = result.params['lin_slope'].value
            slope_err = result.params['lin_slope'].stderr
            intercept = result.params['lin_intercept'].value
            intercept_err = result.params['lin_intercept'].stderr

            fit_text = (f"Slope = {slope:.4f} ± {slope_err:.4f}\n"
                        f"Intercept = {intercept:.4f} ± {intercept_err:.4f}")

            line = np.array([result.params['lin_slope'].value, result.params['lin_intercept'].value])
            fwhm_e = fit_data['g_fwhm'] * result.params['lin_slope'].value
            mean = np.mean(fwhm_e)
            fit_data['e_fwhm'] = fwhm_e

            if plot:
                ax = axs[1, 0]
                ax.set_title('Enegy calibration' + filename)
                ax.scatter(fit_data['g_center'], fit_data['energy'], color='black', s=10)
                ax.plot(fit_data['g_center'], result.best_fit, color='red', label='fit')
                ax.set_title(f'Elastic calibration {filename}')
                ax.set_xlabel('Y pixels')
                ax.grid(visible=True, alpha=0.3)
                ax.set_ylabel(f'Energy (eV)')
                ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
            # df = pd.DataFrame.from_dict(fit_data)


            self.calibration_line.append(np.array([slope, intercept]))
            #self.calibration_data = fit_data #add the data about the gaussians into this to maybe reuse for later

            if 'calibration' not in self.sample.metadata:
                self.sample.metadata['calibration'] = {}



            self.calibration_data[roi_id] = {
                'line': (np.array([slope, intercept])),
                'mean_fwhm': float(np.mean(fit_data['e_fwhm'])),
                'gaussians': fit_data
               }

            logger.info(f"Calibration saved for scan {self.number}. "
                   f"Slope: {slope:.4f}, "
                   f"Intercept: {intercept:.4f}")


            #return fit_data

    def project_XAS(self, remove_elastic=False):




def calibrate_energy_ax(data, line):
    data = np.array(data)
    k = line[0]
    n = line[1]
    return data*k + n


@dataclass
class XESScan(BaseScan):
    #do this last
    def process(self):
        # XES-specific processing
        self.load_data()
        # Process single image
        pass
    def combine_scans(self):
        #combine multiple scans into one to enable energy calibration
        pass


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
            temp_scan = BaseScan(number=scan_number, filename=filename, sample=self)
            if not temp_scan.data:
                logger.warning("Skipping %s due to load failure", filename)
                continue

            scan_type = temp_scan.detect_type()

            if scan_type == 'XAS Scan':
                scan = XASScan(number=scan_number, filename=filename, sample=self, type=scan_type, _preloaded_data=temp_scan.data
)
            elif scan_type == 'RIXS map':
                scan = RIXSMap(number=scan_number, filename=filename, sample=self,type=scan_type,_preloaded_data=temp_scan.data)
            elif scan_type == 'XES Scan':
                scan = XESScan(number=scan_number, filename=filename, sample=self,type=scan_type,_preloaded_data=temp_scan.data)
            else:
                scan = BaseScan(number=scan_number, filename=filename, sample=self,type=scan_type,_preloaded_data=temp_scan.data)

            if energy:
                scan.energy = energy


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


sample = Sample(
    electrode_id=1,
    name="Electrode before cycling",
    cycle_info="1st cycle")


sample.add_scans([7])
ROIS = sample.scans[7].auto_detect_ROI()
#important: always do energy calibration first, then slice or plot XAS from map
sample.scans[7].energy_calibration(plot=True)
sample.scans[7].slice()
plt.show()





