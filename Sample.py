from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
        # XAS-specific processing
        self.load_data()
        # Process Amptek data
        pass

    def normalize_spectrum(self):
        # XAS-specific method
        pass

@dataclass
class RIXSMap(BaseScan):

    def process(self):
        # RIXS-specific processing
        self.load_data()
        # Process image data
        pass

    def analyze_map(self):
        # RIXS-specific method
        pass

    def energy_calibration(self):
        # calibrate energy axis by fitting the gaussians to the elastic peaks:


        pass

@dataclass
class XESScan(BaseScan):
    def process(self):
        # XES-specific processing
        self.load_data()
        # Process single image
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


sample.add_scans([3,45,7])





