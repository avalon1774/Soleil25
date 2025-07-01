import os
import h5py
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import pandas as pd
from dotenv import load_dotenv

import logging

logger = logging.getLogger("Galaxies")
logger.setLevel(logging.INFO)




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


def make_sample(filename: str, rois: list[list], plot=True):
    """
    This function plots the data from a .nxs file. Used for emission Amptek spectrometer.

    :param filename:  The path to the .nxs file relative to the path given in the DATA_PATH variable.
    :param rois: The regions of interest to plot, e.g. [[100, 200], [300, 400]] for two ROIs.
    :param plot: Whether to plot the data. If False, returns a ps.Series with the summed counts in the ROI and the dict of NXS data.
    :return: If only one ROI is given and plot is false, returns a pd.Series with the summed counts in the ROI and the dict of NXS data.
    """
    data_from_file = read_nxs_file(filename)
    position = data_from_file['position']
    # DIODE = data_from_file['DIODE']
    # I01 = data_from_file['I01']
    # SDD = data_from_file['SDD']
    Amptek = data_from_file['Amptek']
    Amptek_roi = None

    fig, ax1, ax2, ax3 = None, None, None, None

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        fig.suptitle(filename)

        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]

        ax1.plot(np.arange(100, 4000), np.sum(Amptek[:, 100:4000], axis=0), 'k-', label='Amptek')
        ax1.set_xlim(100, 1000)
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Sum Counts')
        ax1.legend()

        ax2.set_ylabel('Sum Counts in ROI')
        ax2.set_xlabel('Energy [eV]')

    for nroi, roi in enumerate(rois):
        roi_start = roi[0]
        roi_end = roi[1]
        Amptek_roi = np.sum(Amptek[:, roi_start:roi_end], axis=1)
        if plot:
            ax2.grid()
            ax2.plot(position, Amptek_roi, '-', label=nroi, color='C' + str(nroi))

            ax1.axvline(roi_start, color='C' + str(nroi))
            ax1.axvline(roi_end, color='C' + str(nroi))

    if plot:
        ax3.grid()
        ax3.pcolormesh(np.arange(4096), position, np.log(Amptek + 1), shading='auto')
        ax3.set_xlim(100, 1000)
        ax3.set_ylabel('Energy [eV]')
        ax3.set_xlabel('Pixel')
        ax3.set_title('Logscale of summed ROI counts')

        fig.tight_layout()

    if len(rois) == 1 and not plot:
        df = pd.Series(data=Amptek_roi, index=pd.Series(position, name="energy [eV]"), name=filename)
        return df, data_from_file
    else:
        return None


def detect_scan_from_filename(filename, include_full_data=False):
    """
    Detects the scan type from the filename.

    Parameters:
    filename (str): The name of the file.
    include_full_data (bool): If True, returns the full data dictionary, otherwise returns an empty dictionary.

    Returns:
    tuple[str, str, dict]: The type of scan, the command string and dict of all data.
    """
    all_data = read_nxs_file(filename)
    scan_command = all_data.get('scan_command', '')
    no_of_images = len(all_data.get('images', []))

    if 'sample_' in scan_command:
        scan_type = 'sample alignement'
    elif 'Amptek' in all_data:
        scan_type = 'XAS scan'
    elif 'images' in all_data and no_of_images > 1:
        scan_type = 'RIXS scan'
    elif 'images' in all_data and no_of_images == 1:
        scan_type = 'XES one shot'
    else:
        scan_type = 'Unknown'

    if not include_full_data:
        all_data = {}
    return scan_type, scan_command, all_data


def plot_file(filename, **options):
    """
    Plots data from a file based on the detected scan type.

    Parameters:
    filename (str): The name of the file.
    **options: Additional options for plotting functions.

    TODO: does not work yet, needs to be tested
    """
    scan_type, _, _ = detect_scan_from_filename(filename)
    if scan_type == 'make_sample':
        make_sample(filename, **options)
    elif scan_type == 'plot_2Dmap':
        plot_2Dmap(filename, **options)
    else:
        raise ValueError(f"Unknown scan type for file {filename}")


def plot_2Dmap(filename, pix2e_calibration, roi, *, vmax_sum=None, vmax_rxes=None, replace=None, plot=True):
    """
    This function plots the data from a .nxs file. Used for emission Pilatus spectrometer.
    Plots 2 or 4 subplots, depending on whether the data is 1D or 2D.

    :param filename: The path to the .nxs file relative to the path given in the DATA_PATH variable.
    :param pix2e_calibration: A polynomial calibration for the pixel 2 energy conversion, [slope, intercept]
    :param roi: emission region of interest, e.g. [70, 100] for pixels on the Pilatus detector
    :param vmax_sum: Used for the maximum color value in the sum image, if None, uses the maximum value in the image
    :param vmax_rxes: Used for the maximum color value in the emission RXES map, if None, uses the maximum value in the image
    :param replace: A dictionary to replace missing data in the file. For example, if the file does not contain 'energies', you can provide a
    replacement like {'energies': np.arange(0, 1001, 10)}.
    :param plot: Whether to plot the data. If False, returns the xes_spectrum and RXES_map as Pandas Series.
    :return: If not plotting, returns: pd.Series xes_spectrum, pd.Series xas_spectrum, * dict of file data
    """

    # ===================================
    # filename = 'Electrode_01_0013.nxs'
    # p = np.load('calib.npy')
    # roi = [70,100]
    # roi = [285, 315]
    # ===================================

    data_from_file = read_nxs_file(filename)
    pilatus_image = data_from_file['images']
    if replace is None:
        replace = {}
    energy = data_from_file.get('energies', replace.get('energies', None))

    assert energy is not None, "Energy data not found. Please check the file structure or provide replacement."

    E2 = np.polyval(pix2e_calibration, np.arange(0, 195))

    xas_spectrum = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(1, 2))
    RXES_map = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)

    if len(xas_spectrum) == 1 or len(RXES_map) == 1:
        NROWS = 1
    else:
        NROWS = 2

    axs: np.typing.NDArray[2, matplotlib.axes.Axes]
    if plot:
        fig, axs = plt.subplots(nrows=NROWS, ncols=2, figsize=(16, NROWS*6), squeeze=False)

        ax: matplotlib.axes.Axes
        ax = axs[0, 0]
        if NROWS == 2:
            ax.set_title('Sum over all E - ' + filename)
        else:
            ax.set_title(f"Emission @ {energy[0]:.0f} eV - '{filename}'")  # type: ignore
        im = ax.pcolormesh(np.sum(pilatus_image, axis=0), shading='auto', vmax=vmax_sum, vmin=0)
        ax.axvline(roi[0])
        ax.axvline(roi[1])
        ax.set_ylim(195, 0)
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')
        fig.colorbar(im, ax=ax)

        ax = axs[0, 1]
        ax.set_title('XES - ' + filename)
        ax.plot(E2, np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(0, 2)))
        ax.set_xlabel('Emitted Energy (eV)')
        ax.set_ylabel('Counts')
        ax.set_xlim(min(E2), max(E2))

        if NROWS == 1:
            return None

        ax = axs[1, 0]
        ax.set_title('RXES map - ' + filename)
        im = ax.pcolormesh(E2, energy, RXES_map, shading='auto', vmax=vmax_rxes, vmin=0)
        ax.set_xlabel('Emitted Energy (eV)')
        ax.set_ylabel('Incident Energy (eV)')
        fig.colorbar(im, ax=ax)
        ax.set_xlim(min(E2), max(E2))

        ax = axs[1, 1]
        ax.set_title('XAS - ' + filename)
        ax.plot(energy, xas_spectrum)
        ax.set_xlabel('Incident Energy (eV)')
        ax.set_ylabel('Counts')
        return None

    else:
        xes = pd.Series(index=pd.Series(E2, name="Energy[eV]"),
                        data=np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=(0, 2)), name="XES " + filename)
        xas = pd.Series(index=pd.Series(energy, name="Energy[eV]"),
                        data=xas_spectrum, name="XAS " + filename)
        return xes, xas, data_from_file





def make_slice(filename: str, roi: list, pix2en_calibration: list, e0: float, de=0.1, plot=True):
    """
    This function makes a slice from the data in a .nxs file in emission axis, summing over the specified region of interest (ROI) on detector.


    :param filename: The path to the .nxs file relative to the path given in the DATA_PATH variable.
    :param roi: The region of interest on the detector, e.g. [70, 100] for pixels on the Pilatus detector.
    :param pix2en_calibration: A polynomial calibration for the pixel to energy conversion, e.g. [slope, intercept].
    :param e0: Energy at which the slice is made, e.g. 2471.9 for the RIXS scan.
    :param de: Energy window ABOVE e0 to include in the slice, e.g. 0.1 eV.
    :param plot: Whether to plot the resulting spectrum. Default is True.
    :return: np.ndarray Spectrum, number of spectra summed
    """
    # filename = 'Electrode_01_0007.nxs'
    data_from_file = read_nxs_file(filename)
    energies = data_from_file['energies']
    images = data_from_file['images']
    # roi = roi_right

    spectra = []
    # e0 = 2471.9
    for e, image in zip(energies, images):
        if e < e0 or e > e0 + de:
            continue
        spectra.append(np.sum(image[:, roi[0]:roi[1]], axis=1))
    spectra = np.array(spectra)
    spectrum: np.ndarray = np.sum(spectra, axis=0)

    if plot:
        pixels = np.arange(len(spectrum))
        E = np.polyval(pix2en_calibration, pixels)
        plt.title(f'{filename} N={len(spectra)}')
        plt.plot(E, spectrum, label=f'E={e0}+-{de} eV')
        plt.xlabel('Emitted energy [eV]')
        plt.ylabel('Counts')
        plt.legend()
        plt.grid()

    return spectrum, len(spectra)
