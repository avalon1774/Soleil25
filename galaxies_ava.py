#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:18:07 2025

@author: ava
"""

import os
import h5py
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from lmfit.models import PolynomialModel, GaussianModel, LinearModel, LorentzianModel, VoigtModel
import pandas as pd
from galaxies import read_nxs_file, plot_2Dmap, make_sample



#pixel_calibration = np.array([6.34133926e-01, 7.32260335e+03])
roi_left = [70, 100]
roi_right = [285, 315]

        
def plot_line_XES(filename, pixel_calibration, roi, energy_slices=[2520], vmax = 300):
    # Load and prepare data
    data_from_file = read_nxs_file(filename)

    pilatus_image = data_from_file['images']
    energy = data_from_file['energies']
    
    
    E2 = np.polyval(pixel_calibration, np.arange(0, 195))   # emission energy axis
    RIXS_map = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16,7), squeeze=False)
    
    ax = axs[0, 0]
    ax.set_title('RXES map - ' + filename)
    im = ax.pcolormesh(E2, energy, RIXS_map, shading='auto', vmax = vmax)
    ax.set_xlabel('Emitted Energy (eV)')
    ax.set_ylabel('Incident Energy (eV)')
    fig.colorbar(im, ax=ax)
    # plt.axis('square')
    ax.set_xlim(min(E2), max(E2))
    
    for line in energy_slices: 
        ax.axhline(line)


    ax = axs[0, 1]
    ax.set_title(f'Emission lines at incident energies')
    for target_E in energy_slices:
        # Find closest index to the desired energy
        idx = (np.abs(energy - target_E)).argmin()
        emission_line = RIXS_map[idx]
        ax.plot(E2, emission_line, label=f'{energy[idx]:.2f} eV')
    ax.set_xlabel('Emitted Energy (eV)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(min(E2), max(E2))
    ax.legend()

    plt.tight_layout()
    plt.show()
    
#plot_line_XES('Electrode_01_0007.nxs', pixel_calibration, roi_right, energy_slices = [2473,2470])


def energy_calibration(filename,roi, plot = True):
    """ clibrate the scan by the first few scans, that contian only elastic peaks
    returns slope and intercept for the fit of first N elastic"""
    
    data_from_file = read_nxs_file(filename)

    pilatus_image = data_from_file['images']
    energy = data_from_file['energies']
    RIXS_map = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)


    fit_results = []
    N = 180
    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6), squeeze=False, dpi=300)
        ax: matplotlib.axes.Axes
        ax = axs[0, 0]

    for j in range(0,N,1):
        XES = RIXS_map[j]
        x = np.linspace(0,len(XES),len(XES))
        y = XES
    
        background_order = 1
        back_model = PolynomialModel(degree=background_order, prefix='bkg_')
        gauss_model = GaussianModel(prefix='g_')
        model = back_model + gauss_model
        params = model.make_params()
        
        for i in range(background_order + 1):
            params[f'bkg_c{i}'].set(value=1)
        
        params['g_amplitude'].set(value=np.max(y), min=0)
        params['g_center'].set(value=np.argmax(XES))
        params['g_sigma'].set(value = 10)



        result = model.fit(y, params, x=x)
        fit_results.append({
        'index': j,
        'g_amplitude': result.params['g_amplitude'].value,
        'g_center': result.params['g_center'].value,
        'g_fwhm': 2.3548 * result.params['g_sigma'].value,
#        'bkg_c0': result.params['bkg_c0'].value,
#        'bkg_c1': result.params['bkg_c1'].value,
    })

        if plot and j % 15 == 0:
            ax.plot(x, y, color='black', alpha=0.5)
            # ax.plot(x_fit, result.best_fit, color = 'red', label = 'fit')
            #ax.plot(x, result.eval_components(x=x)['bkg_'], '--', label='Background')

            ax.plot(x, result.eval_components(x=x)['g_'], '--', label=f'Gaussian at {energy[j]:.2f} eV')
            ax.set_title(f'Fit to elastic peak')
            ax.set_xlabel('energy (px)')
            ax.set_ylabel(f'intesity')
            ax.grid(visible=True)
            ax.legend()



    fit_data = pd.DataFrame(fit_results)
    #print(fit_data)
    
    lin_model = LinearModel(prefix='lin_')
    params = lin_model.make_params()
    result = lin_model.fit(energy[:N], params, x=fit_data['g_center'])


    slope = result.params['lin_slope'].value
    slope_err = result.params['lin_slope'].stderr
    intercept = result.params['lin_intercept'].value
    intercept_err = result.params['lin_intercept'].stderr


    fit_text = (f"Slope = {slope:.4f} ± {slope_err:.4f}\n"
                f"Intercept = {intercept:.4f} ± {intercept_err:.4f}")

    line = np.array([result.params['lin_slope'].value, result.params['lin_intercept'].value])

    if plot:

        fig, axs = plt.subplots(1,2, figsize=(16,7), squeeze=False)
        ax: matplotlib.axes.Axes
        ax = axs[0, 0]
        ax.set_title('Enegy calibration' + filename)
        ax.scatter(fit_data['g_center'], energy[:N])
        ax.plot(fit_data['g_center'], result.best_fit, color = 'red', label = 'fit')
        ax.set_title(f'Elastic calibration {filename}')
        ax.set_xlabel('Y pixels')
        ax.set_ylabel(f'Energy (eV)')
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax: matplotlib.axes.Axes
        ax = axs[0, 1]
        ax.set_title('Gasuu FWHM' + filename)
        ax.plot(calibrate_energy_ax(fit_data['g_center'],line), fit_data['g_fwhm']*result.params['lin_slope'].value, color = 'red', label = 'fit')
        ax.set_title(f'Elastic width {filename}')
        ax.set_xlabel('Y pixels')
        ax.set_ylabel(f'Energy (eV)')

    return np.array([result.params['lin_slope'].value, result.params['lin_intercept'].value])


def calibrate_energy_ax(data, line): 
    data = np.array(data)
    k = line[0]
    n = line[1]
    return data*k + n
    

def mask_ROI (image, ROI): 
    ROI_min = ROI[0]
    ROI_max = ROI[1]
    masked_image = image[:, ROI[0]:ROI[1]] 
    
    return masked_image


#%%



line = [2.01161605e-01, 2.44346845e+03]
roi = [285, 315]
    
def remove_elastic(filename, pixel_calibration, roi): 
    """takes a generated RIXS map and removes the elastic peak
    1: built model: background + gauss centered around the incident energy and away from the edge of interest
    2: remove the background
    3: combine all the removed images into the same RIXS map"""
    
    data_from_file = read_nxs_file(filename)
    
    pilatus_image = data_from_file['images']
    energy = data_from_file['energies']
    
    E2 = np.polyval(line, np.arange(0, 195))   # emission energy axis
    RIXS_map = np.sum(pilatus_image[:, :, roi[0]:roi[1]], axis=2)
    clean_RIXS = []
    
    
    for N,XES in enumerate(RIXS_map): 
        x_full = E2
        y_full = XES
        
        #three regions for background fit
        emin = min(x_full)
        emax = max(x_full)
        E_center = energy[N]
        
        mask1 = (x_full >= emin) & (x_full <= 2455)
        mask2 = (x_full >= 2470) & (x_full <= emax)
        mask3 = (x_full >= (E_center - 3)) & (x_full <= (E_center + 3))
        
        mask = mask1 | mask2 | mask3
        
        
        see_el = False
        x_fit = x_full[mask]
        y_fit = y_full[mask]
        
        background_order = 1
        back_model = PolynomialModel(degree=background_order, prefix='bkg_')
        
        if E_center < max(x_full): 
            el_model = LorentzianModel(prefix='g_')
            model = back_model + el_model
            see_el = True
            
        else: 
            model = back_model
            
            
        params = model.make_params()
                
        for i in range(background_order + 1):
            params[f'bkg_c{i}'].set(value=1)
        
        if see_el: 
            params['g_amplitude'].set(value=np.max(y_fit), min=0)
            params['g_center'].set(value=energy[N])
            params['g_sigma'].set(value = 1)
        
        result = model.fit(y_fit, params, x=x_fit)
        
        plot = True
        if plot: 
            
            fig, axs = plt.subplots(1,2, figsize=(10,6), squeeze=False)
            ax: matplotlib.axes.Axes
            ax = axs[0, 0]
            
            ax.scatter(x_fit,y_fit, color='black', label = 'data for fit', s = 4)
            ax.plot(x_full,y_full, color='black', label = 'data', alpha = 0.5)
            #ax.plot(x_fit, result.best_fit, color = 'red', label = 'fit')
            ax.plot(x_full, result.eval_components(x=x_full)['bkg_'], '--', label='Background')
            if see_el: 
                ax.plot(x_full, result.eval_components(x=x_full)['g_'], '--', label='Gaussian')
            ax.set_title(f'Fit to elastic peak')
            ax.set_xlabel('energy (px)')
            ax.set_ylabel(f'intesity')
            ax.grid(visible=True)
            ax.legend()
        
            ax: matplotlib.axes.Axes
            ax = axs[0, 1]
            
            ax.plot(x_full,y_full-result.eval(x=x_full), color='black', label = 'data', alpha = 0.5)
            ax.set_title(f'Removed background')
            ax.set_xlabel('energy (px)')
            ax.set_ylabel(f'intesity')
            ax.grid(visible=True)
            ax.legend()
    
        clean_RIXS.append(y_full-result.eval(x=x_full))
        
    clean_RIXS = np.array(clean_RIXS)
    #clean_RIXS[clean_RIXS < 100] = 0
    
    fig, axs = plt.subplots(1,3, figsize=(15,5), squeeze=False)
    ax: matplotlib.axes.Axes
    ax = axs[0, 0]
    ax.set_title('RXES map - ' + filename)
    im = ax.pcolormesh(E2, energy, clean_RIXS, shading='auto')
    ax.set_xlabel('Emitted Energy (eV)')
    ax.set_ylabel('Incident Energy (eV)')
    fig.colorbar(im, ax=ax)
    # plt.axis('square')
    ax.set_xlim(min(E2), max(E2))
    
    
    ax: matplotlib.axes.Axes
    ax = axs[0, 1]
    ax.set_title('XES projection - ' + filename)
    ax.plot(E2, np.sum(clean_RIXS, axis = 0))
    ax.set_xlabel('Emitted Energy (eV)')
    ax.set_ylabel('Counts')
    ax.set_xlim(min(E2), max(E2))
    
    ax: matplotlib.axes.Axes
    ax = axs[0, 2]
    ax.set_title('XAS projection - ' + filename)
    ax.plot(energy, np.sum(clean_RIXS, axis = 1))
    ax.set_xlabel('Incident energy (eV)')
    ax.set_ylabel('Counts')




def plot_Amptek(data, rois: list[list], plot=True):
    """
    This function plots the data from a .nxs file. Used for emission Amptek spectrometer.

    :param filename:  The path to the .nxs file relative to the path given in the DATA_PATH variable.
    :param rois: The regions of interest to plot, e.g. [[100, 200], [300, 400]] for two ROIs.
    :param plot: Whether to plot the data. If False, returns a ps.Series with the summed counts in the ROI and the dict of NXS data.
    :return: If only one ROI is given and plot is false, returns a pd.Series with the summed counts in the ROI and the dict of NXS data.
    """

    position = data['position']
    # DIODE = data_from_file['DIODE']
    # I01 = data_from_file['I01']
    # SDD = data_from_file['SDD']
    Amptek = data['Amptek']
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