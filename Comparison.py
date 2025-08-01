
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
import logging
import warnings
__Analysis__ = "Analysis"

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__Analysis__)


def load_scans (sample_number: int,
                scans = dict['str', list],
                roi_id: Optional[int] = " ",
                kind: str = 'Amptek',
                energy: Optional[float] = None):
    """ load specified scans from the h5 files that combine all scans of a sample. return df with all the data interpolated onto the same energy axis if needed"""



    path = f"Sample_{sample_number:04d}.h5"
    if not Path(path).exists():
        logger.error(f"Sample {sample_number:04d} does not exist")
        continue

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
                df = pd.DataFrame(data, columns=["incident_energy", "intensity"])


            df["sample"] = sample_name
            df["scan_number"] = scan_number
            df["roi_id"] = roi_id

    return pd.concat(df, ignore_index = True)


samples = [1, 6, 7]
scans = {'1': [8],
         '6': [12],
         '7':[6]}

df = load_scans([1,6,7], scans)
fig, ax = plt.subplots(1,1, figsize=(16, 7))
axin1 = ax.inset_axes([0.01, 0.4, 0.3, 0.6])

for sample in samples:
    df = load_scans(sample, scans)
    df.plot(ax = ax, x="incident_energy", y="intensity", label=f"Sample {sample} ({sample.items()}", alpha=0.5)
    data = df["intensity"].to_numpy()
    axin1.plot(d.index, np.gradient(data), label=d.name,)

for sample_name in scans:
    group = df[df["sample"] == sample_name]
    if not group.empty:
        ax.plot(group["incident_energy"], group["intensity"], label=sample_name)

ax.set_xlabel("Incident Energy (eV)")
ax.set_ylabel("Intensity (counts)")
ax.set_title(f"Amptek XAS")
ax.legend()
ax.grid(True,alpha=0.3)

axin1 = ax.inset_axes([0.01, 0.4, 0.3, 0.6])
for index, d in enumerate(dfs):
    data = d.to_numpy()
    axin1.plot(d.index, np.gradient(data), label=d.name, color=cmap(index * 5))
axin1.grid(True, 'major')
axin1.xaxis.set_major_locator(MultipleLocator(2))
axin1.xaxis.set_minor_locator(MultipleLocator(1))
axin1.set_yticklabels([])
axin1.set_xlim(2466, 2476)
axin1.set_title("Derivatives")


