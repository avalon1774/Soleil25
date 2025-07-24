
From Sample import BaseScan, Sample,
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Dict, Optional, Any
Import cache_utils

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


@dataclass

class ComparisonSet:
    label: str
    scans: List[BaseScan]
    kind: str #amptek xas, pilatus xas, xes at certain energy
    roi: Tuple[int, int] #for pilatus, for amptek roi will be kept kind consistent if no other reason
    absorption_energy: Optional[float] = None #for xes comparison


def load_amptek_xas(scan: XASScan) -> pd.DataFrame:
    df = pd.read_csv(get_spectrum_path(scan, roi, kind='pilatus_xas'))
