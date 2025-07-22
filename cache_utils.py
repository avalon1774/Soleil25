import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union
import logging
__cache_utils__ = "cache_utils"

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__cache_utils__)

CACHE_DIR = Path("cache")


def save_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        logger.info("pickling into %s", path)
        pickle.dump(data, f)

def load_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        print('help')
        return None
    with path.open("rb") as f:
        logger.info("retrieving pickle from %s", path)
        return pickle.load(f)

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        logger.info("saving json into %s", path)
        json.dump(data, f, indent=2)

def load_json(path: Path)   -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r") as f:
        logger.info("retrieving json from %s", path)
        return json.load(f)


def get_sample_dir(sample: 'Sample') -> Path:
    sample_id = f"Electrode_{sample.electrode_id:03d}"
    return CACHE_DIR/f"{sample_id}"

def get_scan_dir(scan: 'BaseScan') -> Path:
    sample_id = get_sample_dir(scan.sample)
    return sample_id/f"scan_{scan.number:04d}"

def get_hash_from_params(params: Dict) -> str:
    """if i want to tag a particular saved cache """
    string = json.dumps(params, sort_keys=True)
    return hashlib.md5(string.encode()).hexdigest()[:8]


def get_calibration_path(scan: 'BaseScan', roi: Tuple[int, int], kind: str = "xas", params: Dict = None) -> Path:
    roi_str = f"{roi[0]}_{roi[1]}"
    suffix = ""
    if params:
        suffix = f"_{get_hash_from_parameters(params)}"
    return get_scan_dir(scan) / f"calibration_{roi_str}{suffix}.pkl"


# def get_roi_path(scan: 'BaseScan') -> Path:
#     #rois all all saved in a .json file in the scan directory
#     return get_scan_dir(scan) / "roi.json"

def get_spectrum_path(scan: 'BaseScan', roi: Tuple[int,int], kind:str = "xas", params: Dict = None) -> Path:
    roi_str = f"{roi[0]}_{roi[1]}"
    suffix = ""
    if params:
        suffix = f"_{get_hash_from_params(params)}"
    return get_scan_dir(scan) / f"spectrum_{roi_str}_{kind}{suffix}.npy"


def clean_scan_cache(sample: 'BaseScan') -> None:
    """Remove all cached data for a scan."""
    scan_dir = get_scan_dir(sample)
    if scan_dir.exists():
        logger.info("Cleaning cache for scan %s", sample.number)
        for item in scan_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        scan_dir.rmdir()  # Remove the scan directory itself
    else:
        logger.warning("Scan directory %s does not exist", scan_dir)

def clean_sample_cache(sample: 'Sample') -> None:
    """Remove all cached data for a sample."""
    sample_dir = get_sample_dir(sample)
    if sample_dir.exists():
        logger.info("Cleaning cache for sample %s", sample.electrode_id)
        for scan_dir in sample_dir.glob("scan_*"):
            if scan_dir.is_dir():
                for f in scan_dir.glob("*"):
                    f.unlink()
                scan_dir.rmdir()
        sample_dir.rmdir() # Remove the sample directory itself
    else:
        logger.warning("Sample directory %s does not exist", sample_dir)