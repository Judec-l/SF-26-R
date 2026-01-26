"""
match_pair.py - Refactored
Main interface for particle matching with file I/O support
"""
import os
import numpy as np
from scipy.io import loadmat, savemat
from typing import Dict, Any, Tuple, Optional
import logging

from match_pair_preprocess import match_pair_preprocess
from match_pairRelax4 import match_pairRelax4

logger = logging.getLogger(__name__)


def setParameter(option: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if option is None:
        return default
    return option.get(key, default)


def match_pair(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
        r1: np.ndarray, r2: np.ndarray, max_dis: float,
        option: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
list, list, list]:
    if len(x1) < 20 or len(x2) < 20:
        logger.warning("Too few particles for matching")
        return (
            0, [], [], [], [], [], [], [],
            [], [], [], [], []
        )

    logger.info(f"Current directory: {os.getcwd()}")
    logger.info("Initializing matching algorithm...")

    N1 = len(x1)
    N2 = len(x2)
    logger.info(f"Frame 1: {N1} particles")
    logger.info(f"Frame 2: {N2} particles")
    logger.info(f"Max displacement: {max_dis}")

    if option is None:
        option = {}

    # Check required directories exist
    if not os.path.exists("../PTLibrary2D/aux_matchpair"):
        raise FileNotFoundError(
            "Directory not found: ../PTLibrary2D/aux_matchpair"
        )

    if not os.path.exists("../PTLibrary2D/res_matchpair"):
        raise FileNotFoundError(
            "Directory not found: ../PTLibrary2D/res_matchpair"
        )

    # Get paths
    resfolder = setParameter(option, "resfolder", "../PTLibrary2D/res_matchpair/")
    auxfolder = setParameter(option, "auxfolder", "../PTLibrary2D/aux_matchpair/")
    savefileStr_temp = setParameter(option, "savefileStr", "")
    loadfileStr_temp = setParameter(option, "loadfileStr", "")

    logger.info(f"Data identifier: {savefileStr_temp}")

    # Set filenames for loading results
    if loadfileStr_temp == "":
        loadfilename = os.path.join(resfolder, "matchpairRes.mat")
    else:
        loadfilename = os.path.join(resfolder, f"matchpairRes_{loadfileStr_temp}.mat")

    # Set filename for preprocessing
    if savefileStr_temp == "":
        preprocessFn = os.path.join(auxfolder, "matchpair_preproc.mat")
    else:
        preprocessFn = os.path.join(auxfolder, f"matchpair_preproc_{savefileStr_temp}.mat")

    # Load or compute preprocessing
    if os.path.exists(preprocessFn):
        logger.info("Loading preprocessing data from MATLAB file...")
        data = loadmat(preprocessFn)
        dist_min1a = data["dist_min1a"].squeeze()
        dist_min1b = data["dist_min1b"].squeeze()
        dist_min2a = data["dist_min2a"].squeeze()
        dist_min2b = data["dist_min2b"].squeeze()
    else:
        logger.info("Calculating preprocessing data...")
        (
            dist_min1a, dist_min1b,
            dist_min2a, dist_min2b,
            index1a, index1b,
            index2a, index2b
        ) = match_pair_preprocess(x1, y1, r1, x2, y2, r2)

        logger.info("Saving preprocessing data to MATLAB file...")
        savemat(
            preprocessFn,
            {
                "dist_min1a": dist_min1a,
                "dist_min1b": dist_min1b,
                "dist_min2a": dist_min2a,
                "dist_min2b": dist_min2b,
                "index1a": index1a,
                "index1b": index1b,
                "index2a": index2a,
                "index2b": index2b,
            }
        )

    # Compute mean distances
    meanDist = {
        "Dist1a": np.mean(dist_min1a),
        "Dist2b": np.mean(dist_min2b),
        "Dist1b": np.mean(dist_min1b),
        "Dist2a": np.mean(dist_min2a),
    }

    logger.info(f"Mean distances: {meanDist}")

    # Try to load existing results
    if option.get("loadRes", "").upper() == "YES" and os.path.exists(loadfilename):
        logger.info("Loading existing matching result from MATLAB file...")
        logger.info(f"Loading from: {loadfilename}")
        data = loadmat(loadfilename)

        I1 = data["I1"].squeeze()
        I2 = data["I2"].squeeze()
        I1u = data["I1u"].squeeze()
        I2u = data["I2u"].squeeze()
        dx = data["dx"].squeeze()
        dy = data["dy"].squeeze()
        dr = data["dr"].squeeze()
        confidence1 = data["confidence1"].squeeze()
        confidence2 = data["confidence2"].squeeze()
        drift = data.get("drift", [])
        info = data.get("info", [])
        debug_info = data.get("debug_info", [])

    else:
        # Perform matching using relaxation algorithm
        logger.info("Performing particle matching...")
        (
            I1, I2, I1u, I2u,
            dx, dy, dr,
            confidence1, confidence2,
            drift, info, debug_info
        ) = match_pairRelax4(
            x1, y1, x2, y2,
            r1, r2, max_dis,
            option
        )

        # Save results if requested
        if option.get("ResFileSave", "").upper() == "YES":
            if savefileStr_temp == "":
                savefilename = os.path.join(resfolder, "matchpairRes.mat")
            else:
                savefilename = os.path.join(
                    resfolder, f"matchpairRes_{savefileStr_temp}.mat"
                )

            logger.info(f"Saving results to MATLAB file: {savefilename}")
            savemat(
                savefilename,
                {
                    "I1": I1,
                    "I2": I2,
                    "I1u": I1u,
                    "I2u": I2u,
                    "dx": dx,
                    "dy": dy,
                    "dr": dr,
                    "confidence1": confidence1,
                    "confidence2": confidence2,
                    "drift": drift,
                    "info": info,
                    "debug_info": debug_info,
                }
            )

    return (
        meanDist, I1, I2, I1u, I2u,
        dx, dy, dr,
        confidence1, confidence2,
        drift, info, debug_info
    )