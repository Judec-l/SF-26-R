"""
match_pairRelax4.py - Refactored
Relaxation-based particle matching algorithm
"""
import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Global parameters (maintained for compatibility)
Rs = None
Rn = None
Rq = None
A = None
B = None
C = None
NR = None


def setParameter(parameter: Optional[Dict[str, Any]], field: str, default: Any) -> Any:
    """
    Safely retrieve parameter from dictionary with default value.

    Args:
        parameter: Parameter dictionary
        field: Key to retrieve
        default: Default value if key not found

    Returns:
        Parameter value or default
    """
    if parameter is None:
        return default
    return parameter.get(field, default)


def match_pairRelax4(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
        r1: np.ndarray, r2: np.ndarray, max_dis: float,
        option: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, list, list, list]:
    """
    Match particles between two frames using relaxation algorithm.

    Args:
        x1, y1: X and Y coordinates of particles in frame 1
        x2, y2: X and Y coordinates of particles in frame 2
        r1, r2: Radius/Z coordinates of particles
        max_dis: Maximum displacement threshold
        option: Optional configuration dictionary

    Returns:
        Tuple containing:
        - I1: Matched indices in frame 1
        - I2: Matched indices in frame 2
        - I1u: Unmatched indices in frame 1
        - I2u: Unmatched indices in frame 2
        - dx, dy, dr: Displacement components
        - confidence1, confidence2: Confidence scores
        - drift, info, debug_info: Additional outputs (empty for compatibility)
    """
    global Rs, Rn, Rq, A, B, C, NR

    # Initialize empty return values for compatibility
    drift = []
    info = []
    debug_info = []

    # Set global parameters
    Rs = 1.5 * max_dis
    Rn = Rs
    Rq = 0.2 * Rs
    NR = 20
    A = 0.3
    B = 3.0
    C = 0.0

    # Get options
    if option is None:
        option = {}

    savefileStr = setParameter(option, "savefileStr", "")
    resfolder = setParameter(option, "resfolder", "../PTLibrary2D/res_matchpair/")
    auxfolder = setParameter(option, "auxfolder", "../PTLibrary2D/aux_matchpair/")
    auxfileSave = setParameter(option, "AuxFileSave", "NO")

    # Create directories
    os.makedirs(resfolder, exist_ok=True)
    os.makedirs(auxfolder, exist_ok=True)

    # Set filenames
    if savefileStr == "":
        preprocessFn = os.path.join(auxfolder, "matchpair_preproc.pkl")
        filename_result = os.path.join(resfolder, "matchpairRelax_res.pkl")
    else:
        preprocessFn = os.path.join(auxfolder, f"matchpair_preproc_{savefileStr}.pkl")
        filename_result = os.path.join(resfolder, f"matchpairRelax_res_{savefileStr}.pkl")

    # Normalize arrays to 1D
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    y1 = np.asarray(y1, dtype=float).reshape(-1)
    r1 = np.asarray(r1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    y2 = np.asarray(y2, dtype=float).reshape(-1)
    r2 = np.asarray(r2, dtype=float).reshape(-1)

    N1 = len(x1)
    N2 = len(x2)

    logger.info(f"Matching {N1} particles in frame 1 with {N2} particles in frame 2")

    # Load or compute preprocessing
    if not os.path.exists(preprocessFn):
        logger.info("Calculating preprocessing data stage 1...")
        from match_pair_preprocess import match_pair_preprocess
        data = match_pair_preprocess(x1, y1, r1, x2, y2, r2)
        with open(preprocessFn, "wb") as f:
            pickle.dump(data, f)
    else:
        logger.info("Loading preprocessing data stage 1...")
        with open(preprocessFn, "rb") as f:
            data = pickle.load(f)

    (dist_min1a, dist_min1b,
     dist_min2a, dist_min2b,
     index1a, index1b,
     index2a, index2b) = data

    # Match forward: frame 1 -> frame 2
    option_a = option.copy()
    option_a["str"] = "a"
    I1_temp, confidence1 = match_pairRelax_aux(
        x1, y1, x2, y2, r1, r2, index1a, index1b, option_a
    )

    # Match backward: frame 2 -> frame 1
    option_b = option.copy()
    option_b["str"] = "b"
    I2_temp, confidence2 = match_pairRelax_aux(
        x2, y2, x1, y1, r2, r1, index2b, index2a, option_b
    )

    # Find bidirectional consistent matches
    I1_list = []
    I2_list = []

    for n in range(N1):
        m = I1_temp[n]
        if m < 0:
            continue

        # Check bidirectional consistency
        if I2_temp[m] == n:
            # Check confidence thresholds
            high_confidence = (confidence1[n] > 0.9 or confidence2[m] > 0.9)
            medium_confidence = (confidence1[n] > 0.5 and confidence2[m] > 0.5)

            if high_confidence or medium_confidence:
                # Verify distance constraint
                p1 = np.array([x1[n], y1[n], r1[n]])
                p2 = np.array([x2[m], y2[m], r2[m]])

                if np.linalg.norm(p1 - p2) < max_dis:
                    I1_list.append(n)
                    I2_list.append(m)

    # Convert to arrays
    I1 = np.array(I1_list, dtype=int)
    I2 = np.array(I2_list, dtype=int)

    logger.info(f"Found {len(I1)} bidirectional matches")

    # Compute displacements
    if len(I1) > 0:
        dx = x2[I2] - x1[I1]
        dy = y2[I2] - y1[I1]
        dr = r2[I2] - r1[I1]
    else:
        dx = np.array([])
        dy = np.array([])
        dr = np.array([])

    # Find unmatched particles
    I1u = np.setdiff1d(np.arange(N1), I1)
    I2u = np.setdiff1d(np.arange(N2), I2)

    logger.info(f"Unmatched: {len(I1u)} in frame 1, {len(I2u)} in frame 2")

    # Save results
    with open(filename_result, "wb") as f:
        pickle.dump(
            (I1, I2, I1u, I2u, dx, dy, dr, confidence1, confidence2),
            f
        )

    return I1, I2, I1u, I2u, dx, dy, dr, confidence1, confidence2, drift, info, debug_info


def match_pairRelax_aux(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
        r1: np.ndarray, r2: np.ndarray,
        indexa: np.ndarray, indexb: np.ndarray,
        option: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary relaxation matching function.

    Args:
        x1, y1, r1: Coordinates of particles in frame 1
        x2, y2, r2: Coordinates of particles in frame 2
        indexa: Indices of nearest neighbors in same frame
        indexb: Indices of nearest neighbors in other frame
        option: Options dictionary

    Returns:
        Tuple of (match_indices, confidence_scores)
    """
    global Rs, Rn, Rq, A, B, C, NR

    N1 = len(x1)
    I_temp = np.full(N1, -1, dtype=int)
    confidence = np.zeros(N1)

    # Initialize probability distributions and neighbor lists
    Pb = [None] * N1
    Nis = [None] * N1

    for n in range(N1):
        # Get valid candidates (filter out invalid indices)
        candidates = indexb[n]
        valid_mask = (candidates >= 0) & (candidates < len(x2))
        valid_candidates = candidates[valid_mask]

        if len(valid_candidates) > 0:
            Pb[n] = np.ones(len(valid_candidates)) / len(valid_candidates)
            Nis[n] = valid_candidates
        else:
            Pb[n] = np.array([])
            Nis[n] = np.array([])

    # Relaxation iterations
    for iteration in range(NR):
        Pb_new = []

        for n in range(N1):
            p = np.array([x1[n], y1[n], r1[n]])
            probs = Pb[n]
            cands = Nis[n]

            if len(cands) == 0 or len(probs) == 0:
                Pb_new.append(np.array([]))
                continue

            # Compute consistency scores
            scores = np.zeros(len(cands))

            for i, m in enumerate(cands):
                q = np.array([x2[m], y2[m], r2[m]])
                dist = np.linalg.norm(p - q)

                # Particles within search radius support this match
                if dist < Rs:
                    scores[i] += probs[i]

            # Update probabilities using relaxation formula
            if scores.sum() > 0:
                probs = probs * (A + B * scores)
                probs = probs / probs.sum()

            Pb_new.append(probs)

        Pb = Pb_new

    # Extract best matches
    for n in range(N1):
        probs = Pb[n]
        if len(probs) > 0:
            idx = np.argmax(probs)
            I_temp[n] = Nis[n][idx]
            confidence[n] = probs[idx]

    return I_temp, confidence