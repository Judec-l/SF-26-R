import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

Rs = None
Rn = None
Rq = None
A = None
B = None
C = None
NR = None


def setParameter(parameter: Optional[Dict[str, Any]], field: str, default: Any) -> Any:
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
    global Rs, Rn, Rq, A, B, C, NR

    drift = []
    info = []
    debug_info = []

    Rs = 1.5 * max_dis
    Rn = Rs
    Rq = 0.2 * Rs
    NR = 30  # Increased iterations for better convergence
    A = 0.1  # Reduced to give more weight to neighbor consistency
    B = 5.0  # Increased to emphasize neighbor information
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

    # Find bidirectional consistent matches with RELAXED thresholds
    I1_list = []
    I2_list = []

    for n in range(N1):
        m = I1_temp[n]
        if m < 0:
            continue

        # Check bidirectional consistency
        if I2_temp[m] == n:
            # RELAXED confidence thresholds - much more lenient
            acceptable_confidence = (
                    confidence1[n] > 0.3 or  # Lowered from 0.9
                    confidence2[m] > 0.3 or  # Lowered from 0.9
                    (confidence1[n] > 0.1 and confidence2[m] > 0.1)  # Lowered from 0.5
            )

            if acceptable_confidence:
                # Verify distance constraint (2D only, ignore z)
                dx = x2[m] - x1[n]
                dy = y2[m] - y1[n]
                dist_2d = np.sqrt(dx * dx + dy * dy)

                if dist_2d < max_dis:
                    I1_list.append(n)
                    I2_list.append(m)

    # Convert to arrays
    I1 = np.array(I1_list, dtype=int)
    I2 = np.array(I2_list, dtype=int)

    logger.info(f"Found {len(I1)} bidirectional matches ({100.0 * len(I1) / N1:.1f}% of frame 1 particles)")

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
    global Rs, Rn, Rq, A, B, C, NR

    N1 = len(x1)
    N2 = len(x2)
    I_temp = np.full(N1, -1, dtype=int)
    confidence = np.zeros(N1)

    # Build distance matrix for all particle pairs within search radius
    # Use 2D distance only (ignore r/z coordinate which is always 0)
    dist_matrix = np.full((N1, N2), np.inf)

    logger.info(f"Building distance matrix with search radius Rs={Rs:.1f}")

    for n in range(N1):
        dx = x2 - x1[n]
        dy = y2 - y1[n]
        dists = np.sqrt(dx ** 2 + dy ** 2)

        # Only consider particles within search radius
        within_radius = dists < Rs
        dist_matrix[n, within_radius] = dists[within_radius]

    # Count candidates per particle
    n_candidates = np.sum(~np.isinf(dist_matrix), axis=1)
    logger.info(
        f"Candidates per particle: mean={np.mean(n_candidates):.1f}, min={np.min(n_candidates)}, max={np.max(n_candidates)}")

    # Initialize probability matrix
    Pb = np.zeros((N1, N2))

    for n in range(N1):
        valid_targets = ~np.isinf(dist_matrix[n, :])
        n_valid = np.sum(valid_targets)

        if n_valid > 0:
            # Initialize with distance-based weights (closer = higher weight)
            dists = dist_matrix[n, valid_targets]
            weights = np.exp(-dists / (Rs / 3.0))  # Gaussian-like weighting
            Pb[n, valid_targets] = weights / weights.sum()

    # Relaxation iterations
    for iteration in range(NR):
        Pb_new = Pb.copy()

        for n in range(N1):
            valid_targets = ~np.isinf(dist_matrix[n, :])

            if not np.any(valid_targets):
                continue

            # Get candidate matches and their probabilities
            cand_indices = np.where(valid_targets)[0]

            # Compute scores based on distances and neighbor consistency
            scores = np.zeros(len(cand_indices))

            for i, m in enumerate(cand_indices):
                # Distance-based score
                dist = dist_matrix[n, m]
                dist_score = np.exp(-dist / (Rs / 3.0))

                # Neighbor consistency: check if neighbors of n map to neighbors of m
                neighbor_score = 0.0
                neighbors_n = indexa[n]  # Neighbors of particle n in frame 1 (same frame)

                count_neighbors = 0
                for nn in neighbors_n:
                    if nn >= 0 and nn < N1:
                        # Check if this neighbor has high probability of matching near m
                        # Look at particles near m in frame 2
                        for mm in range(max(0, m - 15), min(N2, m + 15)):
                            if not np.isinf(dist_matrix[nn, mm]):
                                neighbor_score += Pb[nn, mm]
                                count_neighbors += 1

                if count_neighbors > 0:
                    neighbor_score = neighbor_score / count_neighbors

                # Combined score
                scores[i] = dist_score * (A + B * neighbor_score)

            # Update probabilities
            if scores.sum() > 0:
                Pb_new[n, cand_indices] = scores / scores.sum()
            else:
                Pb_new[n, cand_indices] = Pb[n, cand_indices]

        Pb = Pb_new

    # Extract final matches with VERY low threshold
    for n in range(N1):
        valid_targets = ~np.isinf(dist_matrix[n, :])

        if np.any(valid_targets):
            # Find best match
            best_m = np.argmax(Pb[n, :])
            best_prob = Pb[n, best_m]

            # Accept if probability is above minimal threshold and within distance
            if best_prob > 0.01 and not np.isinf(dist_matrix[n, best_m]):  # Very low threshold!
                I_temp[n] = best_m
                confidence[n] = best_prob

    n_matched = np.sum(I_temp >= 0)
    logger.info(f"Relaxation complete: {n_matched}/{N1} particles matched ({100.0 * n_matched / N1:.1f}%)")

    return I_temp, confidence