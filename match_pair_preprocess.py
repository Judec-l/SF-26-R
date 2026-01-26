"""
match_pair_preprocess.py - Refactored
Preprocessing module for particle matching using spatial partitioning
"""
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

COMM = 0


def match_pair_preprocess(
        x1_unpaired: np.ndarray, y1_unpaired: np.ndarray, r1_unpaired: np.ndarray,
        x2_unpaired: np.ndarray, y2_unpaired: np.ndarray, r2_unpaired: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess particle positions to find nearest neighbors using spatial partitioning.

    Args:
        x1_unpaired, y1_unpaired, r1_unpaired: Coordinates of particles in frame 1
        x2_unpaired, y2_unpaired, r2_unpaired: Coordinates of particles in frame 2

    Returns:
        Tuple containing:
        - dist_min1a, dist_min1b: Distances to 3 nearest neighbors (shape N1 x 3)
        - dist_min2a, dist_min2b: Distances to 3 nearest neighbors (shape N2 x 3)
        - index1a, index1b: Indices of 3 nearest neighbors (shape N1 x 3)
        - index2a, index2b: Indices of 3 nearest neighbors (shape N2 x 3)
    """
    global COMM

    logger.info("Preprocessing particle positions...")
    COMM = 0
    END_LOOP = 0

    # Normalize arrays to 1D
    x1_unpaired = np.asarray(x1_unpaired, dtype=float).ravel()
    y1_unpaired = np.asarray(y1_unpaired, dtype=float).ravel()
    r1_unpaired = np.asarray(r1_unpaired, dtype=float).ravel()
    x2_unpaired = np.asarray(x2_unpaired, dtype=float).ravel()
    y2_unpaired = np.asarray(y2_unpaired, dtype=float).ravel()
    r2_unpaired = np.asarray(r2_unpaired, dtype=float).ravel()

    NT1 = len(x1_unpaired)
    NT2 = len(x2_unpaired)

    logger.info(f"Frame 1: {NT1} particles, Frame 2: {NT2} particles")

    # Initialize result arrays
    dist_min1a = np.zeros((NT1, 3))
    dist_min1b = np.zeros((NT1, 3))
    dist_min2a = np.zeros((NT2, 3))
    dist_min2b = np.zeros((NT2, 3))

    index1a = np.zeros((NT1, 3), dtype=int)
    index1b = np.zeros((NT1, 3), dtype=int)
    index2a = np.zeros((NT2, 3), dtype=int)
    index2b = np.zeros((NT2, 3), dtype=int)

    # Determine initial partition size
    if min(NT1, NT2) > 200:
        NP = int(np.floor((min(NT1, NT2) / 3) ** (1 / 3)))
    elif min(NT1, NT2) > 50:
        NP = 2
    else:
        NP = 1

    NR = 0

    # Adaptive partitioning loop
    while END_LOOP == 0 and NR < 100:
        NR += 1
        NP = max(NP - 2, 1)
        logger.info(f"Partition iteration {NR}, NP={NP}")

        COMM = 0

        # Create 3D spatial partition
        SP1: List[List[List[List[int]]]] = [
            [[[] for _ in range(NP)] for _ in range(NP)] for _ in range(NP)
        ]
        SP2: List[List[List[List[int]]]] = [
            [[[] for _ in range(NP)] for _ in range(NP)] for _ in range(NP)
        ]

        # Compute spatial bounds
        eps = 1e-7
        xmin = min(x1_unpaired.min(), x2_unpaired.min()) - eps
        xmax = max(x1_unpaired.max(), x2_unpaired.max()) + eps
        ymin = min(y1_unpaired.min(), y2_unpaired.min()) - eps
        ymax = max(y1_unpaired.max(), y2_unpaired.max()) + eps
        rmin = min(r1_unpaired.min(), r2_unpaired.min()) - eps
        rmax = max(r1_unpaired.max(), r2_unpaired.max()) + eps

        # Partition frame 1 particles
        for n in range(NT1):
            ix = int(np.floor((x1_unpaired[n] - xmin) / (xmax - xmin) * NP))
            iy = int(np.floor((y1_unpaired[n] - ymin) / (ymax - ymin) * NP))
            ir = int(np.floor((r1_unpaired[n] - rmin) / (rmax - rmin) * NP))
            ix = np.clip(ix, 0, NP - 1)
            iy = np.clip(iy, 0, NP - 1)
            ir = np.clip(ir, 0, NP - 1)
            SP1[ix][iy][ir].append(n)

        # Partition frame 2 particles
        for n in range(NT2):
            ix = int(np.floor((x2_unpaired[n] - xmin) / (xmax - xmin) * NP))
            iy = int(np.floor((y2_unpaired[n] - ymin) / (ymax - ymin) * NP))
            ir = int(np.floor((r2_unpaired[n] - rmin) / (rmax - rmin) * NP))
            ix = np.clip(ix, 0, NP - 1)
            iy = np.clip(iy, 0, NP - 1)
            ir = np.clip(ir, 0, NP - 1)
            SP2[ix][iy][ir].append(n)

        # Track ordering for later reordering
        N1_marker_old = 0
        N2_marker_old = 0
        aux_vector_order1 = np.zeros(NT1, dtype=int)
        aux_vector_order2 = np.zeros(NT2, dtype=int)

        # Process each partition cell
        for n1 in range(NP):
            for n2 in range(NP):
                for n3 in range(NP):
                    # Get neighboring cells (including current)
                    n1min, n1max = max(n1 - 1, 0), min(n1 + 1, NP - 1)
                    n2min, n2max = max(n2 - 1, 0), min(n2 + 1, NP - 1)
                    n3min, n3max = max(n3 - 1, 0), min(n3 + 1, NP - 1)

                    unpaired1_PI = []
                    unpaired2_PI = []

                    # Collect particles from neighboring cells
                    for i1 in range(n1min, n1max + 1):
                        for i2 in range(n2min, n2max + 1):
                            for i3 in range(n3min, n3max + 1):
                                # Mark where center cell starts
                                if i1 == n1 and i2 == n2 and i3 == n3:
                                    range1_min = len(unpaired1_PI)
                                    range2_min = len(unpaired2_PI)

                                unpaired1_PI.extend(SP1[i1][i2][i3])
                                unpaired2_PI.extend(SP2[i1][i2][i3])

                                # Mark where center cell ends
                                if i1 == n1 and i2 == n2 and i3 == n3:
                                    range1_max = len(unpaired1_PI)
                                    range2_max = len(unpaired2_PI)

                    rng = [range1_min, range1_max, range2_min, range2_max]

                    # Process this partition
                    result = match_pair_preprocess_aux(
                        x1_unpaired[unpaired1_PI],
                        x2_unpaired[unpaired2_PI],
                        y1_unpaired[unpaired1_PI],
                        y2_unpaired[unpaired2_PI],
                        r1_unpaired[unpaired1_PI],
                        r2_unpaired[unpaired2_PI],
                        unpaired1_PI,
                        unpaired2_PI,
                        rng
                    )

                    # Check if partition was too small
                    if COMM == 1:
                        break

                    d1a, d1b, d2a, d2b, i1a, i1b, i2a, i2b = result

                    # Store results
                    n1_new = N1_marker_old + (rng[1] - rng[0])
                    n2_new = N2_marker_old + (rng[3] - rng[2])

                    aux_vector_order1[N1_marker_old:n1_new] = unpaired1_PI[rng[0]:rng[1]]
                    aux_vector_order2[N2_marker_old:n2_new] = unpaired2_PI[rng[2]:rng[3]]

                    dist_min1a[N1_marker_old:n1_new, :] = d1a
                    dist_min1b[N1_marker_old:n1_new, :] = d1b
                    dist_min2a[N2_marker_old:n2_new, :] = d2a
                    dist_min2b[N2_marker_old:n2_new, :] = d2b

                    index1a[N1_marker_old:n1_new, :] = i1a
                    index1b[N1_marker_old:n1_new, :] = i1b
                    index2a[N2_marker_old:n2_new, :] = i2a
                    index2b[N2_marker_old:n2_new, :] = i2b

                    N1_marker_old = n1_new
                    N2_marker_old = n2_new

                if COMM == 1:
                    break
            if COMM == 1:
                break

        if COMM == 0:
            END_LOOP = 1

    # Reorder results to match original particle ordering
    I1 = np.argsort(aux_vector_order1)
    I2 = np.argsort(aux_vector_order2)

    logger.info("Preprocessing completed")

    return (
        dist_min1a[I1], dist_min1b[I1],
        dist_min2a[I2], dist_min2b[I2],
        index1a[I1], index1b[I1],
        index2a[I2], index2b[I2]
    )


def match_pair_preprocess_aux(
        x1p: np.ndarray, x2p: np.ndarray,
        y1p: np.ndarray, y2p: np.ndarray,
        r1p: np.ndarray, r2p: np.ndarray,
        unpaired1_PI: List[int], unpaired2_PI: List[int],
        rng: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single partition cell to find nearest neighbors.

    Args:
        x1p, y1p, r1p: Coordinates in partition (frame 1)
        x2p, y2p, r2p: Coordinates in partition (frame 2)
        unpaired1_PI: Original indices for frame 1 particles
        unpaired2_PI: Original indices for frame 2 particles
        rng: [range1_min, range1_max, range2_min, range2_max]

    Returns:
        Tuple of distance and index arrays
    """
    global COMM

    N1 = len(x1p)
    N2 = len(x2p)

    # Check if partition has enough particles
    if N1 <= 3 or N2 <= 3:
        COMM = 1
        return [], [], [], [], [], [], [], []

    N1s = rng[1] - rng[0]
    N2s = rng[3] - rng[2]

    # Initialize arrays with infinity
    inf = np.inf
    d1b = np.full((N1s, 3), inf)
    i1b = np.zeros((N1s, 3), dtype=int)
    d1a = np.full((N1s, 3), inf)
    i1a = np.zeros((N1s, 3), dtype=int)

    d2a = np.full((N2s, 3), inf)
    i2a = np.zeros((N2s, 3), dtype=int)
    d2b = np.full((N2s, 3), inf)
    i2b = np.zeros((N2s, 3), dtype=int)

    # Find nearest neighbors for frame 1 center particles
    for n in range(N1s):
        pI = rng[0] + n
        p = np.array([x1p[pI], y1p[pI], r1p[pI]])

        # Find neighbors in frame 2 (cross-frame)
        for m in range(N2):
            q = np.array([x2p[m], y2p[m], r2p[m]])
            dist = np.linalg.norm(p - q)
            _update_top3(d1b[n], i1b[n], dist, unpaired2_PI[m])

        # Find neighbors in frame 1 (same frame)
        for n1 in range(N1):
            if n1 == pI:
                continue
            q = np.array([x1p[n1], y1p[n1], r1p[n1]])
            dist = np.linalg.norm(p - q)
            _update_top3(d1a[n], i1a[n], dist, unpaired1_PI[n1])

    # Find nearest neighbors for frame 2 center particles
    for m in range(N2s):
        qI = rng[2] + m
        q = np.array([x2p[qI], y2p[qI], r2p[qI]])

        # Find neighbors in frame 1 (cross-frame)
        for n in range(N1):
            p = np.array([x1p[n], y1p[n], r1p[n]])
            dist = np.linalg.norm(p - q)
            _update_top3(d2a[m], i2a[m], dist, unpaired1_PI[n])

        # Find neighbors in frame 2 (same frame)
        for m1 in range(N2):
            if m1 == qI:
                continue
            p = np.array([x2p[m1], y2p[m1], r2p[m1]])
            dist = np.linalg.norm(p - q)
            _update_top3(d2b[m], i2b[m], dist, unpaired2_PI[m1])

    return d1a, d1b, d2a, d2b, i1a, i1b, i2a, i2b


def _update_top3(dist_arr: np.ndarray, idx_arr: np.ndarray,
                 dist: float, idx: int) -> None:
    """
    Update top 3 nearest neighbors in place.

    Args:
        dist_arr: Array of 3 distances (modified in place)
        idx_arr: Array of 3 indices (modified in place)
        dist: New distance to consider
        idx: New index to consider
    """
    if dist < dist_arr[0]:
        dist_arr[2], idx_arr[2] = dist_arr[1], idx_arr[1]
        dist_arr[1], idx_arr[1] = dist_arr[0], idx_arr[0]
        dist_arr[0], idx_arr[0] = dist, idx
    elif dist < dist_arr[1]:
        dist_arr[2], idx_arr[2] = dist_arr[1], idx_arr[1]
        dist_arr[1], idx_arr[1] = dist, idx
    elif dist < dist_arr[2]:
        dist_arr[2], idx_arr[2] = dist, idx