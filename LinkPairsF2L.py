"""
Linking utilities - Refactored
Functions for linking particle pairs across multiple frames
"""
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def LinkPairsF2L(cellPairs: List[np.ndarray]) -> Tuple[List[List[int]], List[int]]:
    """
    Link particle pairs across frames to create trajectories.

    Args:
        cellPairs: List of arrays, each with shape (n_pairs, 2) containing
                   matched indices (index_frame_i, index_frame_i+1)

    Returns:
        Tuple of (PairResult, breakChains)
        - PairResult: List of particle index lists forming continuous tracks
        - breakChains: List of trajectory indices where chains broke
    """
    if not cellPairs or len(cellPairs) == 0:
        logger.warning("No pairs to link")
        return [], []

    LENinp = cellPairs[0].shape[0]
    TolPairs = len(cellPairs)

    logger.info(f"Linking {LENinp} initial pairs across {TolPairs} frame pairs")

    PairResult: List[List[int]] = []
    breakChains: List[int] = []

    # Initialize trajectories from first frame pairs
    for i in range(LENinp):
        # Start trajectory with first two indices
        trajectory = [int(cellPairs[0][i, 0]), int(cellPairs[0][i, 1])]
        PairResult.append(trajectory)

        temp2ed = trajectory[-1]  # Last particle in trajectory

        Pframe = 0

        # Extend trajectory through subsequent frames
        while True:
            Pframe += 1

            # Check if we've processed all frame pairs
            if Pframe >= TolPairs:
                break

            # Find current particle in next frame pairs
            matches = np.where(cellPairs[Pframe][:, 0] == temp2ed)[0]

            if len(matches) == 1:
                # Unique match found - extend trajectory
                temp2ed = int(cellPairs[Pframe][matches[0], 1])
                PairResult[i].append(temp2ed)

            elif len(matches) == 0:
                # No match found - trajectory breaks
                breakChains.append(i)
                logger.debug(f"Trajectory {i} broke at frame {Pframe + 1}")
                break

            else:
                # Multiple matches - ambiguous
                logger.warning(f"Multiple beads matched for trajectory {i} at frame {Pframe + 1}")
                breakChains.append(i)
                break

    logger.info("Linking mission complete!")
    logger.info(f"Created {len(PairResult)} trajectories")
    logger.info(f"Broken chains: {len(breakChains)}")

    return PairResult, breakChains