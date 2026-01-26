import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def LinkPairsF2L(cellPairs: List[np.ndarray]) -> Tuple[List[List[int]], List[int]]:
    if not cellPairs or len(cellPairs) == 0:
        logger.warning("No pairs to link")
        return [], []

    LENinp = cellPairs[0].shape[0]
    TolPairs = len(cellPairs)

    logger.info(f"Linking {LENinp} initial pairs across {TolPairs} frame pairs")

    PairResult: List[List[int]] = []
    breakChains: List[int] = []

    for i in range(LENinp):
        trajectory = [int(cellPairs[0][i, 0]), int(cellPairs[0][i, 1])]
        PairResult.append(trajectory)

        temp2ed = trajectory[-1]

        Pframe = 0

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