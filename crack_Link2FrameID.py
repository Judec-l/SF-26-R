import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def crack_Link2FrameId(
        PrLink: List[List[int]],
        link_temp: List[int],
        FrameId: int
) -> Tuple[np.ndarray, np.ndarray]:
    final_link = np.where(np.array(link_temp) >= FrameId)[0]

    logger.info(f"Filtering {len(PrLink)} trajectories by length >= {FrameId}")
    logger.info(f"Found {len(final_link)} valid trajectories")

    final_pair_array = []

    for idx in final_link:
        pair = PrLink[idx]
        final_pair_array.append(pair[:FrameId])

    if not final_pair_array:
        logger.warning("No trajectories meet the length requirement")
        return np.array([]), np.array([])

    final_pair_array = np.array(final_pair_array)

    firstFrmId = final_pair_array[:, 0]
    lastFrmId = final_pair_array[:, -1]

    logger.info(f"Extracted {len(firstFrmId)} valid trajectory endpoints")

    return firstFrmId, lastFrmId