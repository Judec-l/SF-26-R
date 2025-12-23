import numpy as np

def crack_Link2FrameId(PrLink, link_temp, FrameId):
    """
    This function links the pairs from the 1st frame to a certain frame (FrameId)

    Parameters
    ----------
    PrLink : list
        List of arrays/lists containing link pairs
    link_temp : array-like
        Array indicating the last frame index for each link
    FrameId : int
        Target frame ID

    Returns
    -------
    firstFrmId : numpy.ndarray
        IDs from the first frame
    lastFrmId : numpy.ndarray
        IDs from the FrameId frame
    """

    # MATLAB: final_link = find(link_temp >= FrameId);
    final_link = np.where(np.array(link_temp) >= FrameId)[0]

    # MATLAB: final_pair_cell = PrLink(final_link);
    final_pair_array = []

    for idx in final_link:
        # MATLAB: final_pair_cell{i}(1:FrameId)
        pair = PrLink[idx]
        final_pair_array.append(pair[:FrameId])

    # Convert to NumPy array
    final_pair_array = np.vstack(final_pair_array)

    # MATLAB: firstFrmId = final_pair_array(:, 1);
    firstFrmId = final_pair_array[:, 0]

    # MATLAB: lastFrmId = final_pair_array(:, end);
    lastFrmId = final_pair_array[:, -1]

    return firstFrmId, lastFrmId
