import numpy as np

def crack_Link2FrameId(PrLink, link_temp, FrameId):
    final_link = np.where(np.array(link_temp) >= FrameId)[0]

    # MATLAB: final_pair_cell = PrLink(final_link);
    final_pair_array = []

    for idx in final_link:
        # MATLAB: final_pair_cell{i}(1:FrameId)
        pair = PrLink[idx]
        final_pair_array.append(pair[:FrameId])

    # Convert to NumPy array
    final_pair_array = np.vstack(final_pair_array)

    firstFrmId = final_pair_array[:, 0]
    lastFrmId = final_pair_array[:, -1]

    return firstFrmId, lastFrmId
