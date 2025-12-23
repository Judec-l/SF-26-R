import numpy as np

def LinkPairsF2L(cellPairs):
    """
    This function links bead pairs from the first frame to the last frame.

    Parameters
    ----------
    cellPairs : list of numpy.ndarray
        Length = (number of frames - 1)
        Each element is an array of shape (N, 2) containing bead pair labels
        between two adjacent frames.

    Returns
    -------
    PairResult : list
        List of linked bead chains (each chain is a list)
    breakChains : list
        Indices (0-based) of chains that break before the last frame
    """

    # Initialize
    LENinp = cellPairs[0].shape[0]
    TolPairs = len(cellPairs)

    PairResult = [None] * LENinp
    breakChains = []

    for i in range(LENinp):
        # MATLAB: PairResult{i} = cellPairs{1}(i, 1:2);
        PairResult[i] = list(cellPairs[0][i, :2])
        temp2ed = PairResult[i][1]

        # Pframe is the index of frame-pair
        Pframe = 0  # MATLAB starts from 1

        while True:
            Pframe += 1

            # If touches the last frame, exit loop
            if Pframe >= TolPairs:
                break

            # MATLAB: index2_temp = find(cellPairs{Pframe}(:,1) == temp2ed);
            matches = np.where(cellPairs[Pframe][:, 0] == temp2ed)[0]

            if len(matches) == 1:
                temp2ed = cellPairs[Pframe][matches[0], 1]
                PairResult[i].append(temp2ed)

            elif len(matches) == 0:
                breakChains.append(i)
                break

            else:
                print("Multiple beads! Check immediately!")
                break

    print("Linking Mission Complete!")

    return PairResult, breakChains
