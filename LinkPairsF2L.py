import numpy as np

def LinkPairsF2L(cellPairs):
    LENinp = cellPairs[0].shape[0]
    TolPairs = len(cellPairs)

    PairResult = [None] * LENinp
    breakChains = []

    for i in range(LENinp):
        # MATLAB: PairResult{i} = cellPairs{1}(i, 1:2);
        PairResult[i] = list(cellPairs[0][i, :2])
        temp2ed = PairResult[i][1]

        Pframe = 0

        while True:
            Pframe += 1
            if Pframe >= TolPairs:
                break

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
