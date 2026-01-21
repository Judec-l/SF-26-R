import os
import numpy as np
import pickle

Rs = None
Rn = None
Rq = None
A = None
B = None
C = None
NR = None


def setParameter(parameter, field, default):
    return parameter[field] if field in parameter else default

def match_pairRelax4(x1, y1, x2, y2, r1, r2, max_dis, option):

    global Rs, Rn, Rq, A, B, C, NR

    I1 = []
    I2 = []
    confidence1 = []
    confidence2 = []
    drift = []
    info = []
    debug_info = []

    Rs = 1.5 * max_dis
    Rn = Rs
    Rq = 0.2 * Rs
    NR = 20

    A = 0.3
    B = 3
    C = 0

    savefileStr = setParameter(option, "savefileStr", "")
    resfolder = setParameter(option, "resfolder", "../PTLibrary2D/res_matchpair/")
    auxfolder = setParameter(option, "auxfolder", "../PTLibrary2D/aux_matchpair/")
    auxfileSave = setParameter(option, "AuxFileSave", "NO")

    os.makedirs(resfolder, exist_ok=True)
    os.makedirs(auxfolder, exist_ok=True)

    if savefileStr == "":
        preprocessFn = os.path.join(auxfolder, "matchpair_preproc.pkl")
        filename_result = os.path.join(resfolder, "matchpairRelax_res.pkl")
    else:
        preprocessFn = os.path.join(auxfolder, f"matchpair_preproc_{savefileStr}.pkl")
        filename_result = os.path.join(resfolder, f"matchpairRelax_res_{savefileStr}.pkl")

    x1 = np.asarray(x1).reshape(-1)
    y1 = np.asarray(y1).reshape(-1)
    r1 = np.asarray(r1).reshape(-1)
    x2 = np.asarray(x2).reshape(-1)
    y2 = np.asarray(y2).reshape(-1)
    r2 = np.asarray(r2).reshape(-1)

    N1 = len(x1)
    N2 = len(x2)

    if not os.path.exists(preprocessFn):
        print("calculating preprocess data stage1...")
        from match_pair_preprocess import match_pair_preprocess
        data = match_pair_preprocess(x1, y1, r1, x2, y2, r2)
        with open(preprocessFn, "wb") as f:
            pickle.dump(data, f)
    else:
        print("loading preprocess data stage1...")
        with open(preprocessFn, "rb") as f:
            data = pickle.load(f)

    (dist_min1a, dist_min1b,
     dist_min2a, dist_min2b,
     index1a, index1b,
     index2a, index2b) = data

    option_a = option.copy()
    option_a["str"] = "a"
    I1_temp, confidence1 = match_pairRelax_aux(
        x1, y1, x2, y2, r1, r2, index1a, index1b, option_a
    )

    option_b = option.copy()
    option_b["str"] = "b"
    I2_temp, confidence2 = match_pairRelax_aux(
        x2, y2, x1, y1, r2, r1, index2b, index2a, option_b
    )

    I1 = []
    I2 = []

    for n in range(N1):
        m = I1_temp[n]
        if m < 0:
            continue
        if I2_temp[m] == n:
            if (confidence1[n] > 0.9 or confidence2[m] > 0.9) or \
               (confidence1[n] > 0.5 and confidence2[m] > 0.5):

                p1 = np.array([x1[n], y1[n], r1[n]])
                p2 = np.array([x2[m], y2[m], r2[m]])
                if np.linalg.norm(p1 - p2) < max_dis:
                    I1.append(n)
                    I2.append(m)

    I1 = np.array(I1, dtype=int)
    I2 = np.array(I2, dtype=int)

    dx = x2[I2] - x1[I1]
    dy = y2[I2] - y1[I1]
    dr = r2[I2] - r1[I1]

    I1u = np.setdiff1d(np.arange(N1), I1)
    I2u = np.setdiff1d(np.arange(N2), I2)

    with open(filename_result, "wb") as f:
        pickle.dump(
            (I1, I2, I1u, I2u, dx, dy, dr, confidence1, confidence2),
            f
        )

    return I1, I2, I1u, I2u, dx, dy, dr, confidence1, confidence2, drift, info, debug_info

def match_pairRelax_aux(x1, y1, x2, y2, r1, r2, indexa, indexb, option):
    global Rs, Rn, Rq, A, B, C, NR
    N1 = len(x1)
    I_temp = np.full(N1, -1, dtype=int)
    confidence = np.zeros(N1)

    Pb = [None] * N1
    Nis = [None] * N1

    for n in range(N1):
        candidates = indexb[n]
        Pb[n] = np.ones(len(candidates)) / len(candidates)
        Nis[n] = candidates

    # Relaxation iterations
    for _ in range(NR):
        Pb_new = []
        for n in range(N1):
            p = np.array([x1[n], y1[n], r1[n]])
            probs = Pb[n]
            cands = Nis[n]

            scores = np.zeros(len(cands))
            for i, m in enumerate(cands):
                q = np.array([x2[m], y2[m], r2[m]])
                if np.linalg.norm(p - q) < Rs:
                    scores[i] += probs[i]

            if scores.sum() > 0:
                probs = probs * (A + B * scores)
                probs /= probs.sum()

            Pb_new.append(probs)
        Pb = Pb_new

    for n in range(N1):
        probs = Pb[n]
        if len(probs) > 0:
            idx = np.argmax(probs)
            I_temp[n] = Nis[n][idx]
            confidence[n] = probs[idx]

    return I_temp, confidence
