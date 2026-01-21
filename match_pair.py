import os
import numpy as np
from scipy.io import loadmat, savemat
from match_pair_preprocess import match_pair_preprocess
from match_pairRelax4 import match_pairRelax4

def setParameter(option, key, default):
    if option is None:
        return default
    return option.get(key, default)

def match_pair(x1, y1, x2, y2, r1, r2, max_dis, option, parameters=None):
    if len(x1) < 20 or len(x2) < 20:
        return (
            0, [], [], [], [], [], [], [],
            [], [], [], [], []
        )

    print(os.getcwd())
    print("Initializing matching algorithm...")

    N1 = len(x1)
    N2 = len(x2)
    print(N1)
    print(N2)
    print(max_dis)

    if not os.path.exists("../PTLibrary2D/aux_matchpair"):
        raise FileNotFoundError(
            "There is no aux_matchpair in ../PTLibrary2D/aux_matchpair"
        )

    if not os.path.exists("../PTLibrary2D/res_matchpair"):
        raise FileNotFoundError(
            "There is no res_matchpair in ../PTLibrary2D/res_matchpair"
        )

    resfolder = setParameter(option, "resfolder", "../PTLibrary2D/res_matchpair/")
    auxfolder = setParameter(option, "auxfolder", "../PTLibrary2D/aux_matchpair/")
    savefileStr_temp = setParameter(option, "savefileStr", "")
    print(f"Data: {savefileStr_temp}")

    savefileStr = option.get("savefileStr", "")
    loadfileStr_temp = setParameter(option, "loadfileStr", "")
    loadfileStr = option.get("loadfileStr", "")

    if loadfileStr_temp == "":
        loadfilename = os.path.join(resfolder, "matchpairRes.mat")
    else:
        loadfilename = os.path.join(
            resfolder, f"matchpairRes_{loadfileStr_temp}.mat"
        )

    preprocessFn = os.path.join(
        auxfolder, f"matchpair_preproc_{savefileStr}.mat"
    )

    # Preprocess
    if os.path.exists(preprocessFn):
        print("Loading preprocess data...")
        data = loadmat(preprocessFn)
        dist_min1a = data["dist_min1a"].squeeze()
        dist_min1b = data["dist_min1b"].squeeze()
        dist_min2a = data["dist_min2a"].squeeze()
        dist_min2b = data["dist_min2b"].squeeze()
    else:
        print("Calculating preprocess data...")
        (
            dist_min1a, dist_min1b,
            dist_min2a, dist_min2b,
            index1a, index1b,
            index2a, index2b
        ) = match_pair_preprocess(x1, y1, r1, x2, y2, r2)

        print("Saving preprocess data...")
        savemat(
            preprocessFn,
            {
                "dist_min1a": dist_min1a,
                "dist_min1b": dist_min1b,
                "dist_min2a": dist_min2a,
                "dist_min2b": dist_min2b,
                "index1a": index1a,
                "index1b": index1b,
                "index2a": index2a,
                "index2b": index2b,
            }
        )

    meanDist = {}
    meanDist["Dist1a"] = np.mean(dist_min1a)
    meanDist["Dist2b"] = np.mean(dist_min2b)
    meanDist["Dist1b"] = np.mean(dist_min1b)
    meanDist["Dist2a"] = np.mean(dist_min2a)

    if option.get("loadRes", "").upper() == "YES" and os.path.exists(loadfilename):
        print("Loading existing matching result...")
        print(loadfilename)
        data = loadmat(loadfilename)

        I1 = data["I1"]
        I2 = data["I2"]
        I1u = data["I1u"]
        I2u = data["I2u"]
        dx = data["dx"]
        dy = data["dy"]
        dr = data["dr"]
        confidence1 = data["confidence1"]
        confidence2 = data["confidence2"]
        drift = data["drift"]
        info = data["info"]
        debug_info = data["debug_info"]

    else:
        if parameters is not None:
            (
                I1, I2, I1u, I2u,
                dx, dy, dr,
                confidence1, confidence2,
                drift, info, debug_info
            ) = match_pair(
                x1, y1, x2, y2,
                r1, r2, max_dis,
                option, parameters
            )
        else:
            (
                I1, I2, I1u, I2u,
                dx, dy, dr,
                confidence1, confidence2,
                drift, info, debug_info
            ) = match_pairRelax4(
                x1, y1, x2, y2,
                r1, r2, max_dis,
                option
            )

        if option.get("ResFileSave", "").upper() == "YES":
            if savefileStr_temp == "":
                savefilename = os.path.join(resfolder, "matchpairRes.mat")
            else:
                savefilename = os.path.join(
                    resfolder, f"matchpairRes_{savefileStr_temp}.mat"
                )

            savemat(
                savefilename,
                {
                    "I1": I1,
                    "I2": I2,
                    "I1u": I1u,
                    "I2u": I2u,
                    "dx": dx,
                    "dy": dy,
                    "dr": dr,
                    "confidence1": confidence1,
                    "confidence2": confidence2,
                    "drift": drift,
                    "info": info,
                    "debug_info": debug_info,
                }
            )

    return (
        meanDist, I1, I2, I1u, I2u,
        dx, dy, dr,
        confidence1, confidence2,
        drift, info, debug_info
    )

