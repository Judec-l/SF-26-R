"""
trackBeads.py - Refactored
Main tracking class for multi-frame particle tracking
"""
import numpy as np
from typing import Optional, Dict, Any, List
import logging

from match_pair import match_pair
from beadPosition import beadPosition

logger = logging.getLogger(__name__)


class trackBeads:
    """
    Multi-frame particle tracking system.

    Tracks beads across multiple frames using relaxation-based matching.
    Supports both pre-loaded positions and on-the-fly image analysis.
    """

    def __init__(
            self,
            positionArray1: np.ndarray,
            positionArray2: np.ndarray,
            dataName: str,
            maxBeadDisplacement: float,
            beadParam: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize bead tracker.

        Args:
            positionArray1: Positions in first frame (n x 2 or n x 3)
            positionArray2: Positions in second frame (n x 2 or n x 3)
            dataName: Name identifier for saving/loading
            maxBeadDisplacement: Maximum expected displacement between frames
            beadParam: Optional parameter dictionary
        """
        if beadParam is None:
            beadParam = {}

        self.dataName = dataName
        self.beadParam = beadParam
        self.matchArray = []

        ioInfo = beadParam.get('ioInfo', {})
        self.nFrames = ioInfo.get('nFrames', 2)

        # Setup parameters
        self.beadParam['max_dis'] = maxBeadDisplacement
        self.beadParam.setdefault('matchpair_option', {})
        self.beadParam['matchpair_option'].update({
            'load': 'YES',
            'clean': 'YES',
            'AuxFileSave': 'YES',
            'loadRes': 'YES',
            'ResFileSave': 'YES',
            'dataClean': 1,
            'savefileStr': dataName,
            'loadfileStr': dataName,
            'preprocessFn': f"{dataName}_pre",
            'NOISE_LEVEL': 0.0
        })

        self.beadParam['threshold_option'] = {
            'threshold1': 0.95,
            'thresholdaux': 0.95,
            'threshold1L': 0.9,
            'thresholdauxL': 0.9,
            'threshold1H': 0.999,
            'thresholdauxH': 0.999
        }

        self.beadParam['matchpair_option']['threshold'] = \
            self.beadParam['threshold_option']

        self.beadParam['threshold'] = 0.08
        self.beadParam['MATCH_STRATEGY'] = 'FromPrevious'

        # Initialize position arrays
        self.positionArray = [None] * self.nFrames

        positionArray1 = np.asarray(positionArray1)
        positionArray2 = np.asarray(positionArray2)

        # Ensure 3D positions (add z=0 if 2D)
        if positionArray1.shape[1] == 2:
            positionArray1 = np.column_stack((
                positionArray1,
                np.zeros(len(positionArray1))
            ))
        if positionArray2.shape[1] == 2:
            positionArray2 = np.column_stack((
                positionArray2,
                np.zeros(len(positionArray2))
            ))

        self.positionArray[0] = positionArray1
        self.positionArray[1] = positionArray2

        # Perform initial matching
        self.matchBeadPositionT1T2()

        self.T1 = ioInfo.get('T1', 1)

        logger.info(f"Initialized trackBeads: {dataName}")
        logger.info(f"Frame 1: {len(self.positionArray[0])} beads")
        logger.info(f"Frame 2: {len(self.positionArray[1])} beads")

    def locateBeadPosition(self, ioInfo: Dict[str, Any], k: int) -> None:
        """
        Detect bead positions from image file.

        Args:
            ioInfo: Dictionary containing image path and filename info
            k: Frame index
        """
        imagePath = ioInfo['imagePath'] + ioInfo['imageFnStackArray'][k]

        bp = beadPosition(imagePath, self.beadParam['threshold'])
        bp = bp.calcBeadCenter()

        self.positionArray[k] = bp.beadPositionLocal
        logger.info(f"Located {len(bp.beadPositionLocal)} beads in frame {k}")

    def matchBeadPositionT1T2(self) -> None:
        """
        Match beads between frames T1 and T2 (frames 0 and 1).
        """
        t1 = 0
        t2 = 1

        bp1 = self.positionArray[t1]
        bp2 = self.positionArray[t2]

        # Extract coordinates
        x1, y1, z1 = bp1[:, 0], bp1[:, 1], bp1[:, 2]
        x2, y2, z2 = bp2[:, 0], bp2[:, 1], bp2[:, 2]

        logger.info(f"Matching frames {t1} and {t2}...")

        # Perform matching
        (meanDist, I1, I2, I1u, I2u,
         dx, dy, dz,
         confidence1, confidence2,
         drift, info, debug_info) = match_pair(
            x1, y1, x2, y2, z1, z2,
            self.beadParam['max_dis'],
            self.beadParam['matchpair_option']
        )

        # Create displacement array
        bd = np.column_stack((dx, dy, dz))

        # Store match information
        matchInfo = {
            'I1': I1,
            'I2': I2,
            'I1u': I1u,
            'I2u': I2u,
            'drift': drift,
            'info': info,
            'matchRatio': len(I1) / min(len(x1), len(x2)) if min(len(x1), len(x2)) > 0 else 0,
            't1': t1 + 1,
            't2': t2 + 1
        }

        data = {
            't1': t1 + 1,
            't2': t2 + 1,
            'displacement': bd,
            'matchInfo': matchInfo,
            'meanDist': meanDist
        }

        self.matchArray.append(data)

        logger.info(f"Matched {len(I1)} beads ({matchInfo['matchRatio']:.1%})")

    def matchBeadPosition(self, t2: Optional[int] = None) -> None:
        """
        Match bead positions for specified frame or all frames.

        Args:
            t2: Target frame number (if None, match all frames)
        """
        if t2 is not None:
            if self.beadParam['MATCH_STRATEGY'] == 'FromFirst':
                t1 = self.T1
            else:
                t1 = t2 - 1
            self.matchBeadPositionT1T2()
        else:
            # Match all consecutive frame pairs
            for k in range(self.nFrames - 1):
                self.matchBeadPosition(k + self.T1 + 1)

    def beadPlot(self, t: int, param: Optional[Dict[str, Any]] = None,
                 h: Optional[Any] = None):
        """
        Plot bead positions for a given frame.

        Args:
            t: Frame number (1-indexed)
            param: Plot parameters (color, marker, markersize)
            h: Existing axes handle (if None, creates new figure)

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if param is None:
            param = {'color': 'b', 'marker': 'o', 'markersize': 2}

        if h is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = h

        k = t - self.T1
        bp = self.positionArray[k]

        ax.scatter(
            bp[:, 0], bp[:, 1], bp[:, 2],
            c=param.get('color', 'b'),
            marker=param.get('marker', 'o'),
            s=param.get('markersize', 2)
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {t}')

        return ax

    def matchPlot(self, t1: int, t2: int,
                  param: Optional[Dict[str, Any]] = None,
                  h: Optional[Any] = None):
        """
        Plot matched bead pairs between two frames.

        Args:
            t1: First frame number (1-indexed)
            t2: Second frame number (1-indexed)
            param: Plot parameters (color, linewidth)
            h: Existing axes handle (if None, creates new figure)

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if param is None:
            param = {'color': 'k', 'linewidth': 2}

        if h is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = h

        bp1 = self.positionArray[t1 - self.T1]
        bp2 = self.positionArray[t2 - self.T1]

        # Find matching entry
        matchInfo = None
        for entry in self.matchArray:
            if entry['t1'] == t1 and entry['t2'] == t2:
                matchInfo = entry['matchInfo']
                break

        if matchInfo is None:
            logger.warning(f"No match data found for frames {t1} and {t2}")
            return ax

        I1 = matchInfo['I1']
        I2 = matchInfo['I2']

        # Draw lines between matched pairs
        for i1, i2 in zip(I1, I2):
            ax.plot(
                [bp1[i1, 0], bp2[i2, 0]],
                [bp1[i1, 1], bp2[i2, 1]],
                [bp1[i1, 2], bp2[i2, 2]],
                color=param.get('color', 'k'),
                linewidth=param.get('linewidth', 2)
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Matches: Frame {t1} -> {t2}')

        return ax

    def loadFromOld(self, Fn: str, n: int) -> None:
        """
        Load positions from old MATLAB format.

        Args:
            Fn: Filename of MATLAB file
            n: Index in cell array
        """
        import scipy.io as sio

        tempData = sio.loadmat(Fn)
        cellData = tempData['positionData']['beadCenterUm'][0, n - 1]

        self.nFrames = len(cellData)
        self.positionArray = list(cellData)

        logger.info(f"Loaded {self.nFrames} frames from {Fn}")

    def loadFromSimpleData(self, Fn: str) -> None:
        """
        Load positions from simple MATLAB format.

        Args:
            Fn: Filename of MATLAB file
        """
        import scipy.io as sio

        tempData = sio.loadmat(Fn)

        self.nFrames = 2
        self.positionArray = [None] * 2

        self.positionArray[0] = tempData['bead_position'][0][0]
        displacement = tempData['bead_displacement'][0][0]

        self.positionArray[1] = self.positionArray[0] + displacement

        self.matchArray = [{
            't1': 1,
            't2': 2,
            'displacement': displacement
        }]

        logger.info(f"Loaded simple data from {Fn}")

    def save(self) -> None:
        """
        Save tracking results (placeholder for future implementation).
        """
        logger.warning("Save method not yet implemented")
        pass

    def __repr__(self) -> str:
        return (f"trackBeads(name='{self.dataName}', frames={self.nFrames}, "
                f"matches={len(self.matchArray)})")