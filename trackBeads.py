import numpy as np
from match_pair import match_pair
from beadPosition import beadPosition

class trackBeads:
    def __init__(self, positionArray1, positionArray2, dataName,
                 maxBeadDisplacement, beadParam=None):
        if beadParam is None:
            beadParam = {}

        self.dataName = dataName
        self.beadParam = beadParam
        self.matchArray = []

        ioInfo = beadParam.get('ioInfo', {})
        self.nFrames = ioInfo.get('nFrames', 2)

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

        self.positionArray = [None] * self.nFrames

        positionArray1 = np.asarray(positionArray1)
        positionArray2 = np.asarray(positionArray2)

        if positionArray1.shape[1] == 2:
            positionArray1 = np.column_stack((positionArray1, np.zeros(len(positionArray1))))
        if positionArray2.shape[1] == 2:
            positionArray2 = np.column_stack((positionArray2, np.zeros(len(positionArray2))))

        self.positionArray[0] = positionArray1
        self.positionArray[1] = positionArray2
        self.matchBeadPositionT1T2()
        self.T1 = ioInfo.get('T1', 1)

    def locateBeadPosition(self, ioInfo, k):
        bp = beadPosition(
            ioInfo['imagePath'] + ioInfo['imageFnStackArray'][k],
            self.beadParam['threshold']
        )
        bp = bp.calcBeadCenter()
        self.positionArray[k] = bp.beadPositionLocal

    def matchBeadPositionT1T2(self):
        t1 = 0
        t2 = 1

        bp1 = self.positionArray[t1]
        bp2 = self.positionArray[t2]

        x1, y1, z1 = bp1[:, 0], bp1[:, 1], bp1[:, 2]
        x2, y2, z2 = bp2[:, 0], bp2[:, 1], bp2[:, 2]

        (meanDist, I1, I2, I1u, I2u,
         dx, dy, dz,
         confidence1, confidence2,
         drift, info, debug_info) = match_pair(
            x1, y1, x2, y2, z1, z2,
            self.beadParam['max_dis'],
            self.beadParam['matchpair_option']
        )

        bd = np.column_stack((dx, dy, dz))

        matchInfo = {
            'I1': I1,
            'I2': I2,
            'I1u': I1u,
            'I2u': I2u,
            'drift': drift,
            'info': info,
            'matchRatio': len(I1) / min(len(x1), len(x2)),
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

    def matchBeadPosition(self, t2=None):
        if t2 is not None:
            if self.beadParam['MATCH_STRATEGY'] == 'FromFirst':
                t1 = self.T1
            else:
                t1 = t2 - 1
            self.matchBeadPositionT1T2()
        else:
            for k in range(self.nFrames - 1):
                self.matchBeadPosition(k + self.T1 + 1)

    def beadPlot(self, t, param=None, h=None):
        import matplotlib.pyplot as plt

        if param is None:
            param = {'color': 'b', 'marker': 'o', 'markersize': 2}

        if h is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax = h

        k = t - self.T1
        bp = self.positionArray[k]

        ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2],
                   c=param['color'],
                   marker=param['marker'],
                   s=param['markersize'])

        return ax

    def matchPlot(self, t1, t2, param=None, h=None):
        import matplotlib.pyplot as plt

        if param is None:
            param = {'color': 'k', 'linewidth': 2}

        if h is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax = h

        bp1 = self.positionArray[t1 - self.T1]
        bp2 = self.positionArray[t2 - self.T1]

        for entry in self.matchArray:
            if entry['t1'] == t1 and entry['t2'] == t2:
                matchInfo = entry['matchInfo']
                break

        I1 = matchInfo['I1']
        I2 = matchInfo['I2']

        for i1, i2 in zip(I1, I2):
            ax.plot(
                [bp1[i1, 0], bp2[i2, 0]],
                [bp1[i1, 1], bp2[i2, 1]],
                [bp1[i1, 2], bp2[i2, 2]],
                color=param['color'],
                linewidth=param['linewidth']
            )

        return ax

    def loadFromOld(self, Fn, n):
        import scipy.io as sio
        tempData = sio.loadmat(Fn)
        cellData = tempData['positionData']['beadCenterUm'][0, n - 1]
        self.nFrames = len(cellData)
        self.positionArray = list(cellData)

    def loadFromSimpleData(self, Fn):
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

    def save(self):
        pass
