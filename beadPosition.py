import numpy as np
import cv2

class beadPosition:
    def __init__(self, image_path, threshold):
        self.image_path = image_path
        self.threshold = threshold
        self.image = None
        self.beadPositionLocal = None

    def calcBeadCenter(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot load image: {self.image_path}")

        self.image = img.astype(float)

        # Normalize
        img_norm = self.image / self.image.max()

        # Threshold
        binary = img_norm > self.threshold

        # Connected components
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))

        centers = []
        for lab in range(1, num_labels):
            mask = labels == lab
            if np.sum(mask) < 3:
                continue

            y, x = np.nonzero(mask)
            intensities = self.image[mask]

            # Sub-pixel centroid
            cx = np.sum(x * intensities) / np.sum(intensities)
            cy = np.sum(y * intensities) / np.sum(intensities)

            centers.append([cx, cy])

        self.beadPositionLocal = np.array(centers)
        return self
