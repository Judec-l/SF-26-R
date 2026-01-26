import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class beadPosition:

    def __init__(self, image_path: str, threshold: float):
        self.image_path = image_path
        self.threshold = threshold
        self.image: Optional[np.ndarray] = None
        self.beadPositionLocal: Optional[np.ndarray] = None

    def calcBeadCenter(self) -> 'beadPosition':
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(f"Cannot load image: {self.image_path}")

        self.image = img.astype(float)

        logger.info(f"Loaded image: {self.image_path}, shape: {img.shape}")

        img_norm = self.image / self.image.max()

        binary = (img_norm > self.threshold).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(binary)

        logger.info(f"Found {num_labels - 1} connected components")

        centers = []

        for label_id in range(1, num_labels):  # Skip background (label 0)
            mask = (labels == label_id)
            area = np.sum(mask)

            if area < 3:
                continue

            y_coords, x_coords = np.nonzero(mask)
            intensities = self.image[mask]

            cx = np.sum(x_coords * intensities) / np.sum(intensities)
            cy = np.sum(y_coords * intensities) / np.sum(intensities)

            centers.append([cx, cy])

        self.beadPositionLocal = np.array(centers) if centers else np.empty((0, 2))

        logger.info(f"Detected {len(self.beadPositionLocal)} beads")

        return self

    def visualize(self, output_path: Optional[str] = None) -> np.ndarray:
        if self.image is None or self.beadPositionLocal is None:
            raise ValueError("No image or positions available. Call calcBeadCenter() first.")

        vis_img = cv2.normalize(
            self.image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        for x, y in self.beadPositionLocal:
            cv2.circle(vis_img, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv2.circle(vis_img, (int(x), int(y)), 2, (0, 0, 255), -1)  # Center dot

        if output_path:
            cv2.imwrite(output_path, vis_img)
            logger.info(f"Saved visualization to {output_path}")

        return vis_img

    def __repr__(self) -> str:
        n_beads = len(self.beadPositionLocal) if self.beadPositionLocal is not None else 0
        return f"beadPosition(path='{self.image_path}', threshold={self.threshold}, beads={n_beads})"