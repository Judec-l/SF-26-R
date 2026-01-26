"""
beadPosition.py - Refactored
Bead position detection from images
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class beadPosition:
    """
    Detect bead positions in microscopy images.

    Uses intensity thresholding and connected component analysis
    to find particle centroids with sub-pixel accuracy.
    """

    def __init__(self, image_path: str, threshold: float):
        """
        Initialize bead detector.

        Args:
            image_path: Path to image file
            threshold: Intensity threshold (0-1) for bead detection
        """
        self.image_path = image_path
        self.threshold = threshold
        self.image: Optional[np.ndarray] = None
        self.beadPositionLocal: Optional[np.ndarray] = None

    def calcBeadCenter(self) -> 'beadPosition':
        """
        Calculate bead center positions from image.

        Returns:
            Self for method chaining

        Raises:
            IOError: If image cannot be loaded
        """
        # Load image
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(f"Cannot load image: {self.image_path}")

        self.image = img.astype(float)

        logger.info(f"Loaded image: {self.image_path}, shape: {img.shape}")

        # Normalize to 0-1 range
        img_norm = self.image / self.image.max()

        # Apply threshold
        binary = (img_norm > self.threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)

        logger.info(f"Found {num_labels - 1} connected components")

        # Calculate centroids for each component
        centers = []

        for label_id in range(1, num_labels):  # Skip background (label 0)
            mask = (labels == label_id)
            area = np.sum(mask)

            # Filter by minimum area
            if area < 3:
                continue

            # Get pixel coordinates and intensities
            y_coords, x_coords = np.nonzero(mask)
            intensities = self.image[mask]

            # Calculate intensity-weighted centroid (sub-pixel accuracy)
            cx = np.sum(x_coords * intensities) / np.sum(intensities)
            cy = np.sum(y_coords * intensities) / np.sum(intensities)

            centers.append([cx, cy])

        self.beadPositionLocal = np.array(centers) if centers else np.empty((0, 2))

        logger.info(f"Detected {len(self.beadPositionLocal)} beads")

        return self

    def visualize(self, output_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of detected beads.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        if self.image is None or self.beadPositionLocal is None:
            raise ValueError("No image or positions available. Call calcBeadCenter() first.")

        # Convert to 8-bit for visualization
        vis_img = cv2.normalize(
            self.image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        # Draw markers at detected positions
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