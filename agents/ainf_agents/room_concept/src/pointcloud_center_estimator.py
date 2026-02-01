"""
Pointcloud Center Estimator - Python implementation.

Estimates room center, dimensions, and orientation from LIDAR points
using geometric analysis without requiring ground truth.

Algorithm pipeline:
    Points → Filter → Boundary Extraction → Outlier Removal → Convex Hull → OBB → Center

Based on the C++ implementation in pointcloud_center_estimator.cpp

References:
    - Rotating Calipers algorithm for minimum area bounding box
    - Statistical Outlier Removal (SOR) for noise filtering
    - Graham Scan for convex hull computation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class OBB:
    """Oriented Bounding Box result."""
    center: np.ndarray      # [x, y] center position
    width: float            # Width (larger dimension)
    height: float           # Height (smaller dimension)
    rotation: float         # Rotation angle in radians


@dataclass
class EstimatorConfig:
    """Configuration for the pointcloud center estimator."""
    min_range: float = 0.3          # Minimum valid range (meters)
    max_range: float = 10.0         # Maximum valid range (meters)
    num_sectors: int = 72           # Number of angular sectors (5° each)
    min_valid_points: int = 20      # Minimum points required
    outlier_std_threshold: float = 2.0  # SOR threshold (number of std devs)
    k_neighbors: int = 5            # K for KNN in outlier removal


class PointcloudCenterEstimator:
    """
    Estimates room center and geometry from LIDAR pointcloud.

    This class provides a robust, GT-free method for estimating:
    - Room center (x, y) in robot frame
    - Room dimensions (width, height)
    - Room orientation (rotation angle)

    The algorithm:
    1. Filters points by range
    2. Extracts boundary points (farthest in each angular sector)
    3. Removes statistical outliers
    4. Computes convex hull
    5. Finds minimum-area Oriented Bounding Box (OBB)

    The OBB center gives the room center, its dimensions give room size,
    and its rotation gives room orientation - all without external references.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None):
        self.config = config or EstimatorConfig()

    def estimate(self, points: np.ndarray) -> Optional[OBB]:
        """
        Estimate room geometry from LIDAR points.

        Args:
            points: [N, 2] array of LIDAR points in robot frame (meters)

        Returns:
            OBB with center, dimensions, and rotation, or None if insufficient points
        """
        if len(points) < self.config.min_valid_points:
            return None

        # Step 1: Filter by range
        cleaned = self._filter_points(points)
        if len(cleaned) < 8:
            return self._fallback_centroid(points)

        # Step 2: Extract boundary points (farthest in each sector)
        boundary = self._extract_boundary_points(cleaned)
        if len(boundary) < 4:
            return self._fallback_centroid(cleaned)

        # Step 3: Remove statistical outliers
        boundary = self._remove_statistical_outliers(boundary)
        if len(boundary) < 4:
            return self._fallback_centroid(cleaned)

        # Step 4: Compute convex hull
        hull = self._compute_convex_hull(boundary)
        if len(hull) < 3:
            return self._fallback_centroid(cleaned)

        # Step 5: Compute minimum-area OBB using rotating calipers
        obb = self._compute_obb(hull)

        return obb

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points by range [min_range, max_range]."""
        ranges = np.linalg.norm(points, axis=1)
        mask = (ranges >= self.config.min_range) & (ranges <= self.config.max_range)
        return points[mask]

    def _extract_boundary_points(self, points: np.ndarray) -> np.ndarray:
        """
        Extract boundary points by taking the farthest point in each angular sector.

        This effectively finds the points that lie on the room walls,
        filtering out any interior points (obstacles, etc.).
        """
        num_sectors = self.config.num_sectors
        sector_angle = 2.0 * np.pi / num_sectors

        # Compute angles for all points
        angles = np.arctan2(points[:, 1], points[:, 0])
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

        # Assign points to sectors
        sector_indices = (angles / sector_angle).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)

        # Compute ranges
        ranges = np.linalg.norm(points, axis=1)

        # Find farthest point in each sector
        boundary = []
        for sector in range(num_sectors):
            mask = sector_indices == sector
            if not np.any(mask):
                continue

            sector_points = points[mask]
            sector_ranges = ranges[mask]

            # Get the farthest point
            max_idx = np.argmax(sector_ranges)
            farthest = sector_points[max_idx]

            # Check if it's a local maximum (simple validation)
            if self._is_local_maximum(sector_ranges[max_idx], sector_ranges, threshold=0.5):
                boundary.append(farthest)

        return np.array(boundary) if boundary else np.array([]).reshape(0, 2)

    def _is_local_maximum(self, candidate_range: float,
                          neighbor_ranges: np.ndarray,
                          threshold: float) -> bool:
        """Check if candidate is a local maximum within threshold."""
        return np.all(neighbor_ranges <= candidate_range + threshold)

    def _remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Remove statistical outliers using K-nearest neighbors.

        Points with average distance to K neighbors outside μ ± n·σ are removed.
        """
        if len(points) < 6:
            return points

        k = min(self.config.k_neighbors, len(points) - 1)

        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # For each point, get average distance to K nearest neighbors
        avg_distances = []
        for i in range(len(points)):
            # Sort distances and take K smallest (excluding self at index 0)
            sorted_dist = np.sort(distances[i])[1:k+1]
            avg_distances.append(np.mean(sorted_dist))

        avg_distances = np.array(avg_distances)

        # Compute statistics
        mean = np.mean(avg_distances)
        std = np.std(avg_distances)

        # Filter outliers
        threshold = self.config.outlier_std_threshold * std
        mask = np.abs(avg_distances - mean) < threshold

        return points[mask]

    def _compute_convex_hull(self, points: np.ndarray) -> np.ndarray:
        """
        Compute convex hull using Graham Scan algorithm.

        Returns vertices of the convex hull in counter-clockwise order.
        """
        if len(points) <= 3:
            return points

        # Find pivot (lowest y, then leftmost x)
        pivot_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
        pivot = points[pivot_idx]

        # Sort by polar angle from pivot
        diff = points - pivot
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        # Graham scan
        hull = []
        for p in sorted_points:
            while len(hull) >= 2:
                # Cross product to check turn direction
                a, b = hull[-2], hull[-1]
                ab = b - a
                ac = p - a
                cross = ab[0] * ac[1] - ab[1] * ac[0]

                if cross > 0:  # Left turn, keep point
                    break
                hull.pop()  # Right turn or collinear, remove last point
            hull.append(p)

        return np.array(hull)

    def _compute_obb(self, hull: np.ndarray) -> OBB:
        """
        Compute minimum-area Oriented Bounding Box using rotating calipers.

        For each edge of the convex hull, compute the axis-aligned bounding box
        when that edge is aligned with an axis. Return the one with minimum area.
        """
        if len(hull) < 3:
            # Fallback for degenerate cases
            center = np.mean(hull, axis=0) if len(hull) > 0 else np.zeros(2)
            return OBB(center=center, width=1.0, height=1.0, rotation=0.0)

        min_area = float('inf')
        best_obb = None

        for i in range(len(hull)):
            # Get edge vector
            p1 = hull[i]
            p2 = hull[(i + 1) % len(hull)]
            edge = p2 - p1

            # Compute rotation angle to align edge with x-axis
            angle = np.arctan2(edge[1], edge[0])

            # Rotation matrix
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rotation_matrix = np.array([[cos_a, -sin_a],
                                        [sin_a, cos_a]])

            # Rotate hull points
            rotated = (rotation_matrix @ hull.T).T

            # Compute axis-aligned bounding box
            min_x, max_x = np.min(rotated[:, 0]), np.max(rotated[:, 0])
            min_y, max_y = np.min(rotated[:, 1]), np.max(rotated[:, 1])

            width = max_x - min_x
            height = max_y - min_y
            area = width * height

            if area < min_area:
                min_area = area

                # Compute center in rotated frame, then rotate back
                center_rot = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])

                # Inverse rotation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                inv_rotation = np.array([[cos_a, -sin_a],
                                         [sin_a, cos_a]])
                center = inv_rotation @ center_rot

                # Ensure width >= height (width is the larger dimension)
                if width >= height:
                    best_obb = OBB(center=center, width=width, height=height, rotation=angle)
                else:
                    # Swap and adjust rotation by 90°
                    best_obb = OBB(center=center, width=height, height=width,
                                   rotation=angle + np.pi / 2)

        # Normalize rotation to [-π, π]
        if best_obb:
            best_obb.rotation = np.arctan2(np.sin(best_obb.rotation),
                                           np.cos(best_obb.rotation))

        return best_obb

    def _fallback_centroid(self, points: np.ndarray) -> OBB:
        """
        Fallback method using robust centroid (median) when other methods fail.
        """
        if len(points) == 0:
            return OBB(center=np.zeros(2), width=6.0, height=4.0, rotation=0.0)

        # Use median for robustness
        center = np.array([np.median(points[:, 0]), np.median(points[:, 1])])

        # Estimate dimensions from point spread
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])

        # Ensure width >= height
        if width < height:
            width, height = height, width
            rotation = np.pi / 2
        else:
            rotation = 0.0

        return OBB(center=center, width=width, height=height, rotation=rotation)


def estimate_room_from_lidar(points: np.ndarray,
                              config: Optional[EstimatorConfig] = None) -> Optional[OBB]:
    """
    Convenience function to estimate room geometry from LIDAR points.

    This function provides a GT-free initial estimate of:
    - Robot position in room frame (negative of OBB center)
    - Room dimensions (OBB width and height)
    - Robot orientation (OBB rotation)

    Args:
        points: [N, 2] LIDAR points in robot frame (meters)
        config: Optional configuration parameters

    Returns:
        OBB containing room geometry, or None if estimation failed

    Example:
        >>> lidar_points = get_lidar_data()  # [N, 2] in meters
        >>> obb = estimate_room_from_lidar(lidar_points)
        >>> if obb:
        >>>     robot_x = -obb.center[0]  # Robot position is negative of room center
        >>>     robot_y = -obb.center[1]
        >>>     room_width = obb.width
        >>>     room_height = obb.height
        >>>     robot_theta = -obb.rotation
    """
    estimator = PointcloudCenterEstimator(config)
    return estimator.estimate(points)
