# plane_detector.py
import open3d as o3d
import numpy as np
from rich.console import Console

#console = Console(highlight=False)
console = Console(quiet=True, highlight=False)

class PlaneDetector:
    """
    Detects all significant planes in a point cloud using an iterative
    RANSAC (segment-then-classify) approach.

    Includes a Non-Maximum Suppression (NMS) step to filter
    redundant, overlapping planes.
    """

    def __init__(self,
                 voxel_size=0.05,
                 angle_tolerance_deg=10.0,
                 ransac_threshold=0.01,
                 ransac_n=3,
                 ransac_iterations=1000,
                 min_plane_points=100,
                 nms_normal_dot_threshold=0.99,
                 nms_distance_threshold=0.05,
                 plane_thickness=0.01):

        """
        Initializes the plane detector.

        Args:
            voxel_size (float): Voxel size for downsampling. 0 to disable.
            angle_tolerance_deg (float): Angle (in degrees) to classify a plane
                                         as horizontal or vertical.
            ransac_threshold (float): RANSAC distance threshold (in meters).
            ransac_n (int): Number of points to sample for RANSAC.
            ransac_iterations (int): Number of RANSAC iterations.
            min_plane_points (int): The minimum number of points a plane must
                                    have to be considered significant.
            nms_iou_threshold (float): Intersection-over-Union (IoU) threshold
                                       for NMS. Planes with IoU > this value
                                       will be suppressed.
        """
        self.voxel_size = voxel_size
        self.ransac_threshold = ransac_threshold
        self.ransac_n = ransac_n
        self.ransac_iterations = ransac_iterations
        self.min_plane_points = min_plane_points
        self.nms_normal_dot_threshold = nms_normal_dot_threshold
        self.nms_distance_threshold = nms_distance_threshold
        self.plane_thickness = plane_thickness

        # Gravity vector (assuming Z is up)
        self.gravity_vector = np.array([0, 0, 1.0])

        # Pre-calculate cosine tolerances for classification
        self.horizontal_cos_tolerance = np.cos(np.deg2rad(angle_tolerance_deg))
        self.vertical_cos_cutoff = np.sin(np.deg2rad(angle_tolerance_deg))

        console.log(f"PlaneDetector initialized. Min points: {min_plane_points}, Voxel: {voxel_size}m")
        console.log(f"NMS Params: Normal Dot > {nms_normal_dot_threshold}, Distance < {nms_distance_threshold}m")

    def detect(self, pcd: o3d.geometry.PointCloud):
        """
        Detects all planes in the point cloud iteratively, then filters
        the results using Non-Maximum Suppression.

        Args:
            pcd (o3d.geometry.PointCloud): The (downsampled) input point cloud.

        Returns:
            tuple: A tuple containing:
                (horizontal_planes, vertical_planes, oblique_planes, outlier_indices)

                - Each '_planes' list contains (model, indices) tuples.
                - 'outlier_indices' are the indices of points not
                   belonging to any significant plane.
        """
        horizontal_planes = []
        vertical_planes = []
        oblique_planes = []

        remaining_pcd = pcd
        original_indices = np.arange(len(pcd.points))

        console.log(f"Starting iterative plane detection on {len(pcd.points)} points...")

        while len(remaining_pcd.points) > self.min_plane_points:
            plane_model, inliers_relative = remaining_pcd.segment_plane(
                distance_threshold=self.ransac_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.ransac_iterations)

            if len(inliers_relative) < self.min_plane_points:
                console.log(f"Next best plane has {len(inliers_relative)} points. Stopping.")
                break

            normal_vector = plane_model[:3]
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            inliers_original = original_indices[inliers_relative]
            dot_product = np.abs(np.dot(normal_vector, self.gravity_vector))

            if dot_product > self.horizontal_cos_tolerance:
                if plane_model[2] < 0:
                    plane_model = [-x for x in plane_model]
                console.log(f"[blue]Found Horizontal Plane[/blue]: {len(inliers_original)} points.")
                horizontal_planes.append((plane_model, inliers_original))

            elif dot_product < self.vertical_cos_cutoff:
                console.log(f"[green]Found Vertical Plane[/green]: {len(inliers_original)} points.")
                vertical_planes.append((plane_model, inliers_original))

            else:
                console.log(f"[magenta]Found Oblique Plane[/magenta]: {len(inliers_original)} points.")
                oblique_planes.append((plane_model, inliers_original))

            # Remove the inliers and prepare for next iteration
            remaining_pcd = remaining_pcd.select_by_index(inliers_relative, invert=True)
            original_indices = np.delete(original_indices, inliers_relative)

        outlier_indices = original_indices
        console.log(
            f"Raw detection complete. Found {len(horizontal_planes)}H, {len(vertical_planes)}V, {len(oblique_planes)}O planes.")

        # Apply Non-Maximum Suppression ---

        console.log(f"Applying NMS to {len(horizontal_planes)} horizontal planes...")
        horizontal_planes = self._non_maximum_suppression(horizontal_planes)

        console.log(f"Applying NMS to {len(vertical_planes)} vertical planes...")
        vertical_planes = self._non_maximum_suppression(vertical_planes)

        console.log(f"Applying NMS to {len(oblique_planes)} oblique planes...")
        oblique_planes = self._non_maximum_suppression(oblique_planes)

        console.log(
            f"NMS complete. Remaining: {len(horizontal_planes)}H, {len(vertical_planes)}V, {len(oblique_planes)}O planes.")
        console.log(f"{len(outlier_indices)} points remaining as outliers.")

        return horizontal_planes, vertical_planes, oblique_planes, outlier_indices

    def get_colored_geometries(self, pcd: o3d.geometry.PointCloud):
        """
        A helper function to downsample, detect planes, and return a list
        of colored geometries (planes and outliers) for visualization.
        """
        if self.voxel_size > 0:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            console.log(f"Downsampled from {len(pcd.points)} to {len(pcd_down.points)} points.")
        else:
            pcd_down = pcd

        if not pcd_down.has_points():
            return []

        # Detect and filter planes
        horizontal, vertical, oblique, outliers = self.detect(pcd_down)

        geometries_to_draw = []

        # Color Horizontal planes (Blue)
        for model, indices in horizontal:
            plane_pcd = pcd_down.select_by_index(indices)
            plane_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
            geometries_to_draw.append(plane_pcd)

        # Color Vertical planes (Cycling colors)
        wall_color_map = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        for i, (model, indices) in enumerate(vertical):
            color = wall_color_map[i % len(wall_color_map)]
            plane_pcd = pcd_down.select_by_index(indices)
            plane_pcd.paint_uniform_color(color)
            geometries_to_draw.append(plane_pcd)

        # Color Oblique planes (Magenta)
        for model, indices in oblique:
            plane_pcd = pcd_down.select_by_index(indices)
            plane_pcd.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
            geometries_to_draw.append(plane_pcd)

        # Color Outliers (Gray)
        outlier_cloud = pcd_down.select_by_index(outliers)
        outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        geometries_to_draw.append(outlier_cloud)

        return geometries_to_draw

    def get_plane_primitives(self, pcd: o3d.geometry.PointCloud):
        """
        A helper function to downsample, detect planes, and return a list
        of thin o3d.geometry.OrientedBoundingBox primitives.

        This provides a "cleaner" visualization than drawing all the inliers.
        """
        if self.voxel_size > 0:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            console.log(f"Downsampling for primitives: {len(pcd.points)} -> {len(pcd_down.points)} points.")
        else:
            pcd_down = pcd

        if not pcd_down.has_points():
            return []

        # 1. Detect and filter planes
        horizontal, vertical, oblique, outliers = self.detect(pcd_down)

        geometries_to_draw = []

        # 2. Define colors
        h_color = [0.0, 0.0, 1.0]  # Blue
        v_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]  # Red, Green, Yellow, Cyan
        o_color = [1.0, 0.0, 1.0]  # Magenta

        # 3. Process Horizontal planes
        for model, indices in horizontal:
            inlier_pcd = pcd_down.select_by_index(indices)
            obb = inlier_pcd.get_oriented_bounding_box()

            # obb.extent is read-only. Copy, modify, then assign.
            new_extent = np.array(obb.extent)
            new_extent[np.argmin(new_extent)] = self.plane_thickness
            obb.extent = new_extent

            obb.color = h_color
            geometries_to_draw.append(obb)

        # 4. Process Vertical planes
        for i, (model, indices) in enumerate(vertical):
            inlier_pcd = pcd_down.select_by_index(indices)
            obb = inlier_pcd.get_oriented_bounding_box()

            new_extent = np.array(obb.extent)
            new_extent[np.argmin(new_extent)] = self.plane_thickness
            obb.extent = new_extent

            obb.color = v_colors[i % len(v_colors)]
            geometries_to_draw.append(obb)

        # 5. Process Oblique planes
        for model, indices in oblique:
            inlier_pcd = pcd_down.select_by_index(indices)
            obb = inlier_pcd.get_oriented_bounding_box()

            new_extent = np.array(obb.extent)
            new_extent[np.argmin(new_extent)] = self.plane_thickness
            obb.extent = new_extent

            obb.color = o_color
            geometries_to_draw.append(obb)

        # 6. Add outliers for context
        outlier_cloud = pcd_down.select_by_index(outliers)
        outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        geometries_to_draw.append(outlier_cloud)

        return geometries_to_draw

    def _non_maximum_suppression(self, plane_list):
        """
        Applies Non-Maximum Suppression to a list of (model, indices) tuples.
        Planes are sorted by the number of inliers (score). Overlapping
        planes are suppressed based on normal and distance similarity.
        """
        if not plane_list:
            return []

        # Sort planes by size (number of inliers) descending
        plane_list.sort(key=lambda p: len(p[1]), reverse=True)

        filtered_planes = []
        suppressed_flags = [False] * len(plane_list)

        for i in range(len(plane_list)):
            if suppressed_flags[i]:
                continue

            # This plane is a "winner"
            P_i_model, P_i_indices = plane_list[i]
            filtered_planes.append((P_i_model, P_i_indices))

            # Plane i parameters (A, B, C, D)
            normal_i = P_i_model[:3]
            # We must use the absolute value of D for distance comparison
            dist_i = np.abs(P_i_model[3])

            for j in range(i + 1, len(plane_list)):
                if suppressed_flags[j]:
                    continue

                P_j_model, P_j_indices = plane_list[j]

                # Plane j parameters
                normal_j = P_j_model[:3]
                dist_j = np.abs(P_j_model[3])

                # Compare normals: dot product. Use abs for parallel but opposite normals.
                normal_dot = np.dot(normal_i, normal_j)

                # Compare distances
                dist_diff = np.abs(dist_i - dist_j)

                # Suppress if normals are parallel AND distances are close
                if (normal_dot > self.nms_normal_dot_threshold and
                        dist_diff < self.nms_distance_threshold):
                    suppressed_flags[j] = True

        return filtered_planes

    def get_wall_inlier_points(self, pcd):
        """Return points belonging to vertical planes only."""
        _, vertical_planes, _, _ = self.detect(pcd)
        if not vertical_planes:
            return None

        all_indices = []
        for _, indices in vertical_planes:
            all_indices.extend(indices)

        if not all_indices:
            return None

        points = np.asarray(pcd.points)
        return points[all_indices]