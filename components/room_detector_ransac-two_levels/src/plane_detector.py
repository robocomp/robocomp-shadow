# plane_detector.py
import open3d as o3d
import numpy as np
from rich.console import Console

console = Console(highlight=True)
#console = Console(quiet=True, highlight=False)


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

    def detect_with_prior(self, pcd, particle,
                          angular_tolerance=15.0,
                          distance_tolerance=0.3,
                          refine_alpha=0.7):
        """
        Detect planes with soft prior from room model.

        Strategy:
        - Detect planes normally (unchanged)
        - Validate vertical planes against expected walls
        - Refine matching planes toward model (stability)
        - Keep unmatched planes separate (obstacles/new geometry)

        Args:
            pcd: Point cloud
            particle: Current best particle with room estimate
            angular_tolerance: Max angle difference for matching (degrees)
            distance_tolerance: Max distance difference for matching (meters)
            refine_alpha: Blend toward expected (0=no change, 1=fully snap)

        Returns:
            (horizontal_planes, validated_vertical, unmatched_vertical,
             oblique_planes, outliers)
        """
        if particle is None:
            return [[], [], [], [], []]

        # 1. Normal detection (unchanged)
        h_planes, v_planes, o_planes, outliers = self.detect(pcd)

        # 2. Get expected wall planes from particle
        expected_walls = self._get_expected_walls(particle)

        # 3. Validate vertical planes against model
        validated_v = []
        unmatched_v = []

        for plane_model, indices in v_planes:
            matched = False

            for exp_normal, exp_dist in expected_walls:
                if self._planes_compatible(plane_model, exp_normal, exp_dist,
                                           angular_tolerance, distance_tolerance):
                    # Snap to expected (soft correction)
                    refined_model = self._refine_toward_expected(
                        plane_model, exp_normal, exp_dist, alpha=refine_alpha)
                    validated_v.append((refined_model, indices))
                    matched = True
                    console.log(f"[cyan]Matched wall plane[/cyan]: refined toward model")
                    break

            if not matched:
                unmatched_v.append((plane_model, indices))
                console.log(f"[yellow]Unmatched plane[/yellow]: keeping as obstacle/new geometry")

        console.log(f"Prior validation: {len(validated_v)} validated, {len(unmatched_v)} unmatched")

        return h_planes, validated_v, unmatched_v, o_planes, outliers

    """
    This version SEEDS RANSAC with expected walls first, then discovers additional planes.
    """

    def detect_with_prior_seeded(self, pcd, particle,
                                 angular_tolerance=10.0,
                                 distance_tolerance=0.1,
                                 refine_alpha=0.7,
                                 seed_threshold_factor=0.8):
        """
        Detect planes with STRONG prior from room model.

        Strategy:
        1. Get expected walls from particle
        2. SEED RANSAC: Test each expected wall directly against point cloud
        3. Accept expected walls with sufficient inliers (remove from pcd)
        4. Run RANSAC on remaining points to find novel planes
        5. Validate/refine all detected planes

        This is more aggressive than detect_with_prior() - it FORCES the expected
        walls to be tested first before random sampling.

        Args:
            pcd: Point cloud
            particle: Current best particle with room estimate
            angular_tolerance: Max angle difference for matching (degrees)
            distance_tolerance: Max distance difference for matching (meters)
            refine_alpha: Blend toward expected (0=no change, 1=fully snap)
            seed_threshold_factor: Multiplier for min_plane_points when seeding
                                   (0.8 means expected walls need 80% of normal threshold)

        Returns:
            (horizontal_planes, validated_vertical, unmatched_vertical,
             oblique_planes, outliers)
        """
        if particle is None:
            return [[], [], [], [], []]

        import numpy as np

        # 1. Get expected wall planes from particle
        expected_walls = self._get_expected_walls(particle)
        if not expected_walls:
            # No model available, fall back to normal detection
            return self.detect(pcd)

        console.log(f"[blue]Seeded RANSAC: Testing {len(expected_walls)} expected walls first[/blue]")

        # 2. SEED PHASE: Test each expected wall directly
        seeded_planes = []
        remaining_pcd = pcd
        remaining_indices = np.arange(len(pcd.points))

        min_seed_inliers = int(self.min_plane_points * seed_threshold_factor)

        for i, (exp_normal, exp_dist) in enumerate(expected_walls):
            if len(remaining_pcd.points) < min_seed_inliers:
                break

            # Construct plane model [A, B, C, D] from normal and distance
            # Plane equation: Ax + By + Cz + D = 0, where [A,B,C] is normalized
            plane_model = [exp_normal[0], exp_normal[1], exp_normal[2], -exp_dist]

            # Find inliers for this expected plane
            inliers_relative = self._find_inliers(remaining_pcd, plane_model, self.ransac_threshold)

            if len(inliers_relative) >= min_seed_inliers:
                # Accept this expected wall
                inliers_original = remaining_indices[inliers_relative]

                # Classify as horizontal/vertical/oblique
                normal_vector = np.array(exp_normal)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                dot_product = np.abs(np.dot(normal_vector, self.gravity_vector))

                if dot_product > self.horizontal_cos_tolerance:
                    # Horizontal (shouldn't happen for walls, but check anyway)
                    pass  # We'll handle this later
                elif dot_product < self.vertical_cos_cutoff:
                    # Vertical - this is what we expect for walls
                    seeded_planes.append((plane_model, inliers_original))
                    console.log(f"[green]✓ Seeded wall {i + 1}[/green]: {len(inliers_original)} inliers")

                    # Remove inliers from remaining point cloud
                    remaining_pcd = remaining_pcd.select_by_index(inliers_relative, invert=True)
                    remaining_indices = np.delete(remaining_indices, inliers_relative)

        # 3. DISCOVERY PHASE: Run normal RANSAC on remaining points
        console.log(f"[blue]Discovery phase: RANSAC on {len(remaining_pcd.points)} remaining points[/blue]")

        discovered_h = []
        discovered_v = []
        discovered_o = []

        while len(remaining_pcd.points) > self.min_plane_points:
            plane_model, inliers_relative = remaining_pcd.segment_plane(
                distance_threshold=self.ransac_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.ransac_iterations
            )

            if len(inliers_relative) < self.min_plane_points:
                break

            inliers_original = remaining_indices[inliers_relative]

            # Classify discovered plane
            normal_vector = plane_model[:3]
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            dot_product = np.abs(np.dot(normal_vector, self.gravity_vector))

            if dot_product > self.horizontal_cos_tolerance:
                if plane_model[2] < 0:
                    plane_model = [-x for x in plane_model]
                discovered_h.append((plane_model, inliers_original))
                console.log(f"[blue]Discovered Horizontal[/blue]: {len(inliers_original)} points")
            elif dot_product < self.vertical_cos_cutoff:
                discovered_v.append((plane_model, inliers_original))
                console.log(f"[yellow]Discovered Vertical[/yellow]: {len(inliers_original)} points")
            else:
                discovered_o.append((plane_model, inliers_original))
                console.log(f"[magenta]Discovered Oblique[/magenta]: {len(inliers_original)} points")

            # Remove inliers
            remaining_pcd = remaining_pcd.select_by_index(inliers_relative, invert=True)
            remaining_indices = np.delete(remaining_indices, inliers_relative)

        outlier_indices = remaining_indices

        # 4. Apply NMS to all planes
        all_h = discovered_h
        all_v = seeded_planes + discovered_v
        all_o = discovered_o

        console.log(f"Before NMS: {len(all_h)}H, {len(all_v)}V, {len(all_o)}O")

        all_h = self._non_maximum_suppression(all_h)
        all_v = self._non_maximum_suppression(all_v)
        all_o = self._non_maximum_suppression(all_o)

        console.log(f"After NMS: {len(all_h)}H, {len(all_v)}V, {len(all_o)}O")

        # 5. Validate vertical planes against expected walls
        validated_v = []
        unmatched_v = []

        for plane_model, indices in all_v:
            matched = False

            for exp_normal, exp_dist in expected_walls:
                if self._planes_compatible(plane_model, exp_normal, exp_dist,
                                           angular_tolerance, distance_tolerance):
                    # Refine toward expected
                    refined_model = self._refine_toward_expected(
                        plane_model, exp_normal, exp_dist, alpha=refine_alpha
                    )
                    validated_v.append((refined_model, indices))
                    matched = True
                    console.log(f"[cyan]Validated plane[/cyan]: refined toward model")
                    break

            if not matched:
                unmatched_v.append((plane_model, indices))
                console.log(f"[yellow]Unmatched plane[/yellow]: novel geometry")

        console.log(f"Final: {len(validated_v)} validated, {len(unmatched_v)} unmatched")

        return all_h, validated_v, unmatched_v, all_o, outlier_indices

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
        v_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0]]  # Red, Green, Yellow, Cyan
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

    # ========== SOFT PRIOR METHODS ==========

    def _get_expected_walls(self, particle):
        """
        Get expected wall plane equations from particle.
        Returns list of (normal, distance) for 4 walls in world frame.
        """
        import math
        if particle is None:
            return None

        L, W = particle.length, particle.width
        x, y, theta = particle.x, particle.y, particle.theta

        c, s = math.cos(theta), math.sin(theta)

        # Four walls in local frame: +x, -x, +y, -y
        # Format: (normal_x, normal_y, distance_from_origin)
        walls_local = [
            (1.0, 0.0, L / 2),  # +x wall
            (-1.0, 0.0, L / 2),  # -x wall
            (0.0, 1.0, W / 2),  # +y wall
            (0.0, -1.0, W / 2),  # -y wall
        ]

        walls_world = []
        for (nx_l, ny_l, d_l) in walls_local:
            # Rotate normal to world frame
            nx_w = c * nx_l - s * ny_l
            ny_w = s * nx_l + c * ny_l

            # Wall position in world frame
            wx_l = nx_l * d_l
            wy_l = ny_l * d_l
            wx_w = c * wx_l - s * wy_l + x
            wy_w = s * wx_l + c * wy_l + y

            # Distance from origin in world frame
            d_w = nx_w * wx_w + ny_w * wy_w

            walls_world.append(([nx_w, ny_w, 0.0], abs(d_w)))

        return walls_world

    def _planes_compatible(self, plane_model, expected_normal, expected_dist,
                           angular_tolerance, distance_tolerance):
        """
        Check if detected plane is compatible with expected wall.

        Args:
            plane_model: [A, B, C, D] from RANSAC
            expected_normal: [nx, ny, nz] expected normal
            expected_dist: expected distance from origin
            angular_tolerance: max angle difference in degrees
            distance_tolerance: max distance difference in meters
        """
        # Extract and normalize detected normal
        detected_normal = np.array(plane_model[:3])
        detected_normal = detected_normal / np.linalg.norm(detected_normal)

        # Normalize expected normal
        expected_normal = np.array(expected_normal)
        expected_normal = expected_normal / np.linalg.norm(expected_normal)

        # Check angular alignment (allow opposite directions)
        dot = abs(np.dot(detected_normal, expected_normal))
        angle_deg = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))

        if angle_deg > angular_tolerance:
            return False

        # Check distance alignment
        detected_dist = abs(plane_model[3])
        dist_diff = abs(detected_dist - expected_dist)

        if dist_diff > distance_tolerance:
            return False

        return True

    def _refine_toward_expected(self, plane_model, expected_normal, expected_dist, alpha=0.7):
        """
        Refine detected plane toward expected plane.

        Args:
            plane_model: [A, B, C, D] detected plane
            expected_normal: [nx, ny, nz] expected normal
            expected_dist: expected distance
            alpha: blend factor (1.0=fully expected, 0.0=fully detected)

        Returns:
            Refined plane model [A', B', C', D']
        """
        # Blend normals
        detected_normal = np.array(plane_model[:3])
        detected_normal = detected_normal / np.linalg.norm(detected_normal)

        expected_normal = np.array(expected_normal)
        expected_normal = expected_normal / np.linalg.norm(expected_normal)

        # Ensure same direction (not opposite)
        if np.dot(detected_normal, expected_normal) < 0:
            expected_normal = -expected_normal

        refined_normal = (1 - alpha) * detected_normal + alpha * expected_normal
        refined_normal = refined_normal / np.linalg.norm(refined_normal)

        # Blend distances
        detected_dist = plane_model[3]
        refined_dist = (1 - alpha) * detected_dist + alpha * expected_dist

        # Ensure correct sign
        if detected_dist < 0:
            refined_dist = -abs(refined_dist)
        else:
            refined_dist = abs(refined_dist)

        return [refined_normal[0], refined_normal[1], refined_normal[2], refined_dist]

    def _find_inliers(self, pcd, plane_model, threshold):
        """
        Find inliers for a given plane model.

        Args:
            pcd: Point cloud
            plane_model: [A, B, C, D] plane equation coefficients
            threshold: Distance threshold for inliers

        Returns:
            Array of inlier indices
        """
        import numpy as np

        points = np.asarray(pcd.points)
        A, B, C, D = plane_model

        # Compute distances: |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
        numerator = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
        denominator = np.sqrt(A ** 2 + B ** 2 + C ** 2)
        distances = numerator / denominator

        # Find points within threshold
        inliers = np.where(distances < threshold)[0]

        return inliers
    def get_wall_inlier_points_with_prior(self, pcd, particle, **kwargs):
        """
        Get wall inlier points using prior-validated planes.
        Returns points from both validated and unmatched vertical planes.
        """
        _, validated_v, unmatched_v, _, _ = self.detect_with_prior(pcd, particle, **kwargs)

        all_planes = validated_v + unmatched_v
        if not all_planes:
            return None

        all_indices = []
        for _, indices in all_planes:
            all_indices.extend(indices)

        if not all_indices:
            return None

        points = np.asarray(pcd.points)
        return points[all_indices]