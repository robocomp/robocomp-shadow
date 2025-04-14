import sys

try:
    import time
    import torch
    import numpy as np
    from torch.autograd.functional import hessian
    import torch.nn as nn
    from pytorch3d.loss import chamfer_distance, point_mesh_edge_distance, point_mesh_face_distance
    from pytorch3d.structures import Pointclouds, Meshes
    from pytorch3d.ops import iterative_closest_point
    from pytorch3d.renderer import (
        PerspectiveCameras,
        MeshRasterizer,
        HardPhongShader,
        MeshRenderer,
        RasterizationSettings,
        PointLights,
        look_at_view_transform, FoVPerspectiveCameras, look_at_rotation, TexturesVertex)
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, Transform3d, euler_angles_to_matrix
    from torch.func import functional_call
except ModuleNotFoundError as e:
    print("Required module not found. Ensure PyTorch and PyTorch3D are installed.")
    raise e

class Cameras:
    def __init__(self, robot_rot, robot_trans, device="cuda"):
        # +X is right, +Y is up, +Z is forward
        # we want to rotate the camera to look at the fridge. Room is +X right, +Y front, +Z up
        # so we need the camera Z+ to be aligned with the robot Y+
        # to do that rotate the camera by -90 degrees around the X axis
        # and then rotate it by the robot rotation around the Y axis
        # Step 1: Rotate by -90 degrees around the X-axis
        R1 = axis_angle_to_matrix(torch.tensor([1, 0, 0], dtype=torch.float32, device=device) * -torch.pi / 2.0).unsqueeze(0).to(device)
        # Step 2: Rotate by the robot's rz angle around the Z-axis
        R2 = euler_angles_to_matrix(torch.tensor([0, 0, -robot_rot[2]], dtype=torch.float32, device=device).unsqueeze(0), convention="XYZ").to(device)
        # Combine the rotations
        R = torch.matmul(R2, R1)
        #T = torch.tensor([-3, -1.2, 3], dtype=torch.float32).unsqueeze(0).to(device)
        T = torch.tensor(robot_trans, dtype=torch.float32).unsqueeze(0).to(device)
        self.py_cam = FoVPerspectiveCameras(fov=60.0, R=R, T=T, device=device)

class Rasterizer:
    def __init__(self, cam: Cameras):
        self.raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        self.rasterizer = MeshRasterizer(cameras=cam.py_cam, raster_settings=self.raster_settings)

# =====================================================================================
# Fridge Model
# ======================================================================================
class FridgeModel(nn.Module):
    '''
    Diferentiable fridge Model with 9 parameters: x, y, z, a, b, c, w, d, h
    It can render the fridge as a mesh or as sampled points seen from a camera.
    '''

    def __init__(self, init_params, robot_rot, robot_trans, device="cuda"):
        super().__init__()
        self.cam = Cameras(robot_rot, robot_trans, device)
        self.rasterizer = Rasterizer(self.cam)
        self.device = device

        # Define each parameter separately as a trainable nn.Parameter
        self.x = nn.Parameter(torch.tensor(init_params[0], dtype=torch.float64, requires_grad=True, device=self.device))
        self.y = nn.Parameter(torch.tensor(init_params[1], dtype=torch.float64, requires_grad=True, device=self.device))
        self.z = nn.Parameter(torch.tensor(init_params[2], dtype=torch.float64, requires_grad=True, device=self.device))

        self.a = nn.Parameter(torch.tensor(init_params[3], dtype=torch.float64, requires_grad=True, device=self.device))
        self.b = nn.Parameter(torch.tensor(init_params[4], dtype=torch.float64, requires_grad=True, device=self.device))
        self.c = nn.Parameter(torch.tensor(init_params[5], dtype=torch.float64, requires_grad=True, device=self.device))

        self.w = nn.Parameter(torch.tensor(init_params[6], dtype=torch.float64, requires_grad=True, device=self.device))
        self.d = nn.Parameter(torch.tensor(init_params[7], dtype=torch.float64, requires_grad=True, device=self.device))
        self.h = nn.Parameter(torch.tensor(init_params[8], dtype=torch.float64, requires_grad=True, device=self.device))

        self.attributes = ['x', 'y', 'z', 'a', 'b', 'c', 'w', 'd', 'h'] # List of attributes for loop access
    
    def forward(self) -> Meshes:
        """
            Compute a mesh from the 3D corners of the oriented bounding box
        """

        # nominal corners of the fridge
        base_corners = torch.stack([
            torch.tensor([-0.5, -0.5, -0.5], device=self.device),
            torch.tensor([0.5, -0.5, -0.5], device=self.device),
            torch.tensor([0.5, 0.5, -0.5], device=self.device),
            torch.tensor([-0.5, 0.5, -0.5], device=self.device),
            torch.tensor([-0.5, -0.5, 0.5], device=self.device),
            torch.tensor([0.5, -0.5, 0.5], device=self.device),
            torch.tensor([0.5, 0.5, 0.5], device=self.device),
            torch.tensor([-0.5, 0.5, 0.5], device=self.device)
        ])

        # nominal faces of the fridge
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6], [7, 3, 0], [4, 7, 0], [1, 2, 6], [1, 6, 5]], dtype=torch.int32, device=self.device)


        # Transform the corners to the current model scale, position and orientation
        t = (Transform3d(device="cuda").scale(self.w, self.d, self.h)
                                        .rotate(euler_angles_to_matrix(torch.stack([self.a, self.b, self.c]), "ZYX"))
                                        .translate(self.x, self.y, self.z))

        verts = t.transform_points(base_corners)

        # Create a Meshes object
        my_mesh = Meshes(
            verts=[verts],
            faces=[faces]
            #textures=textures
        )
        return my_mesh

    def forward_visible_faces(self) -> Meshes:
        """
        Get a mask of visible faces using rasterization.
        Returns a set of face indices that are visible in the depth buffer.
        """
        my_mesh = self.forward()
        fragments = self.rasterizer.rasterizer(my_mesh)  # Rasterize the mesh
        visible_faces = fragments.pix_to_face[..., 0]  # Get face indices
        visible_faces = visible_faces.unique()  # Get unique face indices
        visible_faces = visible_faces[visible_faces >= 0]  # Remove -1 (background)

        # Create a new mesh containing only the visible faces.
        faces = my_mesh.faces_packed()  # Get all faces
        verts = my_mesh.verts_packed()  # Get all vertices

        # Select only the visible faces
        visible_faces = faces[visible_faces]

        # Create a new mesh with only visible faces
        filtered_mesh = Meshes(verts=[verts], faces=[visible_faces])
        return filtered_mesh

    def rasterize_visible_faces(self):
        """
        Rasterize the visible faces of the mesh to a binary image.
        """
        # Rasterize the mesh
        my_mesh = self.forward()
        fragments = self.rasterizer.rasterizer(my_mesh)

        # Extract the depth map
        rendered_depth = fragments.zbuf[..., 0]  # (1, H, W)

        # Create pixel coordinates as a 2D grid
        height, width = rendered_depth.shape[1:]
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing="xy"
        )
        x = -x  # Flip x to match the camera frame
        y = -y

        # Mask out pixels with no valid depth
        valid_mask = rendered_depth[0] > 0

        # Get the pixel coordinates for valid depth values
        pixel_coords = torch.stack([x, y], dim=-1)[valid_mask]  # (N, 2)

        # Get the corresponding depth values
        depth_values = rendered_depth[0][valid_mask]  # (N,)

        # Gather coordinates and depth
        points_ndc = torch.cat([pixel_coords,  # (N, 2)
                depth_values.unsqueeze(-1),  # (N, 1)
            ], dim=-1)  # (N, 3)

        # add batch dimension
        points_ndc_batched = points_ndc.unsqueeze(0)  # (1, N, 3)

        # Unproject to world coordinates
        w_points = self.cam.py_cam.unproject_points(points_ndc_batched, world_coordinates=True)

        return w_points.squeeze(0)

    def remove_explained_points(self, points: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Removes points whose minimum distance to any face in the mesh is less than or equal to the threshold.
        Uses geometrically correct point-to-triangle distance calculation, including edges.
        Returns the points not explained by the mesh (distance > threshold).
        """

        # Helper function for squared point-to-segment distance (vectorized)
        def point_segment_distance_sq(
                points: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """
            Computes squared Euclidean distance from points to line segments.
            Args:
                points: Points tensor, shape (..., P, 1, 3) or broadcasts.
                seg_start: Segment start points, shape (..., 1, F, 3) or broadcasts.
                seg_end: Segment end points, shape (..., 1, F, 3) or broadcasts.
                eps: Small value for numerical stability.
            Returns:
                Squared distances, shape broadcastable from inputs, typically (..., P, F).
            """
            # Ensure broadcasting between points (P,1,3) and segments (1,F,3)
            # Vector from segment start to points: results in shape (P, F, 3)
            vec_ap = points - seg_start
            # Vector defining the segment: results in shape (1, F, 3)
            vec_ab = seg_end - seg_start

            # Squared length of the segment vector
            # Shape: (1, F)
            ab_sq = (vec_ab * vec_ab).sum(-1)

            # Project points onto the line defined by the segment
            # t = dot(vec_ap, vec_ab) / dot(ab, ab)
            # Element-wise product broadcasts (P, F, 3) * (1, F, 3) -> (P, F, 3)
            # Sum over last dim -> Shape: (P, F)
            ap_dot_ab = (vec_ap * vec_ab).sum(-1)

            # Add epsilon to avoid division by zero for zero-length segments (degenerate edges)
            # Division broadcasts (P, F) / (1, F) -> (P, F)
            t = ap_dot_ab / (ab_sq + eps)

            # Clamp t to the range [0, 1] to find the closest point on the segment
            t_clamped = torch.clamp(t, 0.0, 1.0)  # Shape (P, F)

            # Calculate the closest point on the segment: proj = A + t_clamped * AB
            # Broadcasting: seg_start(1,F,3) + (t(P,F,1) * vec_ab(1,F,3) -> (P,F,3)) -> (P,F,3)
            proj = seg_start + t_clamped.unsqueeze(-1) * vec_ab  # Shape (P, F, 3)

            # Calculate the squared distance from the original points to the projection on the segment
            # Broadcasting: points(P,1,3) - proj(P,F,3) -> (P,F,3) -> sum -> (P,F)
            dist_sq = ((points - proj) ** 2).sum(-1)  # Shape (P, F)
            return dist_sq

        # --- 1. Mesh Data Preparation ---
        mesh = self.forward()
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        # Handle empty mesh or points
        if faces.numel() == 0 or verts.numel() == 0:
            print("Warning: Mesh is empty, returning all input points.")
            return points
        if points.shape[0] == 0:
            return points

        tris = verts[faces]  # Shape: (F, 3, 3)

        # --- 2. Prepare Tensors for Vectorized Computation ---
        P = points.shape[0]
        F = tris.shape[0]
        # Expand points for broadcasting against faces. Shape: (P, 1, 3) -> (P, F, 3) below needs explicit broadcast
        points_p_dim = points[:, None, :] # Shape (P, 1, 3)
        # Vertices of triangles, shape (F, 3) -> needs broadcasting for P dimension
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2] # Shapes (F, 3)

        # --- 3. Barycentric Coordinates Calculation ---
        # Edge vectors
        v0v1 = v1 - v0 # Shape (F, 3)
        v0v2 = v2 - v0 # Shape (F, 3)
        # Vector from triangle origin (v0) to each point P
        # Broadcasting: points (P, 1, 3) - v0 (1, F, 3) -> pvec (P, F, 3)
        pvec = points_p_dim - v0[None, :, :] # Shape: (P, F, 3)

        # Dot products for barycentric coords
        d00 = (v0v1 * v0v1).sum(-1) # Shape (F,)
        d01 = (v0v1 * v0v2).sum(-1) # Shape (F,)
        d11 = (v0v2 * v0v2).sum(-1) # Shape (F,)
        # Broadcasting: pvec (P, F, 3) * v0v1/v0v2 (1, F, 3) -> sum -> (P, F)
        d20 = (pvec * v0v1[None, :, :]).sum(-1) # Shape (P, F)
        d21 = (pvec * v0v2[None, :, :]).sum(-1) # Shape (P, F)

        # Denominator (related to squared area * 2)
        denom = d00 * d11 - d01 * d01 # Shape (F,)
        # Mask for degenerate triangles (denom is near zero)
        degenerate_mask = torch.abs(denom) < 1e-8 # Shape (F,)

        # Avoid division by zero for degenerate triangles
        # The results for these triangles won't be used in the 'inside' case anyway.
        denom_safe = torch.where(degenerate_mask, torch.ones_like(denom), denom)

        # Calculate barycentric coords v and w
        v = (d11 * d20 - d01 * d21) / denom_safe # Shape (P, F)
        w = (d00 * d21 - d01 * d20) / denom_safe # Shape (P, F)
        u = 1.0 - v - w # Shape (P, F)

        # Check if the projection is inside or on the boundary.
        # Use a small tolerance (e.g., -1e-5) for numerical stability at edges/vertices.
        # Exclude degenerate triangles from being 'inside'.
        inside = ((u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5) &
                  (~degenerate_mask[None, :].expand(P, F))) # Shape (P, F)

        # --- 4. Distance Calculation: Case 1 (Projection Inside Triangle) ---
        # Calculate squared orthogonal distance to the triangle's plane.
        n = torch.linalg.cross(v0v1, v0v2, dim=-1) # Shape (F, 3)
        n_norm_sq = (n * n).sum(-1, keepdim=True) # Shape (F, 1)
        # Project pvec onto normal n: dot(pvec, n)
        pvec_dot_n = (pvec * n[None, :, :]).sum(-1) # Shape (P, F)
        # Squared distance = (dot(pvec, n) / ||n||)^2 = dot(pvec, n)^2 / ||n||^2
        # Avoid division by zero for degenerate triangles (where n_norm_sq is near zero)
        plane_dists_sq = (pvec_dot_n ** 2) / (n_norm_sq.squeeze(-1)[None, :] + 1e-8) # Shape (P, F)

        # --- 5. Distance Calculation: Case 2 (Projection Outside Triangle or Degenerate) ---
        # Calculate the squared distance to the closest point on each of the three edges.
        # We need to broadcast points (P,1,3) with segment starts/ends (1,F,3)
        # The helper function handles the necessary broadcasting internally.
        # We pass points_p_dim (P,1,3) and broadcasted v0/v1/v2 (1,F,3).
        dist_sq_01 = point_segment_distance_sq(points_p_dim, v0[None,:,:], v1[None,:,:]) # Shape (P, F)
        dist_sq_12 = point_segment_distance_sq(points_p_dim, v1[None,:,:], v2[None,:,:]) # Shape (P, F)
        dist_sq_20 = point_segment_distance_sq(points_p_dim, v2[None,:,:], v0[None,:,:]) # Shape (P, F)

        # Minimum squared distance to any of the three edges
        min_edge_dist_sq = torch.min(torch.stack([dist_sq_01, dist_sq_12, dist_sq_20], dim=-1), dim=-1).values # Shape (P, F)

        # --- 6. Combine Distances and Final Selection ---
        # Select the appropriate squared distance based on the 'inside' mask.
        # If 'inside', use the squared distance to the plane.
        # Otherwise (outside or degenerate), use the minimum squared distance to an edge.
        total_dists_sq = torch.where(inside, plane_dists_sq, min_edge_dist_sq) # Shape (P, F)

        # Find the minimum squared distance from each point to ANY face component in the mesh.
        min_dists_sq, _ = total_dists_sq.min(dim=1) # Shape (P,)

        # --- 7. Filtering ---
        # Calculate the actual distances and create the mask.
        # Add clamp(min=0) before sqrt for robustness against tiny negative values due to precision.
        min_dists = torch.sqrt(torch.clamp(min_dists_sq, min=0.0)) # Shape (P,)
        keep_mask = min_dists > threshold # Shape (P,)

        # Return the subset of points satisfying the condition.
        return points[keep_mask]

    # def remove_explained_points_2(self, points: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    #     """
    #     Removes points whose distance to the closest face component (approximated)
    #     in the mesh is less than or equal to the threshold.
    #     Returns the points not explained by the mesh (distance > threshold).
    #     NOTE: This implementation approximates distance for points outside the triangle projection
    #           by using the distance to the nearest vertex, which may not be the true closest point
    #           (which could be on an edge).
    #     """
    #     # --- 1. Mesh Data Preparation ---
    #     # Geometry Reasoning: Access the fundamental geometric components of the mesh:
    #     # vertices (points in 3D space) and faces (sets of 3 vertex indices defining triangles).
    #     mesh = self.forward()  # Get the mesh object (assuming self.forward provides it)
    #     verts = mesh.verts_packed()  # Get all vertex coordinates, shape (V, 3)
    #     faces = mesh.faces_packed()  # Get all face indices, shape (F, 3)
    #
    #     # Geometry Reasoning: Create triangle primitives. Use the face indices to look up
    #     # the actual 3D coordinates of the vertices for each triangle.
    #     # tris now holds F triangles, each defined by the coordinates of its 3 vertices.
    #     tris = verts[faces]  # Shape: (F, 3, 3) -> (NumFaces, VerticesPerFace, CoordsPerVertex)
    #
    #     # --- 2. Prepare Tensors for Vectorized Computation ---
    #     # Geometry Reasoning: To efficiently calculate distances between each point and *every* triangle
    #     # without explicit loops, we expand the points tensor. Each point is replicated F times.
    #     # This allows direct vectorized operations between the i-th point and all F triangles.
    #     P = points.shape[0]  # Number of input points
    #     F = tris.shape[0]  # Number of faces (triangles) in the mesh
    #     # Expand points from (P, 3) to (P, F, 3). The point `points[i]` is now present
    #     # at `points_exp[i, j, :]` for all j from 0 to F-1.
    #     points_exp = points[:, None, :].expand(P, F, 3)
    #
    #     # Geometry Reasoning: Extract the vertices of each triangle for easier access.
    #     # v0, v1, v2 are the first, second, and third vertices of each triangle, respectively.
    #     v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]  # Shape of each: (F, 3)
    #
    #     # --- 3. Barycentric Coordinates Calculation ---
    #     # Geometry Reasoning: This section determines where the orthogonal projection of each point P
    #     # onto the plane of each triangle lies relative to that triangle's boundaries.
    #     # This is achieved by calculating barycentric coordinates (u, v, w).
    #     # If u, v, w are all >= 0, the projection is inside or on the boundary of the triangle.
    #
    #     # Calculate the two edge vectors defining the basis of the triangle within its plane.
    #     v0v1 = v1 - v0  # Vector from v0 to v1 for each triangle, shape (F, 3)
    #     v0v2 = v2 - v0  # Vector from v0 to v2 for each triangle, shape (F, 3)
    #
    #     # Calculate the vector from the triangle's origin (v0) to each point P.
    #     # Broadcasting occurs here: points_exp (P, F, 3) - v0[None, :, :] (1, F, 3) -> (P, F, 3)
    #     pvec = points_exp - v0[None, :, :]  # Shape: (P, F, 3)
    #
    #     # Compute dot products needed for the barycentric coordinate formula.
    #     # This is essentially solving the linear system `pvec_proj = v * v0v1 + w * v0v2` for v and w.
    #     d00 = (v0v1 * v0v1).sum(-1)  # dot(v0v1, v0v1), shape (F,)
    #     d01 = (v0v1 * v0v2).sum(-1)  # dot(v0v1, v0v2), shape (F,)
    #     d11 = (v0v2 * v0v2).sum(-1)  # dot(v0v2, v0v2), shape (F,)
    #     # dot(pvec, v0v1), requires broadcasting v0v1 from (F, 3) to (1, F, 3)
    #     d20 = (pvec * v0v1[None, :, :]).sum(-1)  # Shape (P, F)
    #     # dot(pvec, v0v2), requires broadcasting v0v2 from (F, 3) to (1, F, 3)
    #     d21 = (pvec * v0v2[None, :, :]).sum(-1)  # Shape (P, F)
    #
    #     # Calculate the denominator for Cramer's rule (or equivalent solution method).
    #     # This is related to the square of the triangle's area. Add epsilon for numerical stability.
    #     denom = d00 * d11 - d01 * d01 + 1e-8  # Shape (F,)
    #
    #     # Solve for barycentric coordinates v and w.
    #     # Broadcasting occurs: (F,) / (F,) applied element-wise across the P dimension.
    #     v = (d11 * d20 - d01 * d21) / denom  # Shape (P, F)
    #     w = (d00 * d21 - d01 * d20) / denom  # Shape (P, F)
    #     # Calculate the third coordinate u using the property u + v + w = 1 (for the projection).
    #     u = 1.0 - v - w  # Shape (P, F)
    #
    #     # Determine if the projection of the point lies inside or on the boundary of the triangle.
    #     inside = (u >= 0) & (v >= 0) & (w >= 0)  # Shape (P, F) boolean mask
    #
    #     # --- 4. Distance Calculation: Case 1 (Projection Inside Triangle) ---
    #     # Geometry Reasoning: If the point's projection is inside the triangle, the closest point
    #     # on the infinite plane of the triangle *is* that projection. The distance is the
    #     # orthogonal distance from the point to the plane.
    #
    #     # Calculate the normal vector of each triangle plane.
    #     n = torch.linalg.cross(v0v1, v0v2, dim=-1)  # Shape (F, 3)
    #     # Normalize the normal vectors (add epsilon for stability).
    #     n = n / (torch.linalg.norm(n, dim=-1, keepdim=True) + 1e-8)  # Shape (F, 3)
    #
    #     # Calculate the orthogonal distance to the plane. Project pvec onto the unit normal n.
    #     # The absolute value gives the distance. Requires broadcasting n.
    #     plane_dists = torch.abs((pvec * n[None, :, :]).sum(-1))  # Shape (P, F)
    #
    #     # --- 5. Distance Calculation: Case 2 (Projection Outside Triangle - APPROXIMATION) ---
    #     # Geometry Reasoning: If the point's projection is outside the triangle, the true closest
    #     # point on the triangle lies on one of its edges or vertices. This implementation *approximates*
    #     # this by finding the distance to the *nearest vertex* of the triangle. This is simpler
    #     # but not always geometrically correct (the closest point could be on an edge).
    #
    #     # Calculate squared Euclidean distance from each point to each vertex of each triangle.
    #     # Broadcasting: points(P, 1, 3) - v0(1, F, 3) -> (P, F, 3) -> sum -> (P, F)
    #     dist_v0_sq = ((points[:, None, :] - v0[None, :, :]) ** 2).sum(-1)  # Shape (P, F)
    #     dist_v1_sq = ((points[:, None, :] - v1[None, :, :]) ** 2).sum(-1)  # Shape (P, F)
    #     dist_v2_sq = ((points[:, None, :] - v2[None, :, :]) ** 2).sum(-1)  # Shape (P, F)
    #
    #     # Find the minimum of the squared distances to the three vertices.
    #     # Stack along a new dimension (-1) -> (P, F, 3) then take min along that dimension.
    #     corner_dists_sq = torch.min(torch.stack([dist_v0_sq, dist_v1_sq, dist_v2_sq], dim=-1), dim=-1).values  # Shape (P, F)
    #
    #     # --- 6. Combine Distances and Final Selection ---
    #     # Geometry Reasoning: Select the appropriate distance based on whether the projection was inside.
    #     # If inside, use the orthogonal distance to the plane.
    #     # If outside, use the (approximated) distance to the nearest corner vertex.
    #     # Note: Need sqrt of corner_dists_sq here.
    #     total_dists = torch.where(inside, plane_dists, torch.sqrt(corner_dists_sq))  # Shape (P, F)
    #
    #     # Geometry Reasoning: Find the minimum distance from each point to *any* of the F faces.
    #     # We take the minimum value across the F dimension (dim=1).
    #     min_dists, _ = total_dists.min(dim=1)  # Shape (P,) - min distance for each point
    #
    #     # --- 7. Filtering ---
    #     # Geometry Reasoning: Keep only those points whose calculated minimum distance
    #     # to the mesh surface (as computed/approximated above) is greater than the threshold.
    #     keep_mask = min_dists > threshold  # Shape (P,) boolean mask
    #     return points[keep_mask]  # Return the subset of points satisfying the condition.

    def hessian_and_covariance(self, points: torch.Tensor)-> [torch.Tensor, torch.Tensor]:
        """
        Computes the Hessian of the loss function and its inverse (covariance matrix).
        
        """
        # Extract parameters as a tuple
        optimized_params = tuple([
            self.x, self.y, self.z,
            self.a, self.b, self.c,
            self.w, self.d, self.h
        ])

        # Ensure parameters require gradients
        for param in optimized_params:
            param.requires_grad_(True)

        # Compute Hessian row by row
        param_count = len(optimized_params)
        hessian_matrix = torch.zeros((param_count, param_count), device=self.device, dtype=torch.float64)

        for i, param_i in enumerate(optimized_params):
            grad_i = torch.autograd.grad(self.loss_function(points), param_i, create_graph=True)[0]
            for j, param_j in enumerate(optimized_params):
                hess_ij = torch.autograd.grad(grad_i, param_j, retain_graph=True, allow_unused=True)[0]
                if hess_ij is not None:
                    hessian_matrix[i, j] = hess_ij

        # Add small regularization to avoid singularity
        hessian_matrix += 1e-6 * torch.eye(param_count, device=hessian_matrix.device)

        # Compute covariance matrix (inverse of Hessian)
        try:
            covariance_matrix = torch.inverse(hessian_matrix)
        except RuntimeError:
            covariance_matrix = torch.zeros_like(hessian_matrix)
            print("Warning: Hessian is singular, returning zero covariance.")

        return hessian_matrix, covariance_matrix

    def print_params(self):
        print("Model Parameters:")
        for name, param in self.named_parameters():
            print(f"    {name} value: {param.item()}")

    def print_grads(self):
        print("Gradients:")
        for name, param in self.named_parameters():
            print(f"    {name} grad: {param.grad}")

    ##########################################################
    ### losses
    ##########################################################
    def loss_function(self, real_points: torch.Tensor):
        """ Compute the loss for optimization """
        # Compute mesh from updated model
        my_mesh = self.forward()
        if my_mesh.num_edges_per_mesh() is None:
            print("Warning: Mesh is empty, returning zero loss.")
            sys.exit()

        # Compute loss terms
        loss_1 = point_mesh_edge_distance(my_mesh, Pointclouds([real_points]))  # Edge distance
        loss_2 = point_mesh_face_distance(my_mesh, Pointclouds([real_points]))  # Face distance
        loss_3 = self.prior_size(0.6, 0.6, 1.8)
        loss_4 = self.prior_on_floor()
        loss_5 = self.prior_aligned()
        #loss_6 = self.prior_position_by_mass_center(real_points)

        total_loss = loss_1 + loss_2 + loss_3 + 2 * loss_4 + 2 * loss_5 # + 0.5 * loss_6
        return total_loss

    def prior_size(self, w=0.6, d=0.6, h=1.8):
        '''
            Compute the difference between w,d,h and the prior values
        '''
        return torch.sum((self.w-w)**2 + (self.d-d)**2 + (self.h-h)**2)

    def prior_on_floor(self):
        # The height of the fridge (at center) has to be the semi-height of the fridge
        return torch.sum((self.z - self.h/2)**2)

    def prior_aligned(self):
        # The fridge is aligned with the room axis: a,b,c = 0
        return torch.sum(self.a**2 + self.b**2 + self.c**2)

    def prior_position_by_mass_center(self, points: torch.Tensor  )-> torch.Tensor:
        """
        Compute the prior on the position of the fridge based on the mass center of the points.
        The mass center is computed as the mean of the points.
        """
        # Compute mass center
        mass_center = torch.mean(points, dim=0)
        # Compute distance to mass center
        dist = torch.sqrt((self.x - mass_center[0])**2 + (self.y - mass_center[1])**2 + (self.z - mass_center[2])**2)
        return dist
