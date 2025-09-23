
try:
    import time
    import torch
    import numpy as np
    from torch.autograd.functional import hessian
    import torch.nn as nn
    #import torch.optim as optim
    from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
    from pytorch3d.loss import chamfer_distance, point_mesh_edge_distance, point_mesh_face_distance
    from pytorch3d.structures import Pointclouds, Meshes
    from pytorch3d.ops import iterative_closest_point, knn_points, sample_points_from_meshes
    #import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
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
    def __init__(self, img_size: int, device="cuda"):
        self.device = device
        self.img_size = img_size
        R = axis_angle_to_matrix(torch.tensor([0, 1, 0], dtype=torch.float32, device=device) * -torch.pi/2.0).unsqueeze(0).to(device)
        T = torch.tensor([-3, -1.2, 3], dtype=torch.float32).unsqueeze(0).to(device)
        self.py_cam = FoVPerspectiveCameras(fov=60.0, R=R, T=T, device=device)

class Rasterizer:
    def __init__(self, cam: Cameras, device="cuda"):
        self.device = device
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
class RoomModel(nn.Module):
    '''
    Diferentiable fridge Model with 9 parameters: x, y, z, a, b, c, w, d, h
    It can render the fridge as a mesh or as sampled points seen from a camera.
    '''

    def __init__(self, init_params, timestamp, height=2.5, device="cuda"):
        super().__init__()
        self.cam = Cameras(128, device=device)
        self.rasterizer = Rasterizer(self.cam, device=device)
        self.device = device

        # Corners are 2D points [N, 2] defining the floor plan
        self.corners = torch.tensor(init_params, dtype=torch.float32, device=device)
        self.height = height  # Room height in meters
        self.creation_timestamp = timestamp


        # # Define each parameter separately as a trainable nn.Parameter
        # self.x = nn.Parameter(torch.tensor(init_params[0], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.y = nn.Parameter(torch.tensor(init_params[1], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.z = nn.Parameter(torch.tensor(init_params[2], dtype=torch.float64, requires_grad=True, device=self.device))
        #
        # self.a = nn.Parameter(torch.tensor(init_params[3], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.b = nn.Parameter(torch.tensor(init_params[4], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.c = nn.Parameter(torch.tensor(init_params[5], dtype=torch.float64, requires_grad=True, device=self.device))
        #
        # self.w = nn.Parameter(torch.tensor(init_params[6], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.d = nn.Parameter(torch.tensor(init_params[7], dtype=torch.float64, requires_grad=True, device=self.device))
        # self.h = nn.Parameter(torch.tensor(init_params[8], dtype=torch.float64, requires_grad=True, device=self.device))
        #
        # self.attributes = ['x', 'y', 'z', 'a', 'b', 'c', 'w', 'd', 'h'] # List of attributes for loop access

    def forward(self) -> Meshes:
        """
        Compute a mesh from the 2D corners of the room (walls only, no floor/ceiling)
        """
        num_corners = len(self.corners)

        # Create 3D vertices by adding bottom and top points for each corner
        bottom_verts = torch.cat([self.corners,
                                  torch.full((num_corners, 1), 0.0, device=self.device)], dim=1)
        top_verts = torch.cat([self.corners,
                               torch.full((num_corners, 1), self.height, device=self.device)], dim=1)
        verts = torch.cat([bottom_verts, top_verts])

        # Create faces for walls (quadrilaterals between each pair of corners)
        faces = []
        for i in range(num_corners):
            next_i = (i + 1) % num_corners

            # Create two triangles for each wall quad
            # Bottom triangle
            faces.append([i, next_i, num_corners + i])
            # Top triangle
            faces.append([num_corners + i, next_i, num_corners + next_i])

        faces_tensor = torch.tensor(faces, dtype=torch.int32, device=self.device)

        # Create a Meshes object
        my_mesh = Meshes(
            verts=[verts],
            faces=[faces_tensor]
        )
        return my_mesh

    def get_corners(self):
        return self.corners.cpu().numpy()

    def set_corners(self, corners):
        self.corners = torch.tensor(corners, dtype=torch.float32, device=self.device)


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
        min_dists_sq, closest_face_indices = total_dists_sq.min(dim=1) # Shape (P,)

        # --- 7. Filtering ---
        # Calculate the actual distances and create the mask.
        # Add clamp(min=0) before sqrt for robustness against tiny negative values due to precision.
        min_dists = torch.sqrt(torch.clamp(min_dists_sq, min=0.0)) # Shape (P,)
        unexplained = min_dists > threshold # Shape (P,)
        explained = min_dists <= threshold  # Shape (P,)

        # Count explained points per face
        explained_points_mask = explained
        closest_faces_for_explained = closest_face_indices[explained_points_mask]
        face_counts = torch.bincount(
            closest_faces_for_explained,
            minlength=faces.shape[0]
        ).to(device=points.device)

        # Return the subset of points satisfying the condition.
        return points[unexplained], points[explained], face_counts

    def is_mesh_closed_by_corners(self, polyline):
        """
        Check if a polyline forms a closed loop by comparing first/last corners.

        Args:
            polyline: List of corner positions [(x1,y1), (x2,y2), ...]

        Returns:
            bool: True if closed (first == last), False otherwise
        """
        corner_poses = np.array([corner[1] for corner in polyline])
        return np.allclose(corner_poses[0], corner_poses[-1])

    def is_point_inside_room(self, corners_2d: np.ndarray, point: tuple[float, float] = (0., 0.)) -> bool:
        """
        Check if (0, 0) is inside the 2D floor plan of the room using ray-casting.

        Args:
            corners_2d: Array of shape (N, 2) representing room corners in order.
            point: The test point (default: (0, 0)).

        Returns:
            bool: True if inside, False otherwise.
        """
        x, y = point
        n = len(corners_2d)
        inside = False

        for i in range(n):
            x1, y1 = corners_2d[i]
            x2, y2 = corners_2d[(i + 1) % n]

            # Check if point is exactly on a vertex
            if (x == x1) and (y == y1):
                return True

            # Check if point is on the edge (optional)
            if min(y1, y2) < y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if y1 == y2 or x <= x_intersect:
                        inside = not inside

        return inside

    def project_points_with_pytorch3d(self, points: torch.Tensor, threshold: float = 0.1, num_samples: int = 10000):
        """
        Proyecta cada punto sobre la malla y colorea las caras según cuántos puntos están cerca de ellas.

        Args:
            points: (P, 3)
            threshold: distancia máxima para considerar un punto como explicado
            num_samples: cantidad de puntos a muestrear sobre la malla

        Returns:
            unexplained_points: puntos no explicados por la malla
            mesh_colored: malla con colores por cara en función de la densidad de puntos cercanos
            face_counts: conteo de puntos por cara
        """
        mesh = self.forward()
        device = points.device

        # Paso 1: Muestrear puntos y estimar sus caras
        sampled_points, sampled_face_ids = self.sample_points_with_face_ids(mesh, num_samples=num_samples)

        # Paso 2: Buscar el punto muestrado más cercano a cada punto de entrada
        knn = knn_points(points[None], sampled_points[None], K=1)
        distances = knn.dists[0, :, 0].sqrt()  # (P,)
        idxs = knn.idx[0, :, 0]  # (P,)

        # Paso 3: Filtrar puntos no explicados
        keep_mask = distances <= threshold
        unexplained_points = points[keep_mask]

        # Paso 4: Contar puntos explicados por cara
        close_mask = distances < threshold
        face_ids = sampled_face_ids[idxs[close_mask]]  # caras más cercanas a cada punto dentro del umbral

        faces = mesh.faces_packed()
        F = faces.shape[0]
        face_counts = torch.bincount(face_ids, minlength=F).float()

        # Paso 5: Calcular color (rojo = más puntos, azul = menos)
        max_count = face_counts.max()
        heat = face_counts / (max_count + 1e-8)
        face_colors = torch.stack([heat, 1.0 - heat, torch.zeros_like(heat)], dim=1)  # RGB

        # Paso 6: Propagar color a vértices
        verts = mesh.verts_packed()
        vert_colors = torch.zeros_like(verts)
        counts_per_vertex = torch.zeros(verts.shape[0], device=device)

        for i in range(F):
            for j in range(3):
                vidx = faces[i, j]
                vert_colors[vidx] += face_colors[i]
                counts_per_vertex[vidx] += 1

        vert_colors = vert_colors / (counts_per_vertex.unsqueeze(-1) + 1e-8)

        # Paso 7: Crear malla coloreada
        mesh_colored = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=[vert_colors]))

        return unexplained_points, mesh_colored, face_counts

    def sample_points_with_face_ids(self, mesh: Meshes, num_samples: int):
        """
        Muestra puntos sobre la malla y estima a qué cara pertenece cada punto
        usando KNN con el centroide de cada triángulo.

        Devuelve:
            sampled_points: (N, 3)
            face_ids: (N,)
        """
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        tris = verts[faces]  # (F, 3, 3)
        tris_center = tris.mean(dim=1)  # (F, 3)

        sampled_points = sample_points_from_meshes(mesh, num_samples=num_samples)[0]  # (N, 3)

        # Buscar centroide más cercano para estimar la cara original
        knn = knn_points(sampled_points[None], tris_center[None], K=1)
        face_ids = knn.idx[0, :, 0]  # (N,)

        return sampled_points, face_ids

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
    def loss_function(self, real_points):
        """ Compute the loss for optimization """

        # Compute mesh from updated model
        my_mesh = self.forward_visible_faces()

        # Compute loss terms
        loss_1 = point_mesh_edge_distance(my_mesh, Pointclouds([real_points]))  # Edge distance
        loss_2 = point_mesh_face_distance(my_mesh, Pointclouds([real_points]))  # Face distance
        loss_3 = self.prior_size(0.6, 0.6, 1.8)
        loss_4 = self.prior_on_floor()
        loss_5 = self.prior_aligned()
        loss_6 = self.prior_position_by_mass_center(real_points)

        total_loss = loss_1 + loss_2 + loss_3 + 2 * loss_4 + 2*loss_5 + 0.5*loss_6
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

    def prior_position_by_mass_center(self, points: torch.Tensor):
        # compute the mass center of the point cloud
        mass_center = torch.mean(points, dim=0)
        # displace the mass_center backwards by the prior width and depth of the fridge
        x0 = mass_center[0]
        y0 = mass_center[1]
        return (self.x - x0)**2 + (self.y - y0)**2





