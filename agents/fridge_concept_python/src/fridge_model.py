
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
    from pytorch3d.ops import iterative_closest_point
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
class FridgeModel(nn.Module):
    '''
    Diferentiable fridge Model with 9 parameters: x, y, z, a, b, c, w, d, h
    It can render the fridge as a mesh or as sampled points seen from a camera.
    '''

    def __init__(self, init_params, device="cuda"):
        super().__init__()
        self.cam = Cameras(128, device=device)
        self.rasterizer = Rasterizer(self.cam, device=device)
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
        Removes points whose distance to any face in the mesh is less than the threshold.
        Returns the points not explained by the mesh (distance > threshold).
        """
        mesh = self.forward()
        verts = mesh.verts_packed()  # (V, 3)
        faces = mesh.faces_packed()  # (F, 3)
        tris = verts[faces]  # (F, 3, 3)

        # Expand points to (P, F, 3) and tris to (P, F, 3, 3)
        P = points.shape[0]
        F = tris.shape[0]
        points_exp = points[:, None, :].expand(P, F, 3)
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

        # Compute vectors
        v0v1 = v1 - v0  # (F, 3)
        v0v2 = v2 - v0  # (F, 3)
        pvec = points_exp - v0  # (P, F, 3)

        # Compute dot products
        d00 = (v0v1 * v0v1).sum(-1)  # (F,)
        d01 = (v0v1 * v0v2).sum(-1)
        d11 = (v0v2 * v0v2).sum(-1)
        d20 = (pvec * v0v1[None, :, :]).sum(-1)  # (P, F)
        d21 = (pvec * v0v2[None, :, :]).sum(-1)  # (P, F)

        denom = d00 * d11 - d01 * d01 + 1e-8  # (F,)
        v = (d11 * d20 - d01 * d21) / denom  # (P, F)
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        inside = (u >= 0) & (v >= 0) & (w >= 0)  # (P, F)

        # Distance to plane of triangle
        n = torch.cross(v0v1, v0v2)  # (F, 3)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
        plane_dists = torch.abs((pvec * n[None, :, :]).sum(-1))  # (P, F)

        # Squared Euclidean distance to closest vertex if outside triangle
        dist_v0 = ((points[:, None, :] - v0[None, :, :]) ** 2).sum(-1)
        dist_v1 = ((points[:, None, :] - v1[None, :, :]) ** 2).sum(-1)
        dist_v2 = ((points[:, None, :] - v2[None, :, :]) ** 2).sum(-1)
        corner_dists = torch.min(torch.stack([dist_v0, dist_v1, dist_v2], dim=-1), dim=-1).values  # (P, F)

        # Use plane distance if inside triangle, corner distance otherwise
        total_dists = torch.where(inside, plane_dists, torch.sqrt(corner_dists))  # (P, F)

        # Take min across triangles
        min_dists, _ = total_dists.min(dim=1)  # (P,)

        # Keep only unexplained points
        keep_mask = min_dists > threshold
        return points[keep_mask]
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





