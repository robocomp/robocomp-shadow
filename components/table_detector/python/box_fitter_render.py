
try:
    import time
    import torch
    import numpy as np
    from torch.autograd.functional import hessian
    import torch.nn as nn
    import torch.optim as optim
    from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
    from pytorch3d.loss import chamfer_distance, point_mesh_edge_distance, point_mesh_face_distance
    from pytorch3d.structures import Pointclouds, Meshes
    from pytorch3d.ops import iterative_closest_point
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pytorch3d.renderer import (
        PerspectiveCameras,
        MeshRasterizer,
        HardPhongShader,
        MeshRenderer,
        RasterizationSettings,
        PointLights,
        look_at_view_transform, FoVPerspectiveCameras, look_at_rotation, TexturesVertex)
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, Transform3d, euler_angles_to_matrix
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from torch.func import functional_call
except ModuleNotFoundError as e:
    print("Required module not found. Ensure PyTorch and PyTorch3D are installed.")
    raise e


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

def plot_pointclouds(real_pc, synthetic_pc, model, std_devs=None, unexplained_points=None):
    """
    Plot the real and synthetic point clouds in the same graph.
    """

    ax.clear()
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real vs Synthetic Point Clouds")

    # Set fixed scales for the axes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    # Plot real point cloud
    ax.scatter(real_pc[:, 0].cpu().detach().numpy(),
               real_pc[:, 1].cpu().detach().numpy(),
               real_pc[:, 2].cpu().detach().numpy(),
               c='b', label='Real Point Cloud', s=1)

    # Plot synthetic point cloud
    if synthetic_pc is not None:
        ax.scatter(synthetic_pc[:, 0].cpu().detach().numpy(),
                   synthetic_pc[:, 1].cpu().detach().numpy(),
                   synthetic_pc[:, 2].cpu().detach().numpy(),
                   c='r', label='Synthetic Point Cloud', s=1)

    # Plot the model as a mesh
    mesh = model()
    mesh_faces = []
    verts_list = mesh.verts_list()[0].cpu().detach().numpy()  # Move to CPU and convert to NumPy
    for face in mesh.faces_list()[0]:
        mesh_faces.append([verts_list[face[0].item()], verts_list[face[1].item()], verts_list[face[2].item()]])
    mesh_collection = Poly3DCollection(mesh_faces, alpha=0.3, linewidths=0.5, edgecolors='black')
    mesh_collection.set_facecolor('cyan')
    ax.add_collection3d(mesh_collection)

    # Plot standard deviations as arrows
    if std_devs is not None:
        center = verts_list.mean(axis=0)
        bottom_center = center.copy()
        bottom_center[2] = 0
        middle = center.copy()
        middle[2] = model.h/3
        scale = 0.5
        colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']
        for i, (attr, std_dev) in enumerate(zip(model.attributes, std_devs)):
            direction = np.zeros(3)
            if attr in ['x', 'y', 'z']:
                if attr == 'x':
                    direction[0] = std_dev * scale  # Scale the arrow for visibility
                elif attr == 'y':
                    direction[1] = std_dev * scale
                elif attr == 'z':
                    direction[2] = std_dev * scale
                ax.quiver(bottom_center[0], bottom_center[1], bottom_center[2], direction[0], direction[1], direction[2], color=colors[i % len(colors)], length=scale, normalize=True)
            elif attr in ['a', 'b', 'c']:
                if attr == 'a':
                    direction[0] = std_dev * scale
                elif attr == 'b':
                    direction[1] = std_dev * scale
                elif attr == 'c':
                    direction[2] = std_dev * scale
                ax.quiver(middle[0], middle[1], middle[2], direction[0], direction[1], direction[2], color=colors[i % len(colors)], length=0.1, normalize=True)
            elif attr in ['w', 'd', 'h']:
                if attr == 'w':
                    direction[0] = std_dev * scale
                elif attr == 'd':
                    direction[1] = std_dev * scale
                elif attr == 'h':
                    direction[2] = std_dev * scale
                ax.quiver(center[0], center[1], center[2], direction[0], direction[1], direction[2], color=colors[i % len(colors)], length=0.1, normalize=True)

    # Plot unexplained points
    if unexplained_points is not None:
        ax.scatter(unexplained_points[:, 0].cpu().detach().numpy(),
                   unexplained_points[:, 1].cpu().detach().numpy(),
                   unexplained_points[:, 2].cpu().detach().numpy(),
                   c='r', label='Unexplained Points', s=5)

    ax.legend()
    plt.pause(0.001)
    #plt.show()

# ===========================
# Load the Real Point Cloud
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    real_pc = torch.load("real_fridge_pointcloud.pt").to(device)
except FileNotFoundError:
    print("Error: real_fridge_pointcloud.pt not found. Ensure the file exists.")
    raise

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

    def __init__(self, init_params, cam: Cameras, rasterizer: Rasterizer, device: str):
        super().__init__()
        self.cam = cam
        self.rasterizer = rasterizer
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
            [2, 3, 7], [2, 7, 6], [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]], dtype=torch.int32, device=device)

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
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
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
        mesh = fridge_model.forward()
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
            grad_i = torch.autograd.grad(loss_function(self, points), param_i, create_graph=True)[0]
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
        for name, param in fridge_model.named_parameters():
            print(f"    {name} value: {param.item()}")
    def print_grads(self):
        print("Gradients:")
        for name, param in fridge_model.named_parameters():
            print(f"    {name} grad: {param.grad}")

##########################################################
### losses
##########################################################
def loss_function(model, real_points):
    """ Compute the loss for optimization """

    # Compute mesh from updated model
    my_mesh = model.forward_visible_faces()

    # Compute loss terms
    loss_1 = point_mesh_edge_distance(my_mesh, Pointclouds([real_points]))  # Edge distance
    loss_2 = point_mesh_face_distance(my_mesh, Pointclouds([real_points]))  # Face distance
    loss_3 = prior_size(model, 0.6, 0.6, 1.8)
    loss_4 = prior_on_floor(model)
    loss_5 = prior_aligned(model)
    loss_6 = prior_position_by_mass_center(model, real_points)

    total_loss = loss_1 + loss_2 + loss_3 + 2 * loss_4 + 2*loss_5 + 0.5*loss_6
    return total_loss

def prior_size(fridge: FridgeModel, w=0.6, d=0.6, h=1.8):
    # Compute the difference between w,d,h and the prior values
    return torch.sum((fridge_model.w-w)**2 + (fridge_model.d-d)**2 + (fridge_model.h-h)**2)

def prior_on_floor(fridge: FridgeModel):
    # The height of the fridge (at center) has to be the semi-height of the fridge
    return torch.sum((fridge.z - fridge.h/2)**2)

def prior_aligned(fridge: FridgeModel):
    # The fridge is aligned with the room axis: a,b,c = 0
    return torch.sum(fridge.a**2 + fridge.b**2 + fridge.c**2)

def prior_position_by_mass_center(model: FridgeModel, points: torch.Tensor):
    # compute the mass center of the point cloud
    mass_center = torch.mean(points, dim=0)
    # displace the mass_center backwards by the prior width and depth of the fridge
    x0 = mass_center[0]
    y0 = mass_center[1]
    return (model.x - x0)**2 + (model.y - y0)**2

###########################################################
if __name__ == "__main__":

    # ===========================
    # Optimization Setup
    # ===========================
    init_params = [-0.3, 0.4, 0.9, 0.0, 0.0, 0.1, 0.5, 0.2, 1.8]  # Initial guess
    cam = Cameras(img_size=128, device="cuda")
    rasterizer = Rasterizer(cam, device="cuda")
    fridge_model = FridgeModel(init_params, cam, rasterizer, "cuda").to(device)
 
    optimizer = optim.Adam([
              {'params': [fridge_model.x, fridge_model.y, fridge_model.z], 'lr': 0.1, 'momentum': 0.6},  # Position
              {'params': [fridge_model.a, fridge_model.b, fridge_model.c], 'lr': 0.01, 'momentum': 0.6},  # Rotation (higher LR)
              {'params': [fridge_model.w, fridge_model.d, fridge_model.h], 'lr': 0.1, 'momentum': 0.6}  # Size
          ])

    ICP = False

    # ===========================
    # Initialize the model using ICP
    # ===========================
    if ICP:
        synthetic_pc = fridge_model.rasterize_visible_faces()
        plot_pointclouds(real_pc, synthetic_pc, fridge_model)
        plt.waitforbuttonpress()
        sol = iterative_closest_point(Pointclouds([synthetic_pc]), Pointclouds([real_pc]), max_iterations=1000)
        print("ICP:", sol.converged, sol.rmse.item())
        if sol.converged:
            fridge_model.x.data += sol.RTs.T[0, 0]
            fridge_model.y.data += sol.RTs.T[0, 1]
            fridge_model.z.data += sol.RTs.T[0, 2]
            angles = matrix_to_euler_angles(sol.RTs.R, "XYZ")
            fridge_model.a.data = torch.tensor(0.0, device="cuda") #angles[0, 0]
            fridge_model.b.data = torch.tensor(0.0, device="cuda") #angles[0, 1]
            fridge_model.c.data = torch.tensor(0.0, device="cuda") #angles[0, 2]
            scale = sol.RTs.s
            fridge_model.w.data *= scale[0]
            fridge_model.d.data *= scale[0]
            fridge_model.h.data *= scale[0]
            plot_pointclouds(real_pc, synthetic_pc, fridge_model)
            fridge_model.print_params()
            plt.waitforbuttonpress()

    plot_pointclouds(real_pc, None, fridge_model)
    fridge_model.print_params()
    plt.waitforbuttonpress()

    # ===========================
    # Optimize the Model
    # ===========================
    num_iterations = 2000
    loss = mesh = None
    loss_ant = float('inf')
    now = time.time()
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function(fridge_model, real_pc)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item():6f}")  # Check gradients
            if abs(loss.item() - loss_ant) < 0.00001:    # Convergence criterion
                break
            loss_ant = loss.item()


    print("Optimization complete.")
    fridge_model.print_params()
    fridge_model.print_grads()
    print(f"Elapsed time: {time.time() - now:.4f} seconds")
    hess, cov = fridge_model.hessian_and_covariance(real_pc)
    np.set_printoptions(precision=2, suppress=True)
    print("Covariance Matrix:\n", cov.cpu().numpy())
    std_devs = torch.sqrt(torch.diag(cov))
    print("Standard Deviations (Uncertainty) per Parameter:", std_devs.cpu().numpy())
    eigvals, eigvecs = torch.linalg.eigh(hess)
    print("Eigenvalues of Hessian:", eigvals.cpu().numpy())
    unexplained = fridge_model.remove_explained_points(real_pc, 0.05)
    print("Unexplained points:", unexplained.shape[0], "out of", real_pc.shape[0], "percentage:", unexplained.shape[0] / real_pc.shape[0] * 100 , "%")
    plot_pointclouds(real_pc, None, fridge_model, std_devs.cpu().numpy(), unexplained)
    plt.show()


