from torch._lazy import closure

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
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
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, Transform3d
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ModuleNotFoundError as e:
    print("Required module not found. Ensure PyTorch and PyTorch3D are installed.")
    raise e


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

def plot_pointclouds(real_pc, synthetic_pc, model):
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
    mesh = model.compute_orthohedron()
    mesh_faces = []
    verts_list = mesh.verts_list()[0].cpu().detach().numpy()  # Move to CPU and convert to NumPy
    for face in mesh.faces_list()[0]:
        mesh_faces.append([verts_list[face[0].item()], verts_list[face[1].item()], verts_list[face[2].item()]])
    mesh_collection = Poly3DCollection(mesh_faces, alpha=0.3, linewidths=0.5, edgecolors='black')
    mesh_collection.set_facecolor('cyan')
    ax.add_collection3d(mesh_collection)
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
        T = torch.tensor([-2.5, 1.5, 3], dtype=torch.float32).unsqueeze(0).to(device)
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

# ===========================
# Fridge Model
# ===========================
class FridgeModel(nn.Module):
    def __init__(self, init_params, cam: Cameras, rasterizer: Rasterizer, device: str):
        super().__init__()
        self.cam = cam
        self.rasterizer = rasterizer
        self.device = device

        # Define each parameter separately as a trainable nn.Parameter
        self.x = nn.Parameter(torch.tensor(init_params[0], dtype=torch.float32, requires_grad=True, device=self.device))
        self.y = nn.Parameter(torch.tensor(init_params[1], dtype=torch.float32, requires_grad=True, device=self.device))
        self.z = nn.Parameter(torch.tensor(init_params[2], dtype=torch.float32, requires_grad=True, device=self.device))

        self.a = nn.Parameter(torch.tensor(init_params[3], dtype=torch.float32, requires_grad=True, device=self.device))
        self.b = nn.Parameter(torch.tensor(init_params[4], dtype=torch.float32, requires_grad=True, device=self.device))
        self.c = nn.Parameter(torch.tensor(init_params[5], dtype=torch.float32, requires_grad=True, device=self.device))

        self.w = nn.Parameter(torch.tensor(init_params[6], dtype=torch.float32, requires_grad=True, device=self.device))
        self.d = nn.Parameter(torch.tensor(init_params[7], dtype=torch.float32, requires_grad=True, device=self.device))
        self.h = nn.Parameter(torch.tensor(init_params[8], dtype=torch.float32, requires_grad=True, device=self.device))

    def forward(self):
        mesh = self.compute_orthohedron()
        #synthetic_pc = self.sample_surface_points(corners)
        #synthetic_pc = self.rasterize_visible_faces(mesh)
        return mesh

    def compute_orthohedron(self) -> Meshes:
        """Compute 3D corners of the oriented bounding box with proper gradient tracking."""

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

        R = self.euler_to_rotation_matrix(self.a, self.b, self.c)
        #verts = (R @ base_corners.T).T + torch.stack([self.x, self.y, self.z])
        t = t2 = Transform3d(device="cuda").scale(self.w, self.d, self.h).rotate(R).translate(self.x, self.y, self.z)
        verts = t.transform_points(base_corners)

        faces = torch.tensor([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ], dtype=torch.int32, device=device)

        # Create a Meshes object
        mesh = Meshes(
            verts=[verts],
            faces=[faces]
            #textures=textures
        )
        return mesh

    def euler_to_rotation_matrix(self, a, b, c):
        """Convert Euler angles to a rotation matrix while maintaining gradient flow."""
        cos_a, sin_a = torch.cos(a), torch.sin(a)
        cos_b, sin_b = torch.cos(b), torch.sin(b)
        cos_c, sin_c = torch.cos(c), torch.sin(c)

        Rx = torch.stack([
            torch.stack([torch.tensor(1.0, device=a.device), torch.tensor(0.0, device=a.device), torch.tensor(0.0, device=a.device)]),
            torch.stack([torch.tensor(0.0, device=a.device), cos_a, -sin_a]),
            torch.stack([torch.tensor(0.0, device=a.device), sin_a, cos_a])
        ])

        Ry = torch.stack([
            torch.stack([cos_b, torch.tensor(0.0, device=b.device), sin_b]),
            torch.stack([torch.tensor(0.0, device=b.device), torch.tensor(1.0, device=b.device), torch.tensor(0.0, device=b.device)]),
            torch.stack([-sin_b, torch.tensor(0.0, device=b.device), cos_b])
        ])

        Rz = torch.stack([
            torch.stack([cos_c, -sin_c, torch.tensor(0.0, device=c.device)]),
            torch.stack([sin_c, cos_c, torch.tensor(0.0, device=c.device)]),
            torch.stack([torch.tensor(0.0, device=c.device), torch.tensor(0.0, device=c.device), torch.tensor(1.0, device=c.device)])
        ])

        return Rz @ Ry @ Rx  # Correct rotation order (Z-Y-X intrinsic)

    def sample_surface_points(self, corners, num_samples=1000):
        faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (0, 1, 2, 3), (4, 5, 6, 7)]
        sampled_points = []
        num_samples_per_face = num_samples // len(faces)
        for face in faces:
            p1, p2, p3, p4 = corners[face[0]], corners[face[1]], corners[face[2]], corners[face[3]]
            u = torch.linspace(0, 1, int(num_samples_per_face**0.5), device=self.device)
            v = torch.linspace(0, 1, int(num_samples_per_face**0.5), device=self.device)
            grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
            points = (1 - grid_u[:, :, None]) * (1 - grid_v[:, :, None]) * p1 + \
                     grid_u[:, :, None] * (1 - grid_v[:, :, None]) * p2 + \
                     grid_u[:, :, None] * grid_v[:, :, None] * p3 + \
                     (1 - grid_u[:, :, None]) * grid_v[:, :, None] * p4
            sampled_points.append(points.reshape(-1, 3))
        return torch.cat(sampled_points, dim=0)

    def rasterize_visible_faces(self, mesh: Meshes):
        """
        Rasterize the visible faces of the mesh to a binary image.
        """

        # Rasterize the mesh
        fragments = self.rasterizer.rasterizer(mesh)

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

    def print_params(self):
        print("Model Parameters:")
        print(" x: {:.4f}".format(fridge_model.x.item()))
        print(" y: {:.4f}".format(fridge_model.y.item()))
        print(" z: {:.4f}".format(fridge_model.z.item()))
        print(" a: {:.4f}".format(fridge_model.a.item()))
        print(" b: {:.4f}".format(fridge_model.b.item()))
        print(" c: {:.4f}".format(fridge_model.c.item()))
        print(" w: {:.4f}".format(fridge_model.w.item()))
        print(" d: {:.4f}".format(fridge_model.d.item()))
        print(" h: {:.4f}".format(fridge_model.h.item()))

    def print_grads(self):
        print("Gradients:")
        print(" x: {:.4f}".format(fridge_model.x.grad.item()))
        print(" y: {:.4f}".format(fridge_model.y.grad.item()))
        print(" z: {:.4f}".format(fridge_model.z.grad.item()))
        print(" a: {:.4f}".format(fridge_model.a.grad.item()))
        print(" b: {:.4f}".format(fridge_model.b.grad.item()))
        print(" c: {:.4f}".format(fridge_model.c.grad.item()))
        print(" w: {:.4f}".format(fridge_model.w.grad.item()))
        print(" d: {:.4f}".format(fridge_model.d.grad.item()))
        print(" h: {:.4f}".format(fridge_model.h.grad.item()))

##########################################################
def prior_size(w=0.6, d=0.6, h=1.8):
    # Compute the difference between w,d,h and the prior values
    return torch.sum((fridge_model.w-w)**2 + (fridge_model.d-d)**2 + (fridge_model.h-h)**2)

def prior_on_floor():
    # The height of the fridge (at center) has to be the semi-height of the fridge
    return torch.sum((fridge_model.z - fridge_model.h/2)**2)
###########################################################
if __name__ == "__main__":

    # ===========================
    # Optimization Setup
    # ===========================
    init_params = [0.5, 0, 0.9, 0.0, 0.0, 0.0, 0.6, 0.6, 1.8]  # Initial guess
    cam = Cameras(img_size=128, device="cuda")
    rasterizer = Rasterizer(cam, device="cuda")
    fridge_model = FridgeModel(init_params, cam, rasterizer, "cuda").to(device)
    #fridge_model.params.register_hook(print_grad)
    optimizer = optim.SGD([
         {'params': [fridge_model.x, fridge_model.y, fridge_model.z], 'lr': 0.1, 'momentum': 0.6},  # Position
         {'params': [fridge_model.a, fridge_model.b, fridge_model.c], 'lr': 0.00, 'momentum': 0.6},  # Rotation (higher LR)
         {'params': [fridge_model.w, fridge_model.d, fridge_model.h], 'lr': 0.1, 'momentum': 0.6}  # Size
     ])


    #plot_pointclouds(real_pc, fridge_model(), fridge_model)
    # ===========================
    # Initialize the model using ICP
    # ===========================
    synthetic_pc = fridge_model.rasterize_visible_faces(fridge_model())
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
    plot_pointclouds(real_pc, sol.Xt.points_list()[0], fridge_model)
    fridge_model.print_params()
    mesh = fridge_model.compute_orthohedron()
    print("Dist:", 0.5*point_mesh_face_distance(mesh, Pointclouds([real_pc]))+point_mesh_edge_distance(mesh, Pointclouds([real_pc])))
    plt.waitforbuttonpress()

    # ===========================
    # Optimize the Model
    # ===========================
    num_iterations = 2000
    for i in range(num_iterations):
        optimizer.zero_grad()
        mesh = fridge_model()
        #synthetic_pc = fridge_model.rasterize_visible_faces(mesh)

        # Compute Chamfer Distance
        #loss0, _ = chamfer_distance(Pointclouds([synthetic_pc]), Pointclouds([real_pc]))
        loss_1 = point_mesh_edge_distance(mesh, Pointclouds([real_pc]))  # Edge distance
        loss_2 = point_mesh_face_distance(mesh, Pointclouds([real_pc]))  # Face distance
        # compute distance between w,d,h and 0.6, 0.6, 1.8
        loss_3 = prior_size(0.6, 0.6, 1.8)
        loss_4 = prior_on_floor()

        loss = loss_2 + loss_3 + 2*loss_4
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")  # Check gradients
            plot_pointclouds(real_pc, None, fridge_model)
            fridge_model.print_params()

            #print("Gradients:", synthetic_pc.grad)
            #print(fridge_model.params.detach().cpu().numpy())

    print("Optimization complete.")
    plot_pointclouds(real_pc, None, fridge_model)
    fridge_model.print_params()

    plt.show()



