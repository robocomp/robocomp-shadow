import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.renderer import (MeshRenderer, MeshRasterizer, RasterizationSettings, PerspectiveCameras)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def plot_pointclouds(real_pc, camera_pose, mesh):
#     """
#     Plot the real and synthetic point clouds in the same graph with the camera pose.
#     """
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     #print("plot ", real_pc.shape)
#
#     #plot Pointclouds
#     real = real_pc.points_list()
#     print (len (real))
#
#     # Plot real in ax.scatter
#     ax.scatter(real[0][:, 0].cpu().detach().numpy(),
#                real[0][:, 1].cpu().detach().numpy(),
#                real[0][:, 2].cpu().detach().numpy(),
#                c='b', label='Real Point Cloud', s=1)
#
#     # ax.scatter(real[:, 0].cpu().detach().numpy(),
#     #            real[:, 1].cpu().detach().numpy(),
#     #            real[:, 2].cpu().detach().numpy(),
#     #            c='b', label='Real Point Cloud', s=1)
#
#     # Plot camera position
#     cam_pos = camera_pose[:3, 3].cpu().detach().numpy()
#     ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', label='Camera Position', s=100, marker='o')
#
#     # Plot camera orientation
#     cam_axes = camera_pose[:3, :3].cpu().detach().numpy()
#     axis_length = 0.2  # Adjust axis length for better visualization
#     origin = cam_pos
#
#     ax.quiver(origin[0], origin[1], origin[2], cam_axes[0, 0], cam_axes[1, 0], cam_axes[2, 0], color='r', length=axis_length)
#     ax.quiver(origin[0], origin[1], origin[2], cam_axes[0, 1], cam_axes[1, 1], cam_axes[2, 1], color='g', length=axis_length)
#     ax.quiver(origin[0], origin[1], origin[2], cam_axes[0, 2], cam_axes[1, 2], cam_axes[2, 2], color='b', length=axis_length)
#
#     # plot mesh
#     verts = mesh.verts_list()[0].cpu().detach().numpy()
#     faces = mesh.faces_list()[0].cpu().detach().numpy()
#
#     for face in faces:
#         x = [verts[face[i], 0] for i in range(3)] + [verts[face[0], 0]]
#         y = [verts[face[i], 1] for i in range(3)] + [verts[face[0], 1]]
#         z = [verts[face[i], 2] for i in range(3)] + [verts[face[0], 2]]
#         ax.plot(x, y, z, color='k')
#
#     # Set labels and title
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Real Point Clouds with Camera Orientation")
#     ax.legend()
#
#     plt.show()


def plot_my_scene(mesh, pointcloud, camera_pose):
     fig = plot_scene({"Scene": {"Mesh": mesh,
                                 "Pointcloud": pointcloud,
                                 "Camera": camera_pose}},
                      axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)",  showgrid=True, zeroline=True, showline=True))
     fig.show()

class Camera:
    def __init__(self, device):
        self.device = device
        # Camera positioned at (0, 0, 1.2) in world frame
        # Rotate -Z to align with Y+
        axis_angle = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device) * (-torch.pi / 2)
        R = axis_angle_to_matrix(axis_angle)
        T = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)  # After rotation on X, elevate by 1 on Y
        self.camera = PerspectiveCameras(device=device, R=R.unsqueeze(0), T=T.unsqueeze(0))

    def get_extrinsics(self):
        """
        Return the camera extrinsic transformation matrix (4x4).
        """
        extrinsics = torch.eye(4, dtype=torch.float32, device=self.device)
        extrinsics[:3, :3] = self.camera.R
        extrinsics[:3, 3] = self.camera.T
        return extrinsics

def euler_to_rotation_matrix(a, b, c):
    """Convert Euler angles to a rotation matrix."""
    a, b, c = torch.tensor([a, b, c], dtype=torch.float32, device=device)
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(a), -torch.sin(a)], [0, torch.sin(a), torch.cos(a)]], dtype=torch.float32, device=device)
    Ry = torch.tensor([[torch.cos(b), 0, torch.sin(b)], [0, 1, 0], [-torch.sin(b), 0, torch.cos(b)]], dtype=torch.float32, device=device)
    Rz = torch.tensor([[torch.cos(c), -torch.sin(c), 0], [torch.sin(c), torch.cos(c), 0], [0, 0, 1]], dtype=torch.float32, device=device)
    return Rz @ Ry @ Rx

def compute_orthohedron(x, y, z, a, b, c, w, d, h):
    """
    Compute 3D corners of the oriented bounding box given parameters.
    """
    base_corners = torch.tensor([
        [-w / 2, -d / 2, -h / 2],
        [w / 2, -d / 2, -h / 2],
        [w / 2, d / 2, -h / 2],
        [-w / 2, d / 2, -h / 2],
        [-w / 2, -d / 2, h / 2],
        [w / 2, -d / 2, h / 2],
        [w / 2, d / 2, h / 2],
        [-w / 2, d / 2, h / 2]
    ], dtype=torch.float32, device=device)
    R = euler_to_rotation_matrix(a, b, c)
    rotated_corners = (R @ base_corners.T).T
    translation = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    return rotated_corners + translation

def create_mesh_from_corners(corners):
    faces = torch.tensor([
        [0, 1, 2], [2, 3, 0],  # Bottom face
        [4, 5, 6], [6, 7, 4],  # Top face
        [0, 1, 5], [5, 4, 0],  # Side face
        [1, 2, 6], [6, 5, 1],  # Side face
        [2, 3, 7], [7, 6, 2],  # Side face
        [3, 0, 4], [4, 7, 3]   # Side face
    ], dtype=torch.int64, device=device)
    return Meshes(verts=[corners], faces=[faces])

def render_synthetic_pc(mesh):
    raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=Camera(device).camera, raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    depth_map = fragments.zbuf[..., 0]  # Extract depth values

    # Get valid depth points (non-zero depth)
    valid_mask = depth_map > 0
    coords = torch.nonzero(valid_mask, as_tuple=False).float()  # Pixel indices (H, W)
    depths = depth_map[valid_mask].view(-1, 1)  # Ensure correct shape [P, 1]

    if depths.numel() == 0 or coords.numel() == 0:
        print("Warning: No valid depth points detected.")
        return Pointclouds(points=[torch.empty((0, 3), device=depth_map.device)])  # Return empty point cloud

    # Get camera intrinsics (assumed)
    D, H, W = depth_map.shape
    sensor_size = 1.0  # Assume a normalized camera sensor
    focal_length = 1.0  # Assume default focal length
    # Convert from pixel coordinates to camera space
    x = ((coords[:, 0:1] - W / 2) / (W / 2)) * sensor_size * depths / focal_length
    y = -((coords[:, 1:2] - H / 2) / (H / 2)) * sensor_size * depths / focal_length
    z = depths.view(-1, 1)  # Ensure correct shape


    points = torch.stack([x, y, z])
    points = points.permute(1, 0, 2).view(-1, 3) # Permute and reshape to get the desired shape [1776, 3]

    # Transform points from camera space to world space using extrinsics
    cam_extrinsics = Camera(device).get_extrinsics()  # Retrieve 4x4 transformation matrix
    R, T = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3].unsqueeze(0)  # Extract rotation and translation
    points_world = (R @ points.T).T + T  # Apply transformation

    return Pointclouds(points=[points_world])  # Return as a PyTorch3D Pointclouds object

# def render_synthetic_pc(mesh):
#     raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
#     rasterizer = MeshRasterizer(cameras=Camera(device).camera, raster_settings=raster_settings)
#     fragments = rasterizer(mesh)
#     depth_map = fragments.zbuf[..., 0]  # Extract depth values
#
#     # Get valid depth points (non-zero depth)
#     valid_mask = depth_map > 0
#     coords = torch.nonzero(valid_mask, as_tuple=False).float()
#     depths = depth_map[valid_mask].view(-1, 1)  # Ensure correct shape [P, 1]
#
#     if depths.numel() == 0 or coords.numel() == 0:
#         print("Warning: No valid depth points detected.")
#         return Pointclouds(points=[torch.empty((0, 3), device=depth_map.device)])  # Return empty point cloud
#
#     # Convert image plane coordinates to normalized camera coordinates
#     D, H, W = depth_map.shape
#     focal_length = 1.0  # Assume default focal length
#     sensor_size = 1.0  # Assume a normalized camera sensor
#     x = ((coords[:, 1:2] - W / 2) / (W / 2)) * sensor_size * depths / focal_length
#     y = -((coords[:, 0:1] - H / 2) / (H / 2)) * sensor_size * depths / focal_length
#     z = depths.view(-1, 1)  # Ensure correct shape
#     points = torch.stack([x, y, z])
#     # Permute and reshape to get the desired shape [1776, 3]
#     points = points.permute(1, 0, 2).view(-1, 3)
#     return Pointclouds(points=[points])  # Return as a PyTorch3D Pointclouds object

# Main script
if __name__ == "__main__":
    corners = compute_orthohedron(0.0, 2.0, 0.9, 0.0, 0.0, 0.0, 0.6, 0.6, 1.8)
    mesh = create_mesh_from_corners(corners)
    pointcloud = render_synthetic_pc(mesh)
    filename = "real_fridge_pointcloud.pt"
    torch.save(pointcloud, filename)
    print(f"Fake fridge point cloud saved to {filename}")
    #plot_pointclouds(pointcloud, Camera(device).pose, mesh)
    plot_my_scene(mesh, pointcloud, Camera(device).camera)
    #print(fake_fridge_points)
    print(len(pointcloud))
