import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    HardPhongShader,
    MeshRenderer,
    RasterizationSettings,
    PointLights,
    look_at_view_transform, FoVPerspectiveCameras, look_at_rotation, TexturesVertex
)
from pytorch3d.transforms import axis_angle_to_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs

def plot_mesh_and_points(verts, faces, points, camera_position):
     """
     Plot the mesh, reconstructed points, and camera in the same 3D plot.

     Args:
         verts: Mesh vertices (N, 3)
         faces: Mesh faces (F, 3)
         points: Reconstructed 3D points (M, 3)
         camera_position: Camera position (3,)
         camera_direction: Dictionary with camera direction vectors
         output_file: File to save the plot
     """
     # Convert to numpy for matplotlib
     if torch.is_tensor(verts):
         verts = verts.detach().cpu().numpy()
     if torch.is_tensor(faces):
         faces = faces.detach().cpu().numpy()
     if torch.is_tensor(points):
         points = points.detach().cpu().numpy()

    # Create figure and 3D axis
     fig = plt.figure(figsize=(12, 10))
     ax = fig.add_subplot(111, projection='3d')

     # Plot the mesh
     mesh_faces = []
     for face in faces:
         mesh_faces.append([verts[face[0]], verts[face[1]], verts[face[2]]])

     mesh_collection = Poly3DCollection(mesh_faces, alpha=0.3, linewidths=0.5, edgecolors='black')
     mesh_collection.set_facecolor('cyan')
     ax.add_collection3d(mesh_collection)
#
#     # Plot the reconstructed points
     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=1, alpha=0.8, label='Reconstructed Points')

     # Plot the camera position
     ax.scatter(camera_position[0], camera_position[1], camera_position[2],
                c='green', s=100, marker='^', label='Camera')

     #Set labels and title
     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_zlabel('Z')
     ax.set_title('Mesh, Reconstructed 3D Points, and Camera')


def rasterize_and_recover_points(mesh_verts, mesh_faces, image_size=256, device="cuda"):
    """
    Rasterize a mesh with a FOV camera and recover 3D world coordinates.

    Args:
        mesh_verts: Vertices of the mesh (N, 3)
        mesh_faces: Faces of the mesh (F, 3)
        image_size: Size of the rendered image
        device: Computation device ("cuda" or "cpu")

    Returns:
        rendered_depth: Depth map from rasterization
        world_coords: Recovered 3D world coordinates
        mesh_verts: Mesh vertices
        mesh_faces: Mesh faces
        camera_position: Camera position
        camera_direction: Camera direction vectors
    """
    # Move inputs to specified device
    if isinstance(mesh_verts, np.ndarray):
        mesh_verts = torch.FloatTensor(mesh_verts).unsqueeze(0).to(device)
    if isinstance(mesh_faces, np.ndarray):
        mesh_faces = torch.LongTensor(mesh_faces).unsqueeze(0).to(device)

    texture_rgb = torch.ones_like(mesh_verts)  # N X 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1], dtype=torch.float32).to(device)
    textures = TexturesVertex(texture_rgb.unsqueeze(0))  # important

    # Create a Meshes object
    mesh = Meshes(
        verts=[mesh_verts],
        faces=[mesh_faces],
        textures=textures
    )

    # Define camera position and direction
    #camera_position = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32).to(device)
    camera_R = torch.eye(3, dtype=torch.float32).unsqueeze(0).to(device)
    #R, T = look_at_view_transform(dist=3, elev=-45, azim=0, device=device)
    #T[:, 1] += 0.5
    R = axis_angle_to_matrix(torch.tensor([0, 0, 0], dtype=torch.float32, device=device)* torch.pi).unsqueeze(0).to(device)
    T = torch.tensor([-1.5, -1.5, 3], dtype=torch.float32).unsqueeze(0).to(device)
    #R = look_at_rotation(T, at=((0, 0, 0),), up=((0, 1, 0),), device = 'cuda')

    print("T", T)
    # Initialize a camera with perspective projection
    cameras = FoVPerspectiveCameras(
        fov=60.0,
        R=R,
        T=T,
        device=device
    )

    # Create a rasterizer
    raster_settings = RasterizationSettings(
        image_size=128,
        faces_per_pixel=1,
        perspective_correct=True,
        clip_barycentric_coords=True,
        cull_backfaces=False
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    shader = HardPhongShader(device=device)
    renderer = MeshRenderer( rasterizer=rasterizer, shader=shader,
    )
    lights = PointLights(location=[[0, 0, -3]], device=device)
    image = renderer(mesh, cameras=cameras, lights=lights)
    #plt.imshow(image[0].cpu().numpy())
    #plt.show()

    # Rasterize the mesh
    fragments = rasterizer(mesh)

    # Extract the depth map
    zbuf = fragments.zbuf
    rendered_depth = zbuf[..., 0]  # (1, H, W)

    # Create pixel coordinates
    height, width = rendered_depth.shape[1:]
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="xy"
    )
    x = -x  # FUCKKKK
    y = -y

    # For each pixel with valid depth, recover 3D world coordinates
    valid_mask = rendered_depth[0] > 0  # Remove the batch dimension

    # Get the pixel coordinates for valid depth values
    pixel_coords = torch.stack([x, y], dim=-1)[valid_mask]  # (N, 2)

    # Get the corresponding depth values
    depth_values = rendered_depth[0][valid_mask]  # (N,)

    # Convert to NDC space points with depth
    points_ndc = torch.cat([
        pixel_coords,  # (N, 2)
        depth_values.unsqueeze(-1),  # (N, 1)
    ], dim=-1)  # (N, 3)

    points_ndc_batched = points_ndc.unsqueeze(0)  # (1, N, 3)

    # Unproject to world coordinates
    w_points = cameras.unproject_points(points_ndc_batched, world_coordinates=True)

    # Step 2: Transform from camera space to world space
    #world_to_view_transform = cameras.get_world_to_view_transform()
    #view_to_world_transform = world_to_view_transform.inverse()
    #w_points = view_to_world_transform.transform_points(points_ndc_batched)

    # # Remove the batch dimension
    w_points = w_points.squeeze(0)  # (N, 3)
    pts = Pointclouds(points=[w_points])

    return rendered_depth[0], pts, mesh, cameras

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a simple mesh (a cube)
    verts = torch.tensor([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=torch.float32, device=device)

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
    ], dtype=torch.int64, device=device)

    # Rasterize and recover 3D points
    depth, world_points, mesh, cameras = rasterize_and_recover_points(verts, faces, image_size=256, device=device )

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": mesh,
    #         "Camera": cameras,
    #         "Pointcloud": world_points
    #     }
    # })
    #
    # fig.show()

    camera_center = cameras.get_camera_center().squeeze(0).squeeze(0).squeeze(0).cpu().numpy()
    print(camera_center)
    plot_mesh_and_points(verts, faces, world_points.points_list()[0], camera_center)
    torch.save(world_points.points_list()[0], "real_fridge_pointcloud.pt")
    print(f"Rendered depth shape: {depth.shape}")
    plt.show()

