import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex
)


class DifferentiableRoomRenderer:
    """PyTorch3D-based differentiable renderer"""

    def __init__(self, image_size=512, device='cuda'):
        self.device = device
        self.image_size = image_size

    def render(self, room_model, camera_position, camera_target,
               camera_up=None, fov=60):
        """
        Render the room from a given camera pose

        Args:
            room_model: ParametricRoom instance
            camera_position: [3] tensor - camera location
            camera_target: [3] tensor - look-at point
            camera_up: [3] tensor - up vector (default: [0, 0, 1])
            fov: float - field of view in degrees

        Returns:
            rendered_image: [H, W, 3] RGB image
        """
        if camera_up is None:
            camera_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Get mesh from parametric model
        mesh_data = room_model.to_mesh()

        # Create PyTorch3D mesh
        verts = mesh_data['vertices'].unsqueeze(0)  # [1, 8, 3]
        faces = mesh_data['faces'].unsqueeze(0)  # [1, 12, 3]
        vertex_colors = mesh_data['vertex_colors'].unsqueeze(0)  # [1, 8, 3]

        # Create texture from vertex colors
        textures = TexturesVertex(verts_features=vertex_colors)

        mesh = Meshes(
            verts=verts,
            faces=faces,
            textures=textures
        )

        # Setup camera
        # PyTorch3D uses a different coordinate system, need to transform
        R, T = self._look_at_view_transform(
            camera_position, camera_target, camera_up
        )

        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            fov=fov
        )

        # Setup lights
        lights = PointLights(
            device=self.device,
            location=room_model.light_location.unsqueeze(0),
            ambient_color=room_model.ambient_light.unsqueeze(0),
            diffuse_color=room_model.light_intensity.unsqueeze(0),
            specular_color=torch.zeros(1, 3, device=self.device)
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0  # Use naive rasterization
        )

        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )

        # Render
        images = renderer(mesh)

        # Extract RGB (discard alpha)
        return images[0, ..., :3]

    def _look_at_view_transform(self, eye, at, up):
        """
        Create rotation and translation for look-at camera
        Following your C++ convention: Y+ forward, X+ right, Z+ up
        """
        # Compute camera coordinate frame
        z_axis = at - eye
        z_axis = z_axis / torch.norm(z_axis)

        x_axis = torch.cross(z_axis, up)
        x_axis = x_axis / torch.norm(x_axis)

        y_axis = torch.cross(z_axis, x_axis)

        # Rotation matrix (world to camera)
        R = torch.stack([x_axis, y_axis, z_axis], dim=0)

        # Translation
        T = -torch.matmul(R, eye)

        return R, T