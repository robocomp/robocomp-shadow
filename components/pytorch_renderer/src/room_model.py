import torch
import torch.nn as nn


class ParametricRoom(nn.Module):
    """Differentiable parametric room model matching your C++ version"""

    def __init__(self, initial_corners, height, device='cuda'):
        """
        Args:
            initial_corners: [4, 2] torch.Tensor - room corners in XY plane
            height: float - room height
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device

        # Geometry parameters (can be optimized)
        self.corners = nn.Parameter(
            torch.tensor(initial_corners, dtype=torch.float32, device=device)
        )
        self.height = nn.Parameter(
            torch.tensor(height, dtype=torch.float32, device=device)
        )

        # Appearance parameters
        self.wall_colors = nn.Parameter(
            torch.ones(4, 3, device=device) * 0.7  # Gray walls
        )
        self.floor_color = nn.Parameter(
            torch.tensor([0.6, 0.5, 0.4], device=device)  # Brown floor
        )
        self.ceiling_color = nn.Parameter(
            torch.ones(3, device=device) * 0.9  # White ceiling
        )

        # Lighting parameters - MUCH BRIGHTER
        self.ambient_light = nn.Parameter(
            torch.ones(3, device=device) * 0.6  # Increased from 0.3
        )
        self.light_location = nn.Parameter(
            torch.tensor([0.0, 0.0, 5.0], device=device)  # Higher up
        )
        self.light_intensity = nn.Parameter(
            torch.ones(3, device=device) * 1.5  # Increased from 0.7
        )

    def to_mesh(self):
        """
        Convert parametric room to mesh (vertices, faces, colors)
        Better vertex color assignment
        """
        # Floor corners (z=0)
        floor_corners = torch.cat([
            self.corners,
            torch.zeros(4, 1, device=self.device)
        ], dim=1)

        # Ceiling corners (z=height)
        ceiling_corners = torch.cat([
            self.corners,
            self.height.expand(4, 1)
        ], dim=1)

        # All 8 vertices
        vertices = torch.cat([floor_corners, ceiling_corners], dim=0)  # [8, 3]

        # Define faces
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # Floor
            [4, 6, 5], [4, 7, 6],  # Ceiling
            [0, 1, 5], [0, 5, 4],  # Wall 0
            [1, 2, 6], [1, 6, 5],  # Wall 1
            [2, 3, 7], [2, 7, 6],  # Wall 2
            [3, 0, 4], [3, 4, 7],  # Wall 3
        ], dtype=torch.int64, device=self.device)

        # BETTER vertex colors - each vertex gets average of surfaces it touches
        vertex_colors = torch.zeros(8, 3, device=self.device)

        # Floor vertices: blend floor + two adjacent walls
        for i in range(4):
            wall_idx = i
            prev_wall_idx = (i - 1) % 4
            vertex_colors[i] = (
                    self.floor_color * 0.5 +
                    self.wall_colors[wall_idx] * 0.25 +
                    self.wall_colors[prev_wall_idx] * 0.25
            )

        # Ceiling vertices: blend ceiling + two adjacent walls
        for i in range(4):
            wall_idx = i
            prev_wall_idx = (i - 1) % 4
            vertex_colors[4 + i] = (
                    self.ceiling_color * 0.5 +
                    self.wall_colors[wall_idx] * 0.25 +
                    self.wall_colors[prev_wall_idx] * 0.25
            )

        return {
            'vertices': vertices,
            'faces': faces,
            'vertex_colors': vertex_colors
        }

    def clamp_parameters(self):
        """Clamp parameters to valid ranges"""
        with torch.no_grad():
            self.wall_colors.clamp_(0.0, 1.0)
            self.floor_color.clamp_(0.0, 1.0)
            self.ceiling_color.clamp_(0.0, 1.0)
            self.ambient_light.clamp_(0.0, 1.0)
            self.light_intensity.clamp_(0.0, 2.0)
            self.height.clamp_(1.5, 5.0)  # Reasonable room heights

    def get_state_dict(self):
        """Export current parameters"""
        return {
            'corners': self.corners.detach().cpu().numpy(),
            'height': self.height.item(),
            'wall_colors': self.wall_colors.detach().cpu().numpy(),
            'floor_color': self.floor_color.detach().cpu().numpy(),
            'ceiling_color': self.ceiling_color.detach().cpu().numpy(),
            'ambient_light': self.ambient_light.detach().cpu().numpy(),
            'light_location': self.light_location.detach().cpu().numpy(),
            'light_intensity': self.light_intensity.detach().cpu().numpy(),
        }