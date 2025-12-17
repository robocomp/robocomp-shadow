import torch
import torch.nn as nn
from tqdm import tqdm


class RoomOptimizer:
    """Optimize room parameters to match observed image"""

    def __init__(self, room_model, renderer, config=None):
        self.room_model = room_model
        self.renderer = renderer

        # Default config
        self.config = {
            'lr': 0.01,
            'max_iters': 100,
            'convergence_threshold': 1e-4,
            'optimize_geometry': False,  # Start with appearance only
            'optimize_appearance': True,
            'optimize_lighting': True,
        }
        if config:
            self.config.update(config)

        # Setup optimizer
        self.setup_optimizer()

    def setup_optimizer(self):
        """Configure which parameters to optimize"""
        params = []

        if self.config['optimize_appearance']:
            params.extend([
                self.room_model.wall_colors,
                self.room_model.floor_color,
                self.room_model.ceiling_color
            ])

        if self.config['optimize_lighting']:
            params.extend([
                self.room_model.ambient_light,
                self.room_model.light_location,
                self.room_model.light_intensity
            ])

        if self.config['optimize_geometry']:
            params.extend([
                self.room_model.corners,
                self.room_model.height
            ])

        self.optimizer = torch.optim.Adam(params, lr=self.config['lr'])

    def optimize(self, target_image, camera_position, camera_target,
                 camera_up=None, fov=60):
        """
        Optimize room parameters to match target image

        Args:
            target_image: [H, W, 3] observed image
            camera_position, camera_target, camera_up, fov: camera parameters

        Returns:
            losses: list of loss values per iteration
        """
        losses = []

        # MSE loss
        criterion = nn.MSELoss()

        print("Starting optimization...")
        pbar = tqdm(range(self.config['max_iters']))

        for iter in pbar:
            self.optimizer.zero_grad()

            # Render from current model
            rendered = self.renderer.render(
                self.room_model,
                camera_position,
                camera_target,
                camera_up,
                fov
            )

            # Compute loss
            loss = criterion(rendered, target_image)

            # Backward
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Clamp parameters to valid ranges
            self.room_model.clamp_parameters()

            # Record loss
            loss_val = loss.item()
            losses.append(loss_val)

            # Update progress bar
            pbar.set_description(f"Loss: {loss_val:.6f}")

            # Check convergence
            if iter > 10:
                if abs(losses[-1] - losses[-2]) < self.config['convergence_threshold']:
                    print(f"\nConverged at iteration {iter}")
                    break

        return losses