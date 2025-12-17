#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append('/opt/robocomp/lib')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Now import from src folder
from src.room_model import ParametricRoom
from src.room_renderer import DifferentiableRoomRenderer
from src.optimizer import RoomOptimizer

console = Console(highlight=False)


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        console.print(f"[bold green]Using device: {self.device}[/bold green]")

        # Test state
        self.test_completed = False
        self.iteration_count = 0

        # Initialize renderer
        console.print("[bold cyan]Initializing renderer...[/bold cyan]")
        self.renderer = DifferentiableRoomRenderer(image_size=256, device=self.device)

        # Create ground truth room
        console.print("[bold cyan]Creating ground truth room...[/bold cyan]")
        self.gt_corners = torch.tensor([
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0]
        ], device=self.device)
        self.gt_height = 3.0

        self.gt_room = ParametricRoom(self.gt_corners, self.gt_height, device=self.device)

        # Set VERY BRIGHT distinctive ground truth appearance
        with torch.no_grad():
            self.gt_room.wall_colors[0] = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # Pure red
            self.gt_room.wall_colors[1] = torch.tensor([0.0, 1.0, 0.0], device=self.device)  # Pure green
            self.gt_room.wall_colors[2] = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # Pure blue
            self.gt_room.wall_colors[3] = torch.tensor([1.0, 1.0, 0.0], device=self.device)  # Pure yellow
            self.gt_room.floor_color[:] = torch.tensor([0.8, 0.6, 0.4], device=self.device)  # Bright brown
            self.gt_room.ceiling_color[:] = torch.tensor([1.0, 1.0, 1.0], device=self.device)  # Pure white

        # Camera setup - MULTIPLE VIEWS
        # We'll create several cameras to visualize the room from different angles
        self.camera_configs = [
            {
                'name': 'corner_view',
                'pos': torch.tensor([-3.5, -3.5, 2.5], device=self.device),
                'target': torch.tensor([0.0, 0.0, 1.5], device=self.device),
                'fov': 80.0
            },
            {
                'name': 'front_view',
                'pos': torch.tensor([0.0, -4.5, 1.5], device=self.device),
                'target': torch.tensor([0.0, 0.0, 1.5], device=self.device),
                'fov': 70.0
            },
            {
                'name': 'top_view',
                'pos': torch.tensor([0.0, -1.0, 5.0], device=self.device),
                'target': torch.tensor([0.0, 0.0, 0.0], device=self.device),
                'fov': 90.0
            },
            {
                'name': 'inside_view',
                'pos': torch.tensor([0.0, 0.0, 1.5], device=self.device),
                'target': torch.tensor([1.0, 1.0, 1.5], device=self.device),
                'fov': 90.0
            }
        ]

        # Use corner view for optimization (index 0)
        self.current_camera = self.camera_configs[0]
        self.camera_pos = self.current_camera['pos']
        self.camera_target = self.current_camera['target']
        self.camera_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.fov = self.current_camera['fov']

        # Render ground truth image
        console.print("[bold cyan]Rendering ground truth image...[/bold cyan]")
        with torch.no_grad():
            self.gt_image = self.renderer.render(
                self.gt_room,
                self.camera_pos,
                self.camera_target,
                camera_up=self.camera_up,
                fov=self.fov
            )
        console.print(f"[green]Ground truth image shape: {self.gt_image.shape}[/green]")

        # Create test room with noisy initialization
        console.print("[bold cyan]Creating test room with noisy parameters...[/bold cyan]")
        init_corners = self.gt_corners + torch.randn_like(self.gt_corners) * 0.1
        init_height = self.gt_height + torch.randn(1, device=self.device).item() * 0.1

        self.test_room = ParametricRoom(init_corners, init_height, device=self.device)

        # Initialize with VERY DIFFERENT random appearance (will be optimized)
        with torch.no_grad():
            # Random colors in mid-range (not too dark, not matching GT)
            self.test_room.wall_colors[:] = torch.rand(4, 3, device=self.device) * 0.4 + 0.3  # 0.3-0.7 range
            self.test_room.floor_color[:] = torch.rand(3, device=self.device) * 0.4 + 0.3
            self.test_room.ceiling_color[:] = torch.rand(3, device=self.device) * 0.4 + 0.3

            # Print initial colors for debugging
            console.print("[bold yellow]Initial test room colors (random):[/bold yellow]")
            console.print(f"  Wall colors:\n{self.test_room.wall_colors}")
            console.print(f"  Floor color: {self.test_room.floor_color}")
            console.print(f"  Ceiling color: {self.test_room.ceiling_color}")

        # Setup optimizer
        console.print("[bold cyan]Setting up optimizer...[/bold cyan]")
        self.optimizer_config = {
            'lr': 0.01,
            'max_iters': 100,
            'convergence_threshold': 1e-4,
            'optimize_geometry': False,  # Fix geometry, optimize appearance only
            'optimize_appearance': True,
            'optimize_lighting': True
        }

        self.room_optimizer = RoomOptimizer(
            self.test_room,
            self.renderer,
            config=self.optimizer_config
        )

        self.losses = []

        console.print("[bold green]Initialization complete! Ready to optimize.[/bold green]")

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'test_completed') and self.test_completed:
            console.print("[bold yellow]Cleaning up...[/bold yellow]")

    @QtCore.Slot()
    def compute(self):
        """Run optimization iterations"""

        if self.test_completed:
            return

        # Run one optimization iteration per compute cycle
        if self.iteration_count == 0:
            console.print("[bold magenta]Starting optimization process...[/bold magenta]")

        # Zero gradients
        self.room_optimizer.optimizer.zero_grad()

        # Render from current model
        rendered = self.renderer.render(
            self.test_room,
            self.camera_pos,
            self.camera_target,
            camera_up=self.camera_up,
            fov=self.fov
        )

        # Compute loss
        criterion = torch.nn.MSELoss()
        loss = criterion(rendered, self.gt_image)

        # Backward pass
        loss.backward()

        # Optimization step
        self.room_optimizer.optimizer.step()

        # Clamp parameters
        self.test_room.clamp_parameters()

        # Record loss
        loss_val = loss.item()
        self.losses.append(loss_val)

        # Print progress every 10 iterations
        if self.iteration_count % 10 == 0:
            console.print(f"[cyan]Iteration {self.iteration_count}: Loss = {loss_val:.6f}[/cyan]")

        self.iteration_count += 1

        # Check for completion
        if self.iteration_count >= self.optimizer_config['max_iters']:
            self.finalize_optimization()
        elif self.iteration_count > 10:
            # Check convergence
            if abs(self.losses[-1] - self.losses[-2]) < self.optimizer_config['convergence_threshold']:
                console.print(f"[bold green]Converged at iteration {self.iteration_count}![/bold green]")
                self.finalize_optimization()

    def finalize_optimization(self):
        """Finalize and visualize optimization results"""

        self.test_completed = True

        console.print("[bold green]Optimization complete![/bold green]")
        console.print(f"[green]Total iterations: {self.iteration_count}[/green]")
        console.print(f"[green]Final loss: {self.losses[-1]:.6f}[/green]")

        # Render final optimized result
        console.print("[bold cyan]Rendering final optimized result...[/bold cyan]")
        with torch.no_grad():
            optimized_image = self.renderer.render(
                self.test_room,
                self.camera_pos,
                self.camera_target,
                camera_up=self.camera_up,
                fov=self.fov
            )

        # Print optimized parameters
        console.print("\n[bold yellow]=== Optimized Parameters ===[/bold yellow]")
        state = self.test_room.get_state_dict()
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                if value.size <= 12:  # Print small arrays
                    console.print(f"[yellow]{key}:[/yellow] {value}")
                else:
                    console.print(f"[yellow]{key}:[/yellow] {value.shape}")
            else:
                console.print(f"[yellow]{key}:[/yellow] {value}")

        # Compare with ground truth
        console.print("\n[bold yellow]=== Ground Truth Parameters ===[/bold yellow]")
        gt_state = self.gt_room.get_state_dict()
        for key, value in gt_state.items():
            if isinstance(value, np.ndarray):
                if value.size <= 12:
                    console.print(f"[yellow]{key}:[/yellow] {value}")
                else:
                    console.print(f"[yellow]{key}:[/yellow] {value.shape}")
            else:
                console.print(f"[yellow]{key}:[/yellow] {value}")

        # Compare optimized vs ground truth parameters
        console.print("\n[bold cyan]=== Parameter Comparison ===[/bold cyan]")

        # Wall colors comparison
        gt_walls = self.gt_room.wall_colors.detach().cpu().numpy()
        opt_walls = self.test_room.wall_colors.detach().cpu().numpy()
        console.print("[yellow]Wall Colors Error:[/yellow]")
        for i in range(4):
            error = np.linalg.norm(gt_walls[i] - opt_walls[i])
            console.print(f"  Wall {i}: GT={gt_walls[i]}, Opt={opt_walls[i]}, Error={error:.4f}")

        # Floor color comparison
        gt_floor = self.gt_room.floor_color.detach().cpu().numpy()
        opt_floor = self.test_room.floor_color.detach().cpu().numpy()
        floor_error = np.linalg.norm(gt_floor - opt_floor)
        console.print(f"[yellow]Floor Color:[/yellow] GT={gt_floor}, Opt={opt_floor}, Error={floor_error:.4f}")

        # Ceiling color comparison
        gt_ceiling = self.gt_room.ceiling_color.detach().cpu().numpy()
        opt_ceiling = self.test_room.ceiling_color.detach().cpu().numpy()
        ceiling_error = np.linalg.norm(gt_ceiling - opt_ceiling)
        console.print(f"[yellow]Ceiling Color:[/yellow] GT={gt_ceiling}, Opt={opt_ceiling}, Error={ceiling_error:.4f}")

        # Total parameter error
        total_error = (np.sum([np.linalg.norm(gt_walls[i] - opt_walls[i]) for i in range(4)]) +
                       floor_error + ceiling_error)
        console.print(f"\n[bold green]Total Color Error: {total_error:.4f}[/bold green]")

        # Create visualization
        console.print("\n[bold cyan]Creating visualization...[/bold cyan]")
        self.visualize_results(optimized_image)

        console.print("[bold green]Test complete! Check optimization_result.png for visualization.[/bold green]")

    def visualize_results(self, optimized_image):
        """Create and save visualization with multiple camera angles"""

        # Create a larger figure with 3 rows
        fig = plt.figure(figsize=(16, 18))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Row 1: Multiple views of ground truth
        console.print("[bold cyan]Rendering ground truth from multiple angles...[/bold cyan]")
        for i, cam_config in enumerate(self.camera_configs):
            with torch.no_grad():
                gt_view = self.renderer.render(
                    self.gt_room,
                    cam_config['pos'],
                    cam_config['target'],
                    camera_up=self.camera_up,
                    fov=cam_config['fov']
                )

            ax = fig.add_subplot(gs[0, i])
            ax.imshow(gt_view.detach().cpu().numpy())
            ax.set_title(f"GT: {cam_config['name']}", fontsize=10, fontweight='bold')
            ax.axis('off')

        # Row 2: Multiple views of optimized
        console.print("[bold cyan]Rendering optimized from multiple angles...[/bold cyan]")
        for i, cam_config in enumerate(self.camera_configs):
            with torch.no_grad():
                opt_view = self.renderer.render(
                    self.test_room,
                    cam_config['pos'],
                    cam_config['target'],
                    camera_up=self.camera_up,
                    fov=cam_config['fov']
                )

            ax = fig.add_subplot(gs[1, i])
            ax.imshow(opt_view.detach().cpu().numpy())
            ax.set_title(f"Opt: {cam_config['name']}", fontsize=10, fontweight='bold')
            ax.axis('off')

        # Row 3: Loss curve and difference (spanning 2 columns each)
        ax_diff = fig.add_subplot(gs[2, 0:2])
        gt_np = self.gt_image.detach().cpu().numpy()
        opt_np = optimized_image.detach().cpu().numpy()
        diff = np.abs(gt_np - opt_np)
        im = ax_diff.imshow(diff, cmap='hot')
        ax_diff.set_title("Absolute Difference (optimization view)", fontsize=12, fontweight='bold')
        ax_diff.axis('off')
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

        # Loss curve
        ax_loss = fig.add_subplot(gs[2, 2:4])
        ax_loss.plot(self.losses, linewidth=2, color='blue')
        ax_loss.set_xlabel("Iteration", fontsize=12)
        ax_loss.set_ylabel("Loss (MSE)", fontsize=12)
        ax_loss.set_title("Optimization Progress", fontsize=12, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)

        if max(self.losses) > 0.01:
            ax_loss.set_yscale('log')

        final_loss = self.losses[-1]
        initial_loss = self.losses[0]

        if initial_loss > 1e-6:
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            stats_text = f'Initial: {initial_loss:.6f}\nFinal: {final_loss:.6f}\nImprovement: {improvement:.1f}%'
        else:
            absolute_improvement = initial_loss - final_loss
            stats_text = f'Initial: {initial_loss:.6f}\nFinal: {final_loss:.6f}\nΔ: {absolute_improvement:.6f}'

        ax_loss.text(
            0.5, 0.95,
            stats_text,
            transform=ax_loss.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.savefig("optimization_result.png", dpi=150, bbox_inches='tight')
        console.print("[green]Saved multi-view visualization to optimization_result.png[/green]")
        plt.show()

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)