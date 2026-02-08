#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""TV Manager - Manager for TV obstacle management using refactored modules."""

import sys
import numpy as np
from typing import List, Tuple

sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Attribute

from src.belief_core import DEVICE, DTYPE
from src.belief_manager import BeliefManager
from src.objects.tv.belief import TVBelief, TVBeliefConfig
from src.objects.tv.sdf import compute_tv_sdf


class TVManager(BeliefManager):
    """Specialized BeliefManager for TV obstacles with DSR integration."""

    def __init__(self, g, agent_id: int, config: TVBeliefConfig = None):
        config = config or TVBeliefConfig()
        super().__init__(TVBelief, config, DEVICE)
        self.g = g
        self.agent_id = agent_id

    @staticmethod
    def compute_tv_sdf(points_xyz, tv_params):
        return compute_tv_sdf(points_xyz, tv_params)

    @staticmethod
    def debug_belief_vs_gt(belief_dict, gt_cx=0., gt_cy=0., gt_width=1.0, gt_height=0.56,
                           gt_depth=0.08, gt_theta=0.):
        """Print debug info comparing belief against Ground Truth."""
        est_cx = belief_dict.get('cx', 0)
        est_cy = belief_dict.get('cy', 0)
        est_width = belief_dict.get('width', 1.0)
        est_height = belief_dict.get('height', 0.56)
        est_depth = belief_dict.get('depth', 0.08)
        est_theta = belief_dict.get('angle', 0)

        pos_error = np.sqrt((est_cx - gt_cx)**2 + (est_cy - gt_cy)**2)
        width_error = est_width - gt_width
        height_error = est_height - gt_height
        depth_error = est_depth - gt_depth
        theta_error = np.degrees(est_theta - gt_theta)

        # Aspect ratios
        est_aspect = est_width / (est_height + 0.01)
        gt_aspect = gt_width / (gt_height + 0.01)

        # Width/depth ratio (thin panel check)
        est_wd_ratio = est_width / (est_depth + 0.01)
        gt_wd_ratio = gt_width / (gt_depth + 0.01)

        print(f"\n{'='*60}")
        print(f"TV BELIEF vs GT (GT: {gt_width}x{gt_height}x{gt_depth}m at ({gt_cx},{gt_cy}))")
        print(f"{'='*60}")
        print(f"Position:     Est=({est_cx:.3f}, {est_cy:.3f})  Error={pos_error:.4f}m")
        print(f"Width:        Est={est_width:.3f}m  GT={gt_width:.3f}m  Err={width_error:+.3f}m")
        print(f"Height:       Est={est_height:.3f}m  GT={gt_height:.3f}m  Err={height_error:+.3f}m")
        print(f"Depth:        Est={est_depth:.3f}m  GT={gt_depth:.3f}m  Err={depth_error:+.3f}m")
        print(f"Angle:        Est={np.degrees(est_theta):.1f}°  GT={np.degrees(gt_theta):.1f}°  Err={theta_error:+.1f}°")
        print(f"Screen aspect: Est={est_aspect:.2f}  GT={gt_aspect:.2f} (16:9={16/9:.2f})")
        print(f"W/D ratio:    Est={est_wd_ratio:.1f}  GT={gt_wd_ratio:.1f} (thin panel)")

        sdf_mean = belief_dict.get('last_sdf_mean', 0)
        hist_pts = belief_dict.get('num_historical_points', 0)
        print(f"SDF mean:     {sdf_mean:.4f}")
        print(f"Hist pts:     N={hist_pts}")
        print(f"{'='*60}\n")

    def update(self, lidar_points: np.ndarray, robot_pose: np.ndarray,
               robot_cov: np.ndarray, room_dims: Tuple[float, float]) -> List[TVBelief]:
        beliefs = super().update(lidar_points, robot_pose, robot_cov, room_dims)
        return beliefs

    def _update_dsr(self):
        """Update DSR graph with TV nodes."""
        for bid, b in self.beliefs.items():
            name = f"tv_{bid}"
            node = self.g.get_node(name)
            if node:
                node.attrs["pos_x"] = Attribute(float(b.cx * 1000), self.agent_id)
                node.attrs["pos_y"] = Attribute(float(b.cy * 1000), self.agent_id)
                node.attrs["width"] = Attribute(float(b.width * 1000), self.agent_id)
                node.attrs["height"] = Attribute(float(b.height * 1000), self.agent_id)
                node.attrs["depth"] = Attribute(float(b.depth * 1000), self.agent_id)
                node.attrs["rotation_angle"] = Attribute(float(b.angle), self.agent_id)
                node.attrs["confidence"] = Attribute(float(b.confidence), self.agent_id)
                self.g.update_node(node)
            else:
                node = Node(agent_id=self.agent_id, type="tv", name=name)
                node.attrs["pos_x"] = Attribute(float(b.cx * 1000), self.agent_id)
                node.attrs["pos_y"] = Attribute(float(b.cy * 1000), self.agent_id)
                node.attrs["width"] = Attribute(float(b.width * 1000), self.agent_id)
                node.attrs["height"] = Attribute(float(b.height * 1000), self.agent_id)
                node.attrs["depth"] = Attribute(float(b.depth * 1000), self.agent_id)
                node.attrs["rotation_angle"] = Attribute(float(b.angle), self.agent_id)
                node.attrs["confidence"] = Attribute(float(b.confidence), self.agent_id)
                node.attrs["color"] = Attribute("DarkGray", self.agent_id)
                self.g.insert_node(node)

