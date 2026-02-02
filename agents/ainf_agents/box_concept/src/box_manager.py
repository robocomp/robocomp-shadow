#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Box Manager - Thin wrapper for box obstacle management using refactored modules."""

import sys
import numpy as np
from typing import List, Tuple

sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Attribute

from src.belief_core import DEVICE, DTYPE
from src.belief_manager import BeliefManager
from src.box_belief import BoxBelief, BoxBeliefConfig
from src.sdf_functions import compute_box_sdf


class BoxManager(BeliefManager):
    """Specialized BeliefManager for 3D box obstacles with DSR integration."""

    def __init__(self, g, agent_id: int, config: BoxBeliefConfig = None):
        config = config or BoxBeliefConfig()
        super().__init__(BoxBelief, config, DEVICE)
        self.g = g
        self.agent_id = agent_id

    @staticmethod
    def compute_box_sdf(points_xyz, box_params):
        return compute_box_sdf(points_xyz, box_params)

    @staticmethod
    def debug_belief_vs_gt(belief, gt_cx=0., gt_cy=0., gt_w=0.5, gt_h=0.5, gt_d=0.5, gt_theta=0.):
        BoxBelief.debug_vs_gt(belief, gt_cx, gt_cy, gt_w, gt_h, gt_d, gt_theta)

    def update(self, lidar_points: np.ndarray, robot_pose: np.ndarray,
               robot_cov: np.ndarray, room_dims: Tuple[float, float]) -> List[BoxBelief]:
        beliefs = super().update(lidar_points, robot_pose, robot_cov, room_dims)
        return beliefs

    def _update_dsr(self):
        for bid, b in self.beliefs.items():
            name = f"box_{bid}"
            node = self.g.get_node(name)
            if node:
                node.attrs["pos_x"] = Attribute(float(b.cx*1000), self.agent_id)
                node.attrs["pos_y"] = Attribute(float(b.cy*1000), self.agent_id)
                node.attrs["width"] = Attribute(float(b.width*1000), self.agent_id)
                node.attrs["height"] = Attribute(float(b.height*1000), self.agent_id)
                node.attrs["depth"] = Attribute(float(b.depth*1000), self.agent_id)
                node.attrs["rotation_angle"] = Attribute(float(b.angle), self.agent_id)
                node.attrs["confidence"] = Attribute(float(b.confidence), self.agent_id)
                self.g.update_node(node)
            else:
                node = Node(agent_id=self.agent_id, type="box", name=name)
                node.attrs["pos_x"] = Attribute(float(b.cx*1000), self.agent_id)
                node.attrs["pos_y"] = Attribute(float(b.cy*1000), self.agent_id)
                node.attrs["width"] = Attribute(float(b.width*1000), self.agent_id)
                node.attrs["height"] = Attribute(float(b.height*1000), self.agent_id)
                node.attrs["depth"] = Attribute(float(b.depth*1000), self.agent_id)
                node.attrs["rotation_angle"] = Attribute(float(b.angle), self.agent_id)
                node.attrs["confidence"] = Attribute(float(b.confidence), self.agent_id)
                node.attrs["color"] = Attribute("Orange", self.agent_id)
                self.g.insert_node(node)

# Backward compatibility
RectangleBelief = BoxBelief
RectangleBeliefConfig = BoxBeliefConfig
