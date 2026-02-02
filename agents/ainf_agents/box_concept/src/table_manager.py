#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Table Manager - Thin wrapper for table object management using refactored modules.

This is a specialized BeliefManager for table objects with DSR integration.
"""

import sys
import numpy as np
from typing import List, Tuple

sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Attribute

from src.belief_core import DEVICE, DTYPE
from src.belief_manager import BeliefManager
from src.table_belief import TableBelief, TableBeliefConfig
from src.object_sdf_prior import compute_table_sdf
from src.box_belief import BoxBelief  # For debug_belief_vs_gt


class TableManager(BeliefManager):
    """Specialized BeliefManager for table objects with DSR integration."""

    def __init__(self, g, agent_id: int, config: TableBeliefConfig = None):
        """
        Initialize table manager.

        Args:
            g: DSR graph instance
            agent_id: Agent identifier for DSR operations
            config: Table belief configuration (uses defaults if None)
        """
        config = config or TableBeliefConfig()
        super().__init__(TableBelief, config, DEVICE)
        self.g = g
        self.agent_id = agent_id

    @staticmethod
    def compute_table_sdf(points_xyz, table_params):
        """Compute SDF for table (static method for external use)."""
        return compute_table_sdf(points_xyz, table_params)

    def get_beliefs_as_dicts(self) -> List[dict]:
        """Get all beliefs as dictionaries for visualization."""
        return [belief.to_dict() for belief in self.beliefs.values()]

    def get_historical_points_for_viz(self) -> dict:
        """Get historical points for visualization, keyed by belief ID."""
        result = {}
        for bid, belief in self.beliefs.items():
            if belief.num_historical_points > 0:
                result[bid] = belief.historical_points.cpu().numpy()
        return result

    @staticmethod
    def debug_belief_vs_gt(belief_dict, gt_cx=0., gt_cy=0., gt_w=1.0, gt_h=0.6,
                           gt_table_height=0.75, gt_theta=0.):
        """Print debug info comparing belief against Ground Truth."""
        est_cx = belief_dict.get('cx', 0)
        est_cy = belief_dict.get('cy', 0)
        est_w = belief_dict.get('width', 0)
        est_h = belief_dict.get('depth', 0)
        est_table_height = belief_dict.get('table_height', 0)
        est_leg_length = belief_dict.get('leg_length', 0)
        est_theta = belief_dict.get('angle', 0)

        pos_error = np.sqrt((est_cx - gt_cx)**2 + (est_cy - gt_cy)**2)
        w_error = est_w - gt_w
        h_error = est_h - gt_h
        height_error = est_table_height - gt_table_height
        theta_error = np.degrees(est_theta - gt_theta)

        print(f"\n{'='*60}")
        print(f"TABLE BELIEF vs GT ({gt_w}x{gt_h}, h={gt_table_height}m at ({gt_cx},{gt_cy}))")
        print(f"{'='*60}")
        print(f"Position:     Est=({est_cx:.3f}, {est_cy:.3f})  GT=({gt_cx:.3f}, {gt_cy:.3f})  Error={pos_error:.4f}m")
        print(f"Width(X):     Est={est_w:.3f}m  GT={gt_w:.3f}m  Error={w_error:+.4f}m ({100*w_error/gt_w:+.1f}%)")
        print(f"Depth(Y):     Est={est_h:.3f}m  GT={gt_h:.3f}m  Error={h_error:+.4f}m ({100*h_error/gt_h:+.1f}%)")
        print(f"TableHeight:  Est={est_table_height:.3f}m  GT={gt_table_height:.3f}m  Error={height_error:+.4f}m")
        print(f"LegLength:    Est={est_leg_length:.3f}m")
        print(f"Angle:        Est={np.degrees(est_theta):.1f}°  GT={np.degrees(gt_theta):.1f}°  Error={theta_error:+.1f}°")
        print(f"{'='*60}\n")

    def update_dsr(self):
        """Update DSR graph with detected tables."""
        # TODO: Implement DSR integration for tables
        # Similar to box_manager but with table-specific attributes
        pass
