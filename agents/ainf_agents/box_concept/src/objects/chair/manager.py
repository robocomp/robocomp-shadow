#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Chair Manager - Specialized BeliefManager for chair objects."""
import sys
import numpy as np
from typing import List
sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Attribute
from src.belief_core import DEVICE
from src.belief_manager import BeliefManager
from src.objects.chair.belief import ChairBelief, ChairBeliefConfig
from src.objects.chair.sdf import compute_chair_sdf
class ChairManager(BeliefManager):
    def __init__(self, g, agent_id: int, config: ChairBeliefConfig = None):
        config = config or ChairBeliefConfig()
        super().__init__(ChairBelief, config, DEVICE)
        self.g = g
        self.agent_id = agent_id
    @staticmethod
    def compute_chair_sdf(points_xyz, chair_params):
        return compute_chair_sdf(points_xyz, chair_params)
    def get_beliefs_as_dicts(self) -> List[dict]:
        return [belief.to_dict() for belief in self.beliefs.values()]
    def get_historical_points_for_viz(self) -> dict:
        result = {}
        for bid, belief in self.beliefs.items():
            if belief.num_historical_points > 0:
                result[bid] = belief.historical_points.cpu().numpy()
        return result
    @staticmethod
    def debug_belief_vs_gt(belief_dict, gt_cx=0., gt_cy=0., gt_seat_w=0.45, gt_seat_d=0.45,
                           gt_seat_h=0.45, gt_back_h=0.40, gt_theta=0.):
        est_cx = belief_dict.get('cx', 0)
        est_cy = belief_dict.get('cy', 0)
        est_seat_w = belief_dict.get('seat_width', 0)
        est_seat_d = belief_dict.get('seat_depth', 0)
        est_seat_h = belief_dict.get('seat_height', 0)
        est_back_h = belief_dict.get('back_height', 0)
        est_theta = belief_dict.get('angle', 0)  # In ROOM frame
        back_t = belief_dict.get('back_thickness', 0.05)
        sdf_mean = belief_dict.get('sdf_mean', 0)
        pos_error = np.sqrt((est_cx - gt_cx)**2 + (est_cy - gt_cy)**2)
        num_hist = belief_dict.get('num_historical_points', 0)
        hist_rfe = belief_dict.get('historical_rfe_stats', {})

        # Normalize angle to [-180, 180] for display
        angle_deg = np.degrees(est_theta)
        while angle_deg > 180: angle_deg -= 360
        while angle_deg < -180: angle_deg += 360

        # Compute angle error (handling wrap-around)
        angle_error = angle_deg - np.degrees(gt_theta)
        while angle_error > 180: angle_error -= 360
        while angle_error < -180: angle_error += 360

        print(f"\n{'='*60}")
        print(f"CHAIR BELIEF vs GT (seat {gt_seat_w}x{gt_seat_d}, h={gt_seat_h}m)")
        print(f"{'='*60}")
        print(f"Position:     Est=({est_cx:.3f}, {est_cy:.3f})  Error={pos_error:.4f}m")
        print(f"SeatWidth:    Est={est_seat_w:.3f}m  GT={gt_seat_w:.3f}m  Err={est_seat_w-gt_seat_w:+.3f}m")
        print(f"SeatDepth:    Est={est_seat_d:.3f}m  GT={gt_seat_d:.3f}m  Err={est_seat_d-gt_seat_d:+.3f}m")
        print(f"SeatHeight:   Est={est_seat_h:.3f}m  GT={gt_seat_h:.3f}m  Err={est_seat_h-gt_seat_h:+.3f}m")
        print(f"BackHeight:   Est={est_back_h:.3f}m  GT={gt_back_h:.3f}m  Err={est_back_h-gt_back_h:+.3f}m")
        print(f"Angle(room):  Est={angle_deg:.1f}°  GT={np.degrees(gt_theta):.1f}°  Err={angle_error:+.1f}°")
        print(f"SDF mean:     {sdf_mean:.4f}")
        if num_hist > 0:
            print(f"Hist pts:     N={num_hist}, RFE_mean={hist_rfe.get('mean', 0):.4f}")
            # The SDF mean tells us if historical points are actually on the surface
            # If SDF mean is low but points don't appear on the drawn chair,
            # then there's a visualization bug
        print(f"{'='*60}\n")
        print(f"[NOTE] If SDF is low but points don't match drawn chair, check visualization!")
    def update_dsr(self):
        pass
