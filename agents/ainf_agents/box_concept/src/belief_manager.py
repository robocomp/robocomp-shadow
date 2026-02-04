#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Belief Manager - Generic belief lifecycle management.

Handles:
- Point clustering (DBSCAN-like)
- Data association (Hungarian algorithm)
- Belief creation, update, decay, removal
- VFE optimization

This is object-type agnostic - works with any Belief subclass.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Type
from scipy.optimize import linear_sum_assignment
from rich.console import Console

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.transforms import (transform_points_robot_to_room,
                        transform_object_to_robot_frame,
                        transform_object_to_room_frame,
                        transform_box_with_covariance)

console = Console(highlight=False)


class BeliefManager:
    """
    Generic manager for object beliefs using Active Inference.

    This class handles the belief lifecycle independent of object type.
    Specific object types (box, table, chair) are handled by Belief subclasses.
    """

    def __init__(self,
                 belief_class: Type[Belief],
                 config: BeliefConfig,
                 device: torch.device = DEVICE):
        """
        Initialize the belief manager.

        Args:
            belief_class: The Belief subclass to use (e.g., BoxBelief)
            config: Configuration for beliefs
            device: Torch device
        """
        self.belief_class = belief_class
        self.config = config
        self.device = device

        self.beliefs: Dict[int, Belief] = {}
        self.next_belief_id = 0
        self.frame_count = 0

        # Current robot state
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.robot_cov = np.eye(3) * 0.01

        # Optimization parameters (moved from hardcoded values)
        self.lambda_prior = 0.3  # Weight for state prior term in VFE. Loww
        self.optimization_iters = 10  # Number of gradient descent iterations
        self.optimization_lr = 0.05  # Learning rate for gradient descent
        self.grad_clip = 2.0  # Gradient clipping value
        self.merge_belief_dist = 0.3  # Distance threshold for merging beliefs

        # Visualization data
        self.viz_data = {
            'lidar_points_raw': np.array([]),
            'lidar_points_filtered': np.array([]),
            'clusters': [],
            'room_dims': (6.0, 6.0),
            'robot_pose': np.array([0.0, 0.0, 0.0])
        }

        console.print(f"[cyan]BeliefManager initialized on device: {device}")

    def update(self, lidar_points: np.ndarray, robot_pose: np.ndarray,
               robot_cov: np.ndarray, room_dims: Tuple[float, float]) -> List[Belief]:
        """
        Main perception cycle.

        Args:
            lidar_points: [N, 3] LIDAR points in robot frame
            robot_pose: [x, y, theta] robot pose in room frame
            robot_cov: [3, 3] robot pose covariance
            room_dims: (width, depth) of room

        Returns:
            List of current beliefs
        """
        self.frame_count += 1
        self.robot_pose = robot_pose.copy()
        self.robot_cov = robot_cov.copy()
        self.viz_data['room_dims'] = room_dims
        self.viz_data['robot_pose'] = robot_pose.copy()

        if len(lidar_points) == 0:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Transform to room frame
        world_points = transform_points_robot_to_room(lidar_points, robot_pose)
        self.viz_data['lidar_points_raw'] = world_points.copy()

        # Filter wall points
        filtered_points_room, wall_mask = self._filter_wall_points(world_points, room_dims)
        filtered_points_robot = lidar_points[wall_mask]
        self.viz_data['lidar_points_filtered'] = filtered_points_room.copy()

        if len(filtered_points_robot) < self.config.min_cluster_points:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Cluster points
        clusters_robot = self._cluster_points(filtered_points_robot)
        clusters_room = [transform_points_robot_to_room(c, robot_pose) for c in clusters_robot]
        self.viz_data['clusters'] = [c.copy() for c in clusters_room]

        if len(clusters_robot) == 0:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Data association
        associations, unmatched_clusters, unmatched_beliefs = self._associate_clusters(clusters_room)

        # Update matched beliefs
        self._update_matched_beliefs(associations, clusters_room, clusters_robot)

        # Create new beliefs for unmatched clusters
        self._create_new_beliefs(unmatched_clusters, clusters_room)

        # Decay unmatched beliefs
        self._decay_unmatched_beliefs(unmatched_beliefs)

        # Merge overlapping beliefs
        self._merge_overlapping_beliefs()

        return list(self.beliefs.values())

    def _filter_wall_points(self, points: np.ndarray,
                            room_dims: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Filter wall points and return filtered points + mask."""
        width, depth = room_dims
        half_w, half_d = width / 2, depth / 2
        margin = self.config.wall_margin

        dist_left = np.abs(points[:, 0] + half_w)
        dist_right = np.abs(points[:, 0] - half_w)
        dist_back = np.abs(points[:, 1] + half_d)
        dist_front = np.abs(points[:, 1] - half_d)

        min_wall_dist = np.minimum(np.minimum(dist_left, dist_right),
                                   np.minimum(dist_back, dist_front))

        mask = min_wall_dist > margin
        inside_room = (np.abs(points[:, 0]) < half_w - 0.05) & \
                      (np.abs(points[:, 1]) < half_d - 0.05)
        mask = mask & inside_room

        return points[mask], mask

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """DBSCAN-like clustering using XY distance."""
        if len(points) == 0:
            return []

        eps = self.config.cluster_eps
        min_pts = self.config.min_cluster_points

        clusters = []
        points_xy = points[:, :2]
        used = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if used[i]:
                continue

            distances = np.linalg.norm(points_xy - points_xy[i], axis=1)
            neighbor_mask = (distances < eps) & (~used)

            if np.sum(neighbor_mask) < min_pts:
                continue

            cluster_idx = set(np.where(neighbor_mask)[0])
            used[list(cluster_idx)] = True

            # Expand cluster
            to_check = list(cluster_idx)
            while to_check:
                current = to_check.pop(0)
                distances = np.linalg.norm(points_xy - points_xy[current], axis=1)
                new_neighbors = np.where((distances < eps) & (~used))[0]

                for n in new_neighbors:
                    if n not in cluster_idx:
                        cluster_idx.add(n)
                        used[n] = True
                        to_check.append(n)

            cluster = points[list(cluster_idx)]
            if len(cluster) >= min_pts:
                clusters.append(cluster)

        return self._merge_overlapping_clusters(clusters)

    def _merge_overlapping_clusters(self, clusters: List[np.ndarray]) -> List[np.ndarray]:
        """Merge clusters that are too close (NMS-like)."""
        if len(clusters) <= 1:
            return clusters

        merge_dist = self.config.cluster_eps * 2.5
        centroids = [np.mean(c[:, :2], axis=0) for c in clusters]

        merged = []
        used = [False] * len(clusters)

        for i in range(len(clusters)):
            if used[i]:
                continue

            to_merge = [i]
            used[i] = True

            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue

                for k in to_merge:
                    if np.linalg.norm(centroids[k] - centroids[j]) < merge_dist:
                        to_merge.append(j)
                        used[j] = True
                        break

            merged.append(np.vstack([clusters[idx] for idx in to_merge]))

        return merged

    def _associate_clusters(self, clusters: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], set, set]:
        """Associate clusters to beliefs using Hungarian algorithm."""
        if len(clusters) == 0 or len(self.beliefs) == 0:
            return [], set(range(len(clusters))), set(self.beliefs.keys())

        n_clusters = len(clusters)
        belief_ids = list(self.beliefs.keys())
        n_beliefs = len(belief_ids)

        cost_matrix = np.full((n_clusters, n_beliefs), np.inf)

        for k, cluster in enumerate(clusters):
            centroid = np.mean(cluster[:, :2], axis=0)
            cluster_t = torch.tensor(cluster, dtype=DTYPE, device=self.device)

            for j, bid in enumerate(belief_ids):
                belief = self.beliefs[bid]
                bx, by = belief.position

                center_dist = np.sqrt((centroid[0] - bx)**2 + (centroid[1] - by)**2)
                if center_dist > self.config.max_association_distance:
                    continue

                sdf = belief.sdf(cluster_t)
                sdf_cost = torch.mean(sdf ** 2).item()

                cost_matrix[k, j] = center_dist + sdf_cost

        if np.all(np.isinf(cost_matrix)):
            return [], set(range(n_clusters)), set(belief_ids)

        # Replace inf for Hungarian
        max_finite = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1.0
        cost_safe = np.where(np.isinf(cost_matrix), max_finite * 1000, cost_matrix)

        row_ind, col_ind = linear_sum_assignment(cost_safe)

        associations = []
        matched_clusters = set()
        matched_beliefs = set()

        for k, j in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[k, j]):
                associations.append((k, belief_ids[j]))
                matched_clusters.add(k)
                matched_beliefs.add(belief_ids[j])

        unmatched_clusters = set(range(n_clusters)) - matched_clusters
        unmatched_beliefs = set(belief_ids) - matched_beliefs

        return associations, unmatched_clusters, unmatched_beliefs

    def _update_matched_beliefs(self, associations: List[Tuple[int, int]],
                                 clusters_room: List[np.ndarray],
                                 clusters_robot: List[np.ndarray]):
        """Update matched beliefs via VFE minimization."""
        for cluster_idx, belief_id in associations:
            cluster_robot = clusters_robot[cluster_idx]
            cluster_room = clusters_room[cluster_idx]
            belief = self.beliefs[belief_id]

            # Convert to tensors
            pts_robot = torch.tensor(cluster_robot, dtype=DTYPE, device=self.device)
            pts_room = torch.tensor(cluster_room, dtype=DTYPE, device=self.device)

            # Get historical points
            hist_pts, hist_weights = belief.get_historical_points_in_robot_frame(
                self.robot_pose, self.robot_cov)

            # Combine points
            if len(hist_pts) > 0:
                curr_weights = torch.ones(len(pts_robot), dtype=DTYPE, device=self.device)
                all_pts = torch.cat([pts_robot, hist_pts], dim=0)
                all_weights = torch.cat([curr_weights, hist_weights], dim=0)
            else:
                all_pts = pts_robot
                all_weights = None

            # Get prior
            prior_mu, prior_Sigma = belief.propagate_prior()

            # Transform box to robot frame with covariance
            box_mu_robot, box_Sigma_robot = transform_box_with_covariance(
                belief.mu, belief.Sigma, self.robot_pose, self.robot_cov)

            # Optimize
            opt_mu, sdf_mean = self._optimize_belief(
                all_pts, all_weights, box_mu_robot, box_Sigma_robot,
                prior_mu, prior_Sigma, belief.config, belief)

            # Transform back to room frame
            belief.mu = transform_object_to_room_frame(opt_mu, self.robot_pose)
            belief.last_sdf_mean = sdf_mean

            # Add historical points
            current_sdf = belief.sdf(pts_room)
            belief.add_historical_points(pts_room, current_sdf, self.robot_cov)

            # Update Remembered Free Energy for historical points
            # This accumulates SDF error weighted by robot certainty
            belief.update_rfe(self.robot_cov)

            # Update lifecycle
            belief.update_lifecycle(self.frame_count, was_observed=True)

    def _optimize_belief(self, points: torch.Tensor, weights: Optional[torch.Tensor],
                         box_mu: torch.Tensor, box_Sigma: torch.Tensor,
                         prior_mu: torch.Tensor, prior_Sigma: torch.Tensor,
                         config: BeliefConfig, belief: Belief) -> Tuple[torch.Tensor, float]:
        """Optimize belief parameters via VFE minimization."""
        if len(points) < 3:
            return box_mu, 0.0

        if weights is None:
            weights = torch.ones(len(points), dtype=DTYPE, device=points.device)

        mu = box_mu.clone().detach().requires_grad_(True)
        final_sdf_mean = 0.0

        # Store initial angle for debug
        initial_angle = mu[-1].item()

        for iter_idx in range(self.optimization_iters):
            # Compute SDF directly using mu (not belief.mu) to preserve gradient
            from src.object_sdf_prior import get_sdf_function
            belief_type = belief.to_dict().get('type', 'box')
            sdf_func = get_sdf_function(belief_type)
            sdf = sdf_func(points, mu)

            final_sdf_mean = torch.mean(torch.abs(sdf)).item()

            # Debug: print gradient info for chair occasionally
            if belief_type == 'chair' and self.frame_count % 50 == 0 and iter_idx == 0:
                z_values = points[:, 2]
                seat_h = mu[4].item()
                backrest_pts = (z_values > seat_h).sum().item()
                seat_pts = (z_values <= seat_h).sum().item()
                print(f"[DEBUG] Frame {self.frame_count}: pts={len(points)} (seat={seat_pts}, back={backrest_pts}), SDF={final_sdf_mean:.4f}")

            # =================================================================
            # LIKELIHOOD TERM (prediction error from observations)
            # From ACTIVE_INFERENCE_MATH.md Section 4.3:
            # F_likelihood = (1/2σ²) × Σ SDF(p_i, s)²
            # =================================================================
            weighted_likelihood = torch.sum(weights * sdf**2) / len(points)

            # =================================================================
            # PRIOR TERMS - delegated to model-specific implementations
            # Each belief class implements compute_prior_term() with its own priors
            # =================================================================
            object_prior_term = belief.compute_prior_term(mu, self.robot_pose)

            # =================================================================
            # TOTAL VFE = Likelihood + Model-specific Prior
            # =================================================================
            loss = weighted_likelihood + object_prior_term

            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            with torch.no_grad():
                if mu.grad is not None and not torch.isnan(mu.grad).any():
                    grad = mu.grad.clone()

                    # Debug: print ORIGINAL angle gradient for chair (before any processing)
                    if belief_type == 'chair' and self.frame_count % 50 == 0 and iter_idx == 0:
                        print(f"[GRAD] raw_angle_grad={grad[-1].item():.6f}, loss={loss.item():.6f}")

                    # Clip all gradients
                    grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)

                    # Boost angle gradient only for chair (near-square seat makes gradient very small)
                    # But cap the final angle gradient to prevent wild jumps
                    if belief_type == 'chair':
                        angle_lr_multiplier = 100.0  # Reduced from 500
                        grad[-1] = grad[-1] * angle_lr_multiplier
                        # Cap the boosted angle gradient
                        max_angle_grad = 0.5  # Max ~28 degrees per iteration
                        grad[-1] = torch.clamp(grad[-1], -max_angle_grad, max_angle_grad)

                    mu -= self.optimization_lr * grad

                    # Clamp size dimensions (indices 2 to state_dim-2)
                    state_dim = len(mu)
                    for i in range(2, state_dim - 1):
                        mu[i] = torch.clamp(mu[i], config.min_size, config.max_size)

                    # Normalize angle to [-pi, pi] (preserves direction)
                    mu[-1] = torch.atan2(torch.sin(mu[-1]), torch.cos(mu[-1]))

            mu.requires_grad_(True)

        # Debug: print if angle changed significantly
        final_angle = mu[-1].item()
        # Handle wrap-around: compute shortest angular distance
        angle_diff = final_angle - initial_angle
        # Normalize to [-pi, pi]
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        angle_change_deg = np.degrees(angle_diff)

        if abs(angle_change_deg) > 1.0:  # More than 1 degree change
            print(f"[ANGLE DEBUG] Belief {belief.id}: angle changed {angle_change_deg:.1f}° "
                  f"({np.degrees(initial_angle):.1f}° -> {np.degrees(final_angle):.1f}°), "
                  f"SDF={final_sdf_mean:.4f}")

        opt_mu = mu.detach()

        return opt_mu, final_sdf_mean

    def _create_new_beliefs(self, unmatched_clusters: set, clusters: List[np.ndarray]):
        """Create new beliefs for unmatched clusters."""
        for idx in unmatched_clusters:
            cluster = clusters[idx]
            belief = self.belief_class.from_cluster(
                self.next_belief_id, cluster, self.config, self.device)

            if belief is not None:
                self.beliefs[self.next_belief_id] = belief
                self.next_belief_id += 1

    def _decay_unmatched_beliefs(self, unmatched_ids: set):
        """Decay unmatched beliefs and remove if below threshold."""
        to_remove = []

        for bid in list(self.beliefs.keys()):
            if bid in unmatched_ids:
                belief = self.beliefs[bid]
                belief.update_lifecycle(self.frame_count, was_observed=False)

                if belief.should_remove():
                    to_remove.append(bid)

        for bid in to_remove:
            del self.beliefs[bid]

    def _merge_overlapping_beliefs(self):
        """Merge beliefs that are too close (NMS for beliefs)."""
        if len(self.beliefs) <= 1:
            return

        belief_ids = list(self.beliefs.keys())
        to_remove = set()

        for i, id1 in enumerate(belief_ids):
            if id1 in to_remove:
                continue
            b1 = self.beliefs[id1]

            for j in range(i + 1, len(belief_ids)):
                id2 = belief_ids[j]
                if id2 in to_remove:
                    continue
                b2 = self.beliefs[id2]

                x1, y1 = b1.position
                x2, y2 = b2.position
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

                if dist < self.merge_belief_dist:
                    if b1.confidence >= b2.confidence:
                        to_remove.add(id2)
                    else:
                        to_remove.add(id1)
                        break

        for bid in to_remove:
            del self.beliefs[bid]

    def get_beliefs_as_dicts(self) -> List[dict]:
        """Get beliefs as dictionaries for visualization."""
        return [b.to_dict() for b in self.beliefs.values()]

    def get_historical_points_for_viz(self) -> Dict[int, np.ndarray]:
        """Get historical points per belief for visualization."""
        result = {}
        for bid, belief in self.beliefs.items():
            if belief.num_historical_points > 0:
                result[bid] = belief.historical_points.cpu().numpy()
        return result
