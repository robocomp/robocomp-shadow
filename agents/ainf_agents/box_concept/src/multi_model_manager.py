#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Manager - Belief Manager with Bayesian Model Selection

Extends BeliefManager to support multiple competing model hypotheses per object.
Uses ModelSelector to manage model identity as a discrete latent variable.

Usage:
    manager = MultiModelManager(
        model_classes={
            'table': (TableBelief, TableBeliefConfig()),
            'chair': (ChairBelief, ChairBeliefConfig())
        }
    )

    # In compute loop:
    detected_objects = manager.update(lidar_points, robot_pose, robot_cov, room_dims)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Type
from scipy.optimize import linear_sum_assignment
from rich.console import Console

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.model_selector import (
    ModelSelector, MultiModelBelief, ModelHypothesis,
    ModelSelectorConfig, ModelState
)
from src.transforms import (
    transform_points_robot_to_room,
    transform_object_to_robot_frame,
    transform_object_to_room_frame,
    transform_box_with_covariance
)
from src.object_sdf_prior import get_sdf_function

console = Console(highlight=False)


class MultiModelManager:
    """
    Manager for objects with uncertain model identity.

    Each detected cluster creates a MultiModelBelief that maintains
    parallel hypotheses until commitment criteria are met.
    """

    def __init__(self,
                 model_classes: Dict[str, Tuple[Type[Belief], BeliefConfig]],
                 selector_config: ModelSelectorConfig = None,
                 device: torch.device = DEVICE):
        """
        Initialize the multi-model manager.

        Args:
            model_classes: Dict mapping model_type → (BeliefClass, config)
            selector_config: Configuration for model selection
            device: Torch device
        """
        self.model_classes = model_classes
        self.selector_config = selector_config or ModelSelectorConfig()
        self.device = device

        # Create model selector
        self.selector = ModelSelector(
            model_classes=model_classes,
            config=self.selector_config,
            device=device
        )

        # Active multi-model beliefs
        self.beliefs: Dict[int, MultiModelBelief] = {}

        # Frame counter
        self.frame_count = 0

        # Robot state
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.robot_cov = np.eye(3) * 0.01

        # Clustering parameters (use from first model's config)
        first_config = list(model_classes.values())[0][1]
        self.cluster_eps = getattr(first_config, 'cluster_eps', 0.15)
        self.min_cluster_points = getattr(first_config, 'min_cluster_points', 15)
        self.wall_margin = getattr(first_config, 'wall_margin', 0.25)
        self.max_association_distance = getattr(first_config, 'max_association_distance', 0.8)

        # Optimization parameters
        self.optimization_iters = 10  # Reduced for better performance
        self.optimization_lr = 0.05
        self.grad_clip = 2.0

        # Early exit threshold - skip optimization if SDF error is below this
        self.early_exit_sdf_threshold = 0.05  # Higher threshold for more early exits when stable

        # Debug counters
        self._early_exits = 0
        self._full_optimizations = 0

        # Visualization data
        self.viz_data = {
            'lidar_points_raw': np.array([]),
            'lidar_points_filtered': np.array([]),
            'clusters': [],
            'room_dims': (6.0, 6.0),
            'robot_pose': np.array([0.0, 0.0, 0.0])
        }

        console.print(f"[cyan]MultiModelManager initialized with models: {list(model_classes.keys())}")

    def update(self, lidar_points: np.ndarray, robot_pose: np.ndarray,
               robot_cov: np.ndarray, room_dims: Tuple[float, float]) -> List[MultiModelBelief]:
        """
        Main perception cycle.

        Args:
            lidar_points: [N, 3] LIDAR points in robot frame
            robot_pose: [x, y, theta] robot pose in room frame
            robot_cov: [3, 3] robot pose covariance
            room_dims: (width, depth) of room

        Returns:
            List of current MultiModelBelief objects
        """
        import time
        t_start = time.perf_counter()

        self.frame_count += 1
        self.robot_pose = robot_pose.copy()
        self.robot_cov = robot_cov.copy()
        self.viz_data['room_dims'] = room_dims
        self.viz_data['robot_pose'] = robot_pose.copy()

        if len(lidar_points) == 0:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Transform to room frame
        t0 = time.perf_counter()
        world_points = transform_points_robot_to_room(lidar_points, robot_pose)
        self.viz_data['lidar_points_raw'] = world_points.copy()
        t_transform = (time.perf_counter() - t0) * 1000

        # Filter wall points
        t0 = time.perf_counter()
        filtered_points_room, wall_mask = self._filter_wall_points(world_points, room_dims)
        filtered_points_robot = lidar_points[wall_mask]
        self.viz_data['lidar_points_filtered'] = filtered_points_room.copy()
        t_filter = (time.perf_counter() - t0) * 1000

        if len(filtered_points_robot) < self.min_cluster_points:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Cluster points
        t0 = time.perf_counter()
        clusters_robot = self._cluster_points(filtered_points_robot)
        clusters_room = [transform_points_robot_to_room(c, robot_pose) for c in clusters_robot]
        self.viz_data['clusters'] = [c.copy() for c in clusters_room]
        t_cluster = (time.perf_counter() - t0) * 1000

        if len(clusters_robot) == 0:
            self._decay_unmatched_beliefs(set(self.beliefs.keys()))
            return list(self.beliefs.values())

        # Data association using active belief's SDF
        t0 = time.perf_counter()
        associations, unmatched_clusters, unmatched_beliefs = self._associate_clusters(clusters_room)
        t_assoc = (time.perf_counter() - t0) * 1000

        # Update matched beliefs (all hypotheses)
        t0 = time.perf_counter()
        self._update_matched_beliefs(associations, clusters_room, clusters_robot)
        t_update = (time.perf_counter() - t0) * 1000

        # Create new multi-model beliefs for unmatched clusters
        t0 = time.perf_counter()
        self._create_new_beliefs(unmatched_clusters, clusters_room)
        t_create = (time.perf_counter() - t0) * 1000

        # Decay unmatched beliefs
        self._decay_unmatched_beliefs(unmatched_beliefs)

        # Check for model commitment
        self._process_commitments()

        # Debug output with timing
        if self.frame_count % self.selector_config.debug_interval == 0:
            t_total = (time.perf_counter() - t_start) * 1000
            console.print(f"[dim]Manager: {t_total:.1f}ms (trans:{t_transform:.1f} filt:{t_filter:.1f} clust:{t_cluster:.1f} assoc:{t_assoc:.1f} upd:{t_update:.1f} new:{t_create:.1f}) early:{self._early_exits} full:{self._full_optimizations}[/dim]")
            self._early_exits = 0
            self._full_optimizations = 0
            self._debug_print()

        return list(self.beliefs.values())

    def _filter_wall_points(self, points: np.ndarray,
                            room_dims: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Filter wall points and return filtered points + mask."""
        width, depth = room_dims
        half_w, half_d = width / 2, depth / 2
        margin = self.wall_margin

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
        """DBSCAN clustering using sklearn (fast C implementation)."""
        if len(points) == 0:
            return []

        from sklearn.cluster import DBSCAN

        # Use only XY for clustering
        points_xy = points[:, :2]

        # sklearn DBSCAN - use leaf_size for faster tree queries
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster_points,
                    algorithm='ball_tree', leaf_size=50)
        labels = db.fit_predict(points_xy)

        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            mask = labels == label
            cluster = points[mask]
            if len(cluster) >= self.min_cluster_points:
                clusters.append(cluster)

        return self._merge_overlapping_clusters(clusters)

    def _merge_overlapping_clusters(self, clusters: List[np.ndarray]) -> List[np.ndarray]:
        """Merge clusters that are too close."""
        if len(clusters) <= 1:
            return clusters

        merge_dist = self.cluster_eps * 2.5
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
                multi_belief = self.beliefs[bid]
                # Use active belief for association
                belief = multi_belief.get_active_belief()
                bx, by = belief.position

                center_dist = np.sqrt((centroid[0] - bx)**2 + (centroid[1] - by)**2)
                if center_dist > self.max_association_distance:
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
        """Update all hypotheses for matched beliefs."""
        for cluster_idx, belief_id in associations:
            cluster_robot = clusters_robot[cluster_idx]
            cluster_room = clusters_room[cluster_idx]
            multi_belief = self.beliefs[belief_id]

            pts_robot = torch.tensor(cluster_robot, dtype=DTYPE, device=self.device)
            pts_room = torch.tensor(cluster_room, dtype=DTYPE, device=self.device)

            # Update each hypothesis
            for model_type, hypothesis in multi_belief.hypotheses.items():
                belief = hypothesis.belief

                # Transform belief to robot frame for prediction
                belief_mu_robot, belief_Sigma_robot = transform_box_with_covariance(
                    belief.mu, belief.Sigma, self.robot_pose, self.robot_cov)

                # EARLY EXIT: Check if prediction is good enough
                sdf_func = get_sdf_function(model_type)
                predicted_sdf = sdf_func(pts_robot, belief_mu_robot)
                predicted_sdf_mean = torch.mean(torch.abs(predicted_sdf)).item()

                if predicted_sdf_mean < self.early_exit_sdf_threshold:
                    # Prediction is good, skip optimization
                    self._early_exits += 1
                    vfe = predicted_sdf_mean
                    belief.last_sdf_mean = vfe
                    self.selector.update_hypothesis_vfe(multi_belief, model_type, vfe)
                    belief.update_lifecycle(self.frame_count, was_observed=True)
                    # Still add historical points for good predictions
                    current_sdf = belief.sdf(pts_room)
                    belief.add_historical_points(pts_room, current_sdf, self.robot_cov)
                    continue

                self._full_optimizations += 1

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

                # Optimize this hypothesis
                opt_mu, vfe = self._optimize_hypothesis(
                    all_pts, all_weights, belief_mu_robot, belief,
                    model_type)

                # Transform back to room frame and update
                belief.mu = transform_object_to_room_frame(opt_mu, self.robot_pose)
                belief.last_sdf_mean = vfe

                # Update VFE for model selection
                self.selector.update_hypothesis_vfe(multi_belief, model_type, vfe)

                # Add historical points
                current_sdf = belief.sdf(pts_room)
                belief.add_historical_points(pts_room, current_sdf, self.robot_cov)
                belief.update_rfe(self.robot_cov)
                belief.update_lifecycle(self.frame_count, was_observed=True)

            # Update model posterior q(m)
            multi_belief.update_posterior()

            # Debug: Show VFE for each model every N frames
            if self.frame_count % 50 == 0:
                vfe_info = ", ".join([f"{m}={h.current_vfe:.4f}" for m, h in multi_belief.hypotheses.items()])
                console.print(f"[dim][VFE Debug] Belief {belief_id}: {vfe_info}[/dim]")

    def _optimize_hypothesis(self, points: torch.Tensor, weights: Optional[torch.Tensor],
                             mu_robot: torch.Tensor, belief: Belief,
                             model_type: str) -> Tuple[torch.Tensor, float]:
        """Optimize a single hypothesis and return (optimized_mu, VFE)."""
        if len(points) < 3:
            return mu_robot, float('inf')

        if weights is None:
            weights = torch.ones(len(points), dtype=DTYPE, device=points.device)

        mu = mu_robot.clone().detach().requires_grad_(True)
        final_vfe = 0.0

        sdf_func = get_sdf_function(model_type)

        for _ in range(self.optimization_iters):
            sdf = sdf_func(points, mu)
            sdf_mean = torch.mean(torch.abs(sdf)).item()

            # Likelihood term
            weighted_likelihood = torch.sum(weights * sdf**2) / len(points)

            # Prior term
            prior_term = belief.compute_prior_term(mu, self.robot_pose)

            # Total VFE
            loss = weighted_likelihood + prior_term
            final_vfe = loss.item()

            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            with torch.no_grad():
                if mu.grad is not None and not torch.isnan(mu.grad).any():
                    grad = torch.clamp(mu.grad, -self.grad_clip, self.grad_clip)

                    # Boost angle gradient for chairs
                    if model_type == 'chair':
                        grad[-1] = torch.clamp(grad[-1] * 100.0, -0.5, 0.5)

                    mu -= self.optimization_lr * grad

                    # Clamp sizes
                    config = belief.config
                    state_dim = len(mu)
                    for i in range(2, state_dim - 1):
                        mu[i] = torch.clamp(mu[i], config.min_size, config.max_size)

                    # Normalize angle
                    mu[-1] = torch.atan2(torch.sin(mu[-1]), torch.cos(mu[-1]))

                    # Apply model-specific constraints (e.g., table leg_length <= table_height)
                    mu = belief.apply_constraints(mu)

            mu.requires_grad_(True)

        return mu.detach(), final_vfe

    def _create_new_beliefs(self, unmatched_clusters: set, clusters: List[np.ndarray]):
        """Create new multi-model beliefs for unmatched clusters."""
        for idx in unmatched_clusters:
            cluster = clusters[idx]
            multi_belief = self.selector.create_multi_belief(cluster)

            if multi_belief is not None:
                self.beliefs[multi_belief.id] = multi_belief
                console.print(f"[green]New multi-model belief: id={multi_belief.id}, "
                            f"models={list(multi_belief.hypotheses.keys())}")

    def _decay_unmatched_beliefs(self, unmatched_ids: set):
        """Decay unmatched beliefs and remove if below threshold."""
        to_remove = []

        for bid in list(self.beliefs.keys()):
            if bid in unmatched_ids:
                multi_belief = self.beliefs[bid]
                # Decay all hypotheses
                for hyp in multi_belief.hypotheses.values():
                    hyp.belief.update_lifecycle(self.frame_count, was_observed=False)

                # Check if active belief should be removed
                active = multi_belief.get_active_belief()
                if active.should_remove():
                    to_remove.append(bid)

        for bid in to_remove:
            del self.beliefs[bid]

    def _process_commitments(self):
        """Check and process model commitments for all beliefs."""
        for multi_belief in self.beliefs.values():
            if multi_belief.state == ModelState.UNCERTAIN:
                self.selector.process_commitment(multi_belief)

    def _debug_print(self):
        """Print debug information about all beliefs."""
        print(f"\n{'='*70}")
        print(f"MULTI-MODEL MANAGER - Frame {self.frame_count}")
        print(f"{'='*70}")
        print(f"Active beliefs: {len(self.beliefs)}")

        for bid, mb in self.beliefs.items():
            state_str = mb.state.value
            if mb.state == ModelState.COMMITTED:
                print(f"  Belief {bid}: COMMITTED to {mb.committed_model}")
            else:
                q_str = ", ".join(f"{m}={p:.2f}" for m, p in mb.q_m.items())
                entropy = mb.compute_entropy()
                print(f"  Belief {bid}: UNCERTAIN q(m)=[{q_str}], H={entropy:.3f}")

        print(f"{'='*70}\n")

    def get_beliefs_as_dicts(self) -> List[dict]:
        """Get beliefs as dictionaries for visualization."""
        return [mb.to_dict() for mb in self.beliefs.values()]

    def get_historical_points_for_viz(self) -> Dict[int, np.ndarray]:
        """Get historical points per belief for visualization."""
        result = {}
        for bid, multi_belief in self.beliefs.items():
            active_belief = multi_belief.get_active_belief()
            if active_belief.num_historical_points > 0:
                result[bid] = active_belief.historical_points.cpu().numpy()
        return result

    def get_viz_data(self) -> dict:
        """Get visualization data."""
        return self.viz_data
