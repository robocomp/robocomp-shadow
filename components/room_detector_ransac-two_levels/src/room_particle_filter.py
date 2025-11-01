import numpy as np
from dataclasses import dataclass, field
import open3d as o3d
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Gradient-based refinement disabled.")


# We need o3d here to get OBBs
# Make sure to import 'plane_detector' if you pass the class,
# but it's cleaner to pass the object, so we don't need the import.

@dataclass
class RoomHypothesis:
    """
    Data class holding the current top-down room model.
    This object is the "constraint" that gets passed back down.
    """
    confidence: float = 0.0
    # Stores the (A,B,C,D) model of the *best* plane for each wall
    wall_models: dict = field(default_factory=dict)
    # Stores the distance from origin (e.g., x_min, x_max)
    extents: dict = field(default_factory=dict)
    # Store the dynamic axes of the room model
    axis_1: np.ndarray = None  # e.g., the room's "X" axis
    axis_2: np.ndarray = None  # e.g., the room's "Y" axis


@dataclass
class Particle:
    """A single room hypothesis."""
    x: float
    y: float
    theta: float
    length: float
    width: float
    weight: float = 1.0
    height: float = 2.5  # Fixed room height
    z_center: float = 1.25  # Vertical center of room (height/2)

    def get_synthetic_walls(self):
        """
        Calculates the 4 plane models (A,B,C,D) for this particle's
        room box AND their expected extents.
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        center = np.array([self.x, self.y])

        axis_1 = R @ np.array([1, 0])  # Room's local X-axis
        axis_2 = R @ np.array([0, 1])  # Room's local Y-axis

        p_pos_x = center + axis_1 * (self.length / 2.0)
        p_neg_x = center - axis_1 * (self.length / 2.0)
        p_pos_y = center + axis_2 * (self.width / 2.0)
        p_neg_y = center - axis_2 * (self.width / 2.0)

        # The walls along axis_1 have an extent (length) of self.width
        # The walls along axis_2 have an extent (length) of self.length
        walls = {
            'ax1_pos': {
                'model': np.array([axis_1[0], axis_1[1], 0.0, -np.dot(axis_1, p_pos_x)]),
                'extent': self.width
            },
            'ax1_neg': {
                'model': np.array([-axis_1[0], -axis_1[1], 0.0, -np.dot(-axis_1, p_neg_x)]),
                'extent': self.width
            },
            'ax2_pos': {
                'model': np.array([axis_2[0], axis_2[1], 0.0, -np.dot(axis_2, p_pos_y)]),
                'extent': self.length
            },
            'ax2_neg': {
                'model': np.array([-axis_2[0], -axis_2[1], 0.0, -np.dot(-axis_2, p_neg_y)]),
                'extent': self.length
            }
        }
        return walls


class RoomParticleFilter:
    def __init__(self, num_particles, initial_hypothesis=None, elite_count=1, use_gradient_refinement=True):
        self.num_particles = num_particles
        self.particles = []

        self.motion_magnitude_threshold = 0.08  # Threshold to consider "stationary" over modulus of differential motion
        # --- Motion/Process Noise ---
        self.pos_noise = 0.05  # Increased for better exploration
        self.rot_noise = np.deg2rad(1.0)  # Increased
        self.size_noise = 0.10
        self.z_noise = 0.02  # Vertical position noise

        # --- Minimum noise when stationary (for numerical stability) ---
        self.min_pos_noise = 0.00  # Increased from 0.001
        self.min_rot_noise = np.deg2rad(0.0)  # Increased
        self.min_size_noise = 0.000
        self.min_z_noise = 0.001

        # --- Scoring Parameters (These need tuning!) ---
        self.w_angle = 1.0
        self.w_dist = 0.5
        self.w_extent = 0.04
        self.w_height = 0.5  # Weight for vertical position matching
        self.match_threshold = 0.5
        self.no_match_penalty = 2.0
        self.score_sharpness_k = 3.0  # Increased from 3.0 for faster convergence

        # --- Stability Improvements ---
        # Exponential moving average for smoothing
        self.best_estimate = None
        self.ema_alpha = 0.3  # Smoothing factor (0=no update, 1=full update)

        # Adaptive resampling
        self.resample_threshold = 0.5  # Resample when Neff < threshold * N
        
        # Elite strategy - keep best N particles across generations
        self.elite_count = min(elite_count, num_particles // 10)  # Cap at 10% of population
        
        # Gradient-based refinement
        self.use_gradient_refinement = use_gradient_refinement and TORCH_AVAILABLE
        if use_gradient_refinement and not TORCH_AVAILABLE:
            print("Warning: Gradient refinement requested but PyTorch not available")

        # Track previous best for temporal consistency
        self.previous_best = None

        self._initialize_particles(initial_hypothesis)

    def step(self, odometry_delta, plane_detector, pcd):
        """
        Runs one full cycle of the filter:
        Predict, Detect (biased), Update (score), Resample.

        Args:
            odometry_delta (tuple): (dx, dy, dtheta) from robot's motion.
            plane_detector (PlaneDetector): The detector object.
            pcd (o3d.geometry.PointCloud): The full, downsampled point cloud.

        Returns:
            tuple: (best_particle, (h_planes, v_planes, o_planes, outliers))
                   The new best particle for visualization,
                   AND the detected plane data for visualization.
        """
        # 1. PREDICT STEP
        self.predict(odometry_delta)

        # 2. GET BEST HYPOTHESIS (Top-Down belief)
        best_particle_for_bias = self.get_best_particle()
        hypothesis_for_detection = None

        if best_particle_for_bias:
            # Convert the particle to a RoomHypothesis for the detector
            synthetic_walls = best_particle_for_bias.get_synthetic_walls()
            wall_models = {key: val['model'] for key, val in synthetic_walls.items()}
            hypothesis_for_detection = RoomHypothesis(
                confidence=best_particle_for_bias.weight * self.num_particles,
                wall_models=wall_models,
                axis_1=np.array([np.cos(best_particle_for_bias.theta), np.sin(best_particle_for_bias.theta), 0]),
                axis_2=np.array([-np.sin(best_particle_for_bias.theta), np.cos(best_particle_for_bias.theta), 0])
            )

        # 3. DETECT STEP (Bottom-Up, biased by hypothesis)
        (h_planes, v_planes, o_planes, outliers) = plane_detector.detect(pcd)

        # 4. PREPARE DATA FOR UPDATE
        measured_walls_data = []
        for model, indices in v_planes:
            inlier_pcd = pcd.select_by_index(indices)
            # Need at least 3 points to make an OBB
            if len(inlier_pcd.points) < 3:
                continue
            obb = inlier_pcd.get_oriented_bounding_box()
            measured_walls_data.append((model, obb))

        # 5. UPDATE STEP (Scoring)
        self.update(measured_walls_data)

        # 6. RESAMPLE STEP
        self.resample()
        
        # 7. GRADIENT REFINEMENT (Optional but recommended)
        # Refine top particles using gradient descent on point-to-wall distance
        if self.use_gradient_refinement:
            self.refine_best_particles_gradient(pcd, top_n=3, num_steps=40, lr=0.01)

        # 8. RETURN RESULTS
        # Return the best particle from the *newly* resampled set
        # and the detected planes from step 3
        return self.get_best_particle(), (h_planes, v_planes, o_planes, outliers)

    def _initialize_particles(self, initial_guess: Particle = None):  # <-- MODIFIED
        """
        Creates the particle cloud. If an initial_guess is provided,
        it samples around that guess. Otherwise, it samples randomly.
        """
        self.particles = []

        if initial_guess:
            print(f"Initializing {self.num_particles} particles around ground truth guess...")
            # First particle is the *exact* guess
            initial_guess.weight = 1.0 / self.num_particles
            self.particles.append(initial_guess)

            # Create the rest by adding noise
            for _ in range(self.num_particles - 1):
                p = Particle(
                    x=initial_guess.x + np.random.normal(0.0, self.pos_noise),
                    y=initial_guess.y + np.random.normal(0.0, self.pos_noise),
                    theta=initial_guess.theta + np.random.normal(0.0, self.rot_noise),
                    length=initial_guess.length + np.random.normal(0.0, self.size_noise),
                    width=initial_guess.width + np.random.normal(0.0, self.size_noise),
                    z_center=initial_guess.z_center + np.random.normal(0.0, self.z_noise),
                    weight=1.0 / self.num_particles
                )
                p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi  # Normalize
                p.z_center = max(0.5, p.z_center)  # Keep above ground
                self.particles.append(p)
        else:
            print("Initializing random particles (no initial guess provided)")
            for _ in range(self.num_particles):
                p = Particle(
                    x=np.random.uniform(-5, 5),
                    y=np.random.uniform(-5, 5),
                    theta=np.random.uniform(-np.pi, np.pi),
                    length=np.random.uniform(3, 10),
                    width=np.random.uniform(3, 10),
                    z_center=np.random.uniform(0.5, 2.0),  # Random height
                    weight=1.0 / self.num_particles
                )
                self.particles.append(p)

        print("...Initialization complete.")

    def predict(self, odometry_delta):
        dx, dy, dtheta = odometry_delta

        # Adaptive noise: Use minimal noise if stationary, normal noise if moving
        motion_magnitude = np.sqrt(dx ** 2 + dy ** 2) + abs(dtheta)
        is_stationary = motion_magnitude < self.motion_magnitude_threshold  # Threshold for "stationary"

        if is_stationary:
            pos_noise = self.min_pos_noise
            rot_noise = self.min_rot_noise
        else:
            pos_noise = self.pos_noise
            rot_noise = self.rot_noise

        for p in self.particles:
            # Update pose (with noise for tracking)
            p.x += dx + np.random.normal(0.0, pos_noise)
            p.y += dy + np.random.normal(0.0, pos_noise)
            p.theta += dtheta + np.random.normal(0.0, rot_noise)
            
            # Shape parameters: NO noise - refined by gradient optimization
            # p.length, p.width, p.z_center remain unchanged
            
            # Normalize theta
            p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi

    def _compute_wall_match_error(self, synth_wall, measured_walls_data, particle_z_center):
        """Compute error between one synthetic wall and all measured walls, return best match and its index"""
        synth_model = synth_wall['model']
        synth_extent = synth_wall['extent']
        min_error = float('inf')
        best_match_idx = -1

        for idx, (measured_model, measured_obb) in enumerate(measured_walls_data):
            # Normalize normals for comparison
            n1 = synth_model[:3]
            n2 = measured_model[:3]
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)

            # Normal error: how parallel are the planes?
            normal_error = 1.0 - np.abs(np.dot(n1, n2))

            # Distance error: how far apart are the planes?
            dist_error = np.abs(np.abs(synth_model[3]) - np.abs(measured_model[3]))

            # Extent error: how different are their lengths?
            exts = np.sort(measured_obb.extent)
            measured_extent = exts[-2]
            extent_error = np.abs(synth_extent - measured_extent)

            # Height error: how different are their vertical centers?
            measured_z_center = measured_obb.center[2]
            height_error = np.abs(particle_z_center - measured_z_center)

            hybrid_error = (self.w_angle * normal_error) + \
                           (self.w_dist * dist_error) + \
                           (self.w_extent * extent_error) + \
                           (self.w_height * height_error)

            if hybrid_error < min_error:
                min_error = hybrid_error
                best_match_idx = idx

        if min_error > self.match_threshold:
            return self.no_match_penalty, -1

        return min_error, best_match_idx

    def update(self, measured_walls_data):
        """Update particle weights using bidirectional matching"""
        if not measured_walls_data:
            return

        total_weight = 0
        for p in self.particles:
            synthetic_walls = p.get_synthetic_walls()
            
            # Forward matching: synthetic -> measured
            total_error = 0
            matched_measured_indices = set()
            
            for key, synth_wall in synthetic_walls.items():
                error, match_idx = self._compute_wall_match_error(synth_wall, measured_walls_data, p.z_center)
                total_error += error
                if match_idx >= 0:
                    matched_measured_indices.add(match_idx)
            
            # Penalty for unmatched measured walls (reverse check)
            unmatched_measured = len(measured_walls_data) - len(matched_measured_indices)
            if unmatched_measured > 0:
                total_error += unmatched_measured * self.no_match_penalty * 0.5
            
            # Position consistency penalty: check if particle center is reasonable given wall positions
            if len(matched_measured_indices) > 0:
                # Compute average center of matched measured walls
                wall_centers = []
                for idx in matched_measured_indices:
                    _, obb = measured_walls_data[idx]
                    wall_centers.append(obb.center[:2])  # XY only
                avg_wall_center = np.mean(wall_centers, axis=0)
                particle_center = np.array([p.x, p.y])
                
                # Particle should be near the center of its walls
                center_offset = np.linalg.norm(particle_center - avg_wall_center)
                # Only penalize if significantly off (allow some tolerance)
                if center_offset > 1.0:  # More than 1 meter offset
                    total_error += (center_offset - 1.0) * 0.5

            p.weight = np.exp(-self.score_sharpness_k * total_error)
            total_weight += p.weight

        if total_weight > 1e-9:
            for p in self.particles:
                p.weight /= total_weight
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

    def resample(self):
        """
        Adaptive resampling with elitism: only resample when effective sample size is low.
        Elite particles (best N) are always preserved across generations.
        """
        weights = np.array([p.weight for p in self.particles])

        # Calculate effective sample size
        weights_squared_sum = np.sum(weights ** 2)
        if weights_squared_sum > 0:
            neff = 1.0 / weights_squared_sum
        else:
            neff = 0

        # Only resample if effective sample size is below threshold
        if neff < self.resample_threshold * self.num_particles:
            new_particles = []
            if np.sum(weights) < 1e-9:
                weights = np.ones(self.num_particles) / self.num_particles

            # Elite strategy: preserve best particles
            if self.elite_count > 0:
                # Sort particles by weight (descending)
                sorted_indices = np.argsort(weights)[::-1]
                
                # Keep elite particles
                for i in range(self.elite_count):
                    elite_idx = sorted_indices[i]
                    p_old = self.particles[elite_idx]
                    p_elite = Particle(
                        x=p_old.x, y=p_old.y, theta=p_old.theta,
                        length=p_old.length, width=p_old.width,
                        z_center=p_old.z_center,
                        weight=1.0 / self.num_particles
                    )
                    new_particles.append(p_elite)
                
                # Sample remaining particles
                remaining_count = self.num_particles - self.elite_count
                indices = np.random.choice(
                    self.num_particles, remaining_count, p=weights, replace=True
                )
            else:
                # No elitism - standard resampling
                indices = np.random.choice(
                    self.num_particles, self.num_particles, p=weights, replace=True
                )

            for i in indices:
                p_old = self.particles[i]
                p_new = Particle(
                    x=p_old.x, y=p_old.y, theta=p_old.theta,
                    length=p_old.length, width=p_old.width,
                    z_center=p_old.z_center,
                    weight=1.0 / self.num_particles
                )
                new_particles.append(p_new)
            self.particles = new_particles
        else:
            # Don't resample, just normalize weights
            total_weight = np.sum(weights)
            if total_weight > 0:
                for p in self.particles:
                    p.weight /= total_weight

    def _compute_differentiable_loss(self, x, y, theta, length, width, points_tensor):
        """
        Compute differentiable loss using Signed Distance Function (SDF) for a box.
        SDF is smoother and faster than per-wall distance computation.
        
        Args:
            x, y, theta, length, width: Room parameters (torch tensors)
            points_tensor: Nx3 tensor of point cloud points
        
        Returns:
            loss: Scalar tensor
        """
        # Rotation matrix (inverse to transform points to room frame)
        c = torch.cos(theta)
        s = torch.sin(theta)
        R_inv = torch.stack([
            torch.stack([c, s]),
            torch.stack([-s, c])
        ])
        
        # Transform points to room-centric frame
        center = torch.stack([x, y])
        points_xy = points_tensor[:, :2]  # Nx2
        local_points = torch.matmul(points_xy - center, R_inv.T)  # Nx2
        
        # Half extents of the box
        half_extents = torch.stack([length / 2.0, width / 2.0])
        
        # SDF for axis-aligned box in local frame
        # SDF(p) = ||max(|p| - half_extents, 0)|| + min(max(|p| - half_extents), 0)
        q = torch.abs(local_points) - half_extents  # Nx2
        
        # Outside component: ||max(q, 0)||
        q_positive = torch.maximum(q, torch.zeros_like(q))
        outside_dist = torch.norm(q_positive, dim=1)  # N
        
        # Inside component: -min(max(q_x, q_y), 0) since inside points have negative distances
        inside_dist = -torch.minimum(torch.max(q, dim=1)[0], torch.zeros(q.shape[0], device=q.device))
        
        # Combined SDF: negative inside, positive outside
        sdf = outside_dist + inside_dist  # N
        
        # Loss: penalize points outside or too close to walls
        margin = -0.02  # 20cm safety margin inside
        violations = torch.relu(sdf - margin)  # Penalize if SDF > -margin
        total_loss = torch.sum(violations ** 2)
        
        # Regularization: prefer sizes close to initial guess (prevents collapse or explosion)
        size_loss = 0.0001 * ((length - 5.0) ** 2 + (width - 10.0) ** 2)
        # size_lambda = 1e-4
        # if length is not None and width is not None:
        #     size_loss = size_lambda * ((length - p_length0) ** 2 + (width - p_width0) ** 2)
        # else:
        #     size_loss = 0.0
        #
        return total_loss + size_loss

    def refine_best_particles_gradient(self, pcd, top_n=3, num_steps=20, lr=0.01):
        """
        Refine the top N particles using gradient descent on a differentiable loss.
        
        Args:
            pcd: Open3D point cloud
            top_n: Number of top particles to refine
            num_steps: Number of gradient descent steps
            lr: Learning rate
        """
        if not TORCH_AVAILABLE:
            return
        
        if len(self.particles) == 0:
            return
        
        # Get top N particles by weight
        weights = np.array([p.weight for p in self.particles])
        top_indices = np.argsort(weights)[-top_n:]
        
        # Convert point cloud to torch tensor
        points_np = np.asarray(pcd.points)
        if len(points_np) == 0:
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        points_tensor = torch.from_numpy(points_np).float().to(device)
        
        # Refine each top particle
        for idx in top_indices:
            p = self.particles[idx]
            
            # Initialize parameters as torch tensors with gradients
            x = torch.tensor(p.x, requires_grad=True, dtype=torch.float32, device=device)
            y = torch.tensor(p.y, requires_grad=True, dtype=torch.float32, device=device)
            theta = torch.tensor(p.theta, requires_grad=True, dtype=torch.float32, device=device)
            length = torch.tensor(p.length, requires_grad=True, dtype=torch.float32, device=device)
            width = torch.tensor(p.width, requires_grad=True, dtype=torch.float32, device=device)
            
            # Use Adam optimizer
            optimizer = torch.optim.Adam([x, y, theta, length, width], lr=lr)

            # keep a gentle prior to avoid big pose jumps
            pose_lambda = 1e-3  # tune: 1e-4 .. 1e-2
            theta_lambda = 1e-3
            x0 = torch.tensor(p.x, dtype=torch.float32, device=device)
            y0 = torch.tensor(p.y, dtype=torch.float32, device=device)
            theta0 = torch.tensor(p.theta, dtype=torch.float32, device=device)

            # Gradient descent
            for step in range(num_steps):
                optimizer.zero_grad()
                loss = self._compute_differentiable_loss(x, y, theta, length, width, points_tensor)
                # angle wrap for the prior
                dth = ((theta - theta0 + np.pi) % (2 * np.pi)) - np.pi

                data_loss = loss \
                       + pose_lambda * ((x - x0) ** 2 + (y - y0) ** 2) \
                       + theta_lambda * (dth ** 2)
                data_loss.backward()
                optimizer.step()
                
                # Keep theta in valid range
                with torch.no_grad():
                    theta.data = (theta.data + np.pi) % (2 * np.pi) - np.pi
                    length.data = torch.clamp(length.data, min=1.0)
                    width.data = torch.clamp(width.data, min=1.0)
            
            # Update particle with refined values
            with torch.no_grad():
                p.x = x.item()
                p.y = y.item()
                p.theta = theta.item()
                p.length = length.item()
                p.width = width.item()

    def get_best_particle(self) -> Particle:
        """
        Gets the particle with the highest weight, smoothed over time
        using exponential moving average for stability.
        Returns None if confidence is too low.
        """
        # Find current best particle
        best_p = max(self.particles, key=lambda p: p.weight)
        scaled_weight = best_p.weight * self.num_particles

        if scaled_weight < 0.1:  # Confidence threshold
            return None

        # Apply exponential moving average for smoothing
        if self.best_estimate is None:
            # First frame: initialize with current best
            self.best_estimate = Particle(
                x=best_p.x,
                y=best_p.y,
                theta=best_p.theta,
                length=best_p.length,
                width=best_p.width,
                z_center=best_p.z_center,
                weight=best_p.weight
            )
        else:
            # Smooth the estimate over time
            alpha = self.ema_alpha

            # Handle angle wrapping for theta
            theta_diff = best_p.theta - self.best_estimate.theta
            theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi

            self.best_estimate.x = (1 - alpha) * self.best_estimate.x + alpha * best_p.x
            self.best_estimate.y = (1 - alpha) * self.best_estimate.y + alpha * best_p.y
            self.best_estimate.theta = self.best_estimate.theta + alpha * theta_diff
            self.best_estimate.length = (1 - alpha) * self.best_estimate.length + alpha * best_p.length
            self.best_estimate.width = (1 - alpha) * self.best_estimate.width + alpha * best_p.width
            self.best_estimate.z_center = (1 - alpha) * self.best_estimate.z_center + alpha * best_p.z_center
            self.best_estimate.weight = best_p.weight

            # Normalize theta
            self.best_estimate.theta = (self.best_estimate.theta + np.pi) % (2 * np.pi) - np.pi

        return self.best_estimate