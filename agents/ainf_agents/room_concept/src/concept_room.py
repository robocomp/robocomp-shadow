import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# Use float32 for speed, GPU if available
DTYPE = torch.float32
GPU_AVAILABLE = False
DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    try:
        # Test if GPU actually works (RTX 50xx needs PyTorch nightly with sm_120)
        _test = torch.zeros(1, device='cuda') + 1
        DEVICE = torch.device('cuda')
        GPU_AVAILABLE = True
        print(f"[RoomEstimator] Using GPU: {torch.cuda.get_device_name(0)}")
    except RuntimeError:
        print("[RoomEstimator] GPU not compatible with current PyTorch, using CPU")


@dataclass
class RoomBelief:
    """
    Belief state for room geometry and robot pose.

    Coordinate convention: x+ = right, y+ = forward, θ = heading angle (θ=0 → +y)

    State vector: s = (x, y, θ, W, L)
    - (x, y, θ): robot pose in room frame (in METERS)
      * x: lateral position (right is positive)
      * y: forward position (front is positive)
      * θ: heading angle (radians), θ=0 means facing +y (forward)
    - (W, L): room width and length (in METERS)
      * W: width in x direction (lateral extent)
      * L: length in y direction (forward extent)

    Belief is Gaussian: q(s) = N(μ, Σ)
    """
    # Mean (in meters)
    x: float = 0.0  # lateral position (right is positive)
    y: float = 0.0  # forward position
    theta: float = 0.0  # heading angle (θ=0 → facing +y)
    width: float = 6.0   # room width (x direction) - 6m initial guess
    length: float = 4.0  # room length (y direction) - 4m initial guess

    # Covariance (5x5 matrix)
    cov: np.ndarray = None

    # Convergence flag
    converged: bool = False

    def __post_init__(self):
        if self.cov is None:
            # High initial uncertainty reflecting Gaussian prior on room dimensions
            # Pose uncertainty: [x, y, theta] - in meters and radians
            # Room uncertainty: [width, length] with std ~0.5m (50cm)
            self.cov = np.diag([2.0, 2.0, 1.0, 0.25, 0.25])  # 0.5^2 = 0.25

    @property
    def pose(self) -> np.ndarray:
        """Robot pose [x, y, θ]"""
        return np.array([self.x, self.y, self.theta])

    @property
    def pose_cov(self) -> np.ndarray:
        """Pose covariance [3x3]"""
        return self.cov[:3, :3].copy()

    @property
    def room_params(self) -> np.ndarray:
        """Room parameters [W, L]"""
        return np.array([self.width, self.length])

    @property
    def state_vector(self) -> np.ndarray:
        """Full state [x, y, θ, W, L]"""
        return np.array([self.x, self.y, self.theta, self.width, self.length])

    def uncertainty(self) -> float:
        """Total uncertainty (trace of pose covariance)"""
        return np.trace(self.cov[:3, :3])


class RoomPoseEstimatorV2:
    """
    Room and Pose Estimation using SDF minimization with PyTorch.

    Two-phase operation:

    Phase 1 (Initialization):
        - Robot is STATIC
        - Collect upper LIDAR points
        - Minimize SDF error wrt full state (x, y, θ, W, L)
        - Use PyTorch autograd for gradients
        - Once error < threshold, fix room parameters

    Phase 2 (Tracking):
        - Room parameters (W, L) are FIXED
        - Each step: predict pose using velocity, then correct with SDF
        - Minimize: F = SDF_likelihood + KL_prior
        - Output pose and covariance for obstacle models
    """

    def __init__(self,
                 room_width: float = 6.0,
                 room_height: float = 4.0,
                 distance_noise_std: float = 0.005,
                 use_known_room_dimensions: bool = False):
        """
        Args:
            room_width: True room width in METERS (for reporting/comparison only)
            room_height: True room height in METERS (for reporting/comparison only)
            distance_noise_std: Gaussian noise std dev on distance measurements in METERS
            use_known_room_dimensions: If True, fix room dimensions (only estimate pose)
        """
        # Ground truth (for reporting only) - in METERS
        self.true_width = room_width
        self.true_height = room_height
        self.use_known_room_dimensions = use_known_room_dimensions
        self.distance_noise_std = distance_noise_std

        # Belief state
        self.belief: Optional[RoomBelief] = None

        # Phase: 'init' or 'tracking'
        self.phase = 'init'

        # Initialization parameters
        self.init_buffer = []  # Collected LIDAR points
        self.min_points_for_init = 500  # Need more points for robust estimation
        self.init_convergence_threshold = 0.15  # 15cm threshold - more permissive for static detection
        self.max_init_iterations = 100  # Reduced for faster convergence with static robot
        self.init_iterations = 0

        # Tracking parameters - adjusted for meters
        # sigma_sdf represents the effective uncertainty in wall distance measurements
        # This includes sensor noise + dynamic effects (motion, rotation, timing)
        # Needs to be large enough that prior (motion model) remains influential
        self.sigma_sdf = 0.15  # 15cm - accounts for dynamic uncertainty during motion
        self.Q_pose = np.diag([0.01, 0.01, 0.05])  # Process noise in meters

        # Velocity-adaptive precision weighting parameters
        # These weights adjust which parameters (x, y, theta) get more optimization effort
        # based on the current velocity profile
        self.velocity_adaptive_weights = True  # Enable velocity-based weighting
        self.linear_velocity_threshold = 0.05   # m/s - below this = "not moving linearly"
        self.angular_velocity_threshold = 0.05  # rad/s - below this = "not rotating"
        self.weight_base = 1.0                  # Base weight for all parameters
        self.weight_boost_factor = 2.0          # Multiplier for emphasized parameters
        self.weight_reduction_factor = 0.5     # Multiplier for de-emphasized parameters
        self._current_velocity_weights = np.array([1.0, 1.0, 1.0])  # [x, y, theta] weights

        # Optimization parameters
        self.lr_init = 0.5  # Higher learning rate for faster convergence
        self.lr_tracking = 0.1  # Balanced for convergence with theta constraint

        # Dynamic CPU optimization parameters
        self.adaptive_cpu = True  # Enable adaptive CPU optimization
        self.min_iterations = 5   # Minimum iterations in optimization loop
        self.max_iterations_init = 100  # Max iterations for init phase
        self.max_iterations_tracking = 50  # Max iterations for tracking phase
        self.early_stop_threshold = 1e-5  # Stop if loss improvement < this
        self.skip_frames_when_stable = True  # Skip processing when stable
        self.stability_threshold = 0.02  # SDF error below this = stable (2cm)
        self.max_skip_frames = 3  # Maximum frames to skip when stable
        self._skip_counter = 0   # Current skip counter
        self._last_sdf_error = float('inf')  # Last SDF error for stability check
        self.lidar_subsample_factor = 1  # Subsample LIDAR points (1 = no subsampling)
        self.adaptive_subsampling = False  # DISABLED - subsampling now done at LIDAR proxy level
        self.min_lidar_points = 100  # Minimum LIDAR points to keep

        # Statistics
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': [],
            'iterations_used': [],  # Track actual iterations per step
            'frames_skipped': 0     # Track skipped frames
        }

    def _subsample_lidar(self, points: torch.Tensor) -> torch.Tensor:
        """Adaptively subsample LIDAR points based on stability.

        When the system is stable (low SDF error), we can use fewer points
        to reduce CPU usage without significantly affecting accuracy.
        """
        if not self.adaptive_subsampling or len(points) <= self.min_lidar_points:
            return points

        # Determine subsample factor based on last SDF error
        if self._last_sdf_error < self.stability_threshold:
            # Very stable - use fewer points
            factor = min(4, max(1, len(points) // self.min_lidar_points))
        elif self._last_sdf_error < self.stability_threshold * 2:
            # Moderately stable
            factor = min(2, max(1, len(points) // (self.min_lidar_points * 2)))
        else:
            # Not stable - use all points
            factor = 1

        if factor > 1:
            indices = torch.arange(0, len(points), factor, device=points.device)
            return points[indices]
        return points

    def _should_skip_frame(self, robot_velocity: np.ndarray, angular_velocity: float) -> bool:
        """Determine if we should skip processing this frame.

        Skip when:
        - Robot is not moving AND system is stable
        - We haven't skipped too many frames in a row
        """
        if not self.skip_frames_when_stable or not self.adaptive_cpu:
            return False

        # Check if robot is moving
        speed = np.linalg.norm(robot_velocity) if robot_velocity is not None else 0
        is_moving = speed > 0.01 or abs(angular_velocity) > 0.01  # 1cm/s or 0.01 rad/s

        # Check if system is stable
        is_stable = self._last_sdf_error < self.stability_threshold

        # Skip only if stable, not moving, and haven't skipped too many
        if is_stable and not is_moving and self._skip_counter < self.max_skip_frames:
            self._skip_counter += 1
            self.stats['frames_skipped'] += 1
            return True

        # Reset skip counter when we process a frame
        self._skip_counter = 0
        return False

    def _compute_velocity_adaptive_weights(self,
                                           robot_velocity: np.ndarray,
                                           angular_velocity: float) -> np.ndarray:
        """
        Compute velocity-adaptive precision weights for [x, y, theta].

        Based on the current velocity profile:
        - If rotating (high angular, low linear): boost theta weight, reduce x,y
        - If moving straight (high linear, low angular): boost x,y, reduce theta
        - If stationary: use base weights (uniform)

        The weights are used to scale the gradient/update for each parameter
        during optimization, making the system more responsive to parameters
        that are expected to change based on current motion.

        Args:
            robot_velocity: [vx, vy] velocity in robot frame (m/s)
            angular_velocity: Angular velocity (rad/s)

        Returns:
            np.ndarray: [w_x, w_y, w_theta] weight factors
        """
        if not self.velocity_adaptive_weights:
            return np.array([1.0, 1.0, 1.0])

        linear_speed = np.linalg.norm(robot_velocity) if robot_velocity is not None else 0.0
        angular_speed = abs(angular_velocity)

        # Determine motion profile
        is_rotating = angular_speed > self.angular_velocity_threshold
        is_translating = linear_speed > self.linear_velocity_threshold

        if is_rotating and not is_translating:
            # Pure rotation: emphasize theta, de-emphasize x, y
            w_x = self.weight_reduction_factor
            w_y = self.weight_reduction_factor
            w_theta = self.weight_boost_factor
        elif is_translating and not is_rotating:
            # Pure translation: emphasize x, y, de-emphasize theta
            # Also consider direction of motion (forward vs lateral)
            vx = robot_velocity[0] if robot_velocity is not None else 0
            vy = robot_velocity[1] if robot_velocity is not None else 0

            # Weight x and y based on which direction is dominant
            if abs(vy) > abs(vx):
                # Mostly forward/backward motion - emphasize y (forward axis)
                w_x = self.weight_base
                w_y = self.weight_boost_factor
            else:
                # Mostly lateral motion - emphasize x
                w_x = self.weight_boost_factor
                w_y = self.weight_base

            w_theta = self.weight_reduction_factor
        elif is_rotating and is_translating:
            # Combined motion: boost all moderately
            w_x = self.weight_base * 1.2
            w_y = self.weight_base * 1.2
            w_theta = self.weight_base * 1.2
        else:
            # Stationary: use base weights, allows gradual convergence
            w_x = self.weight_base
            w_y = self.weight_base
            w_theta = self.weight_base

        weights = np.array([w_x, w_y, w_theta])

        # Smooth transition using exponential moving average
        alpha = 0.3  # Smoothing factor
        self._current_velocity_weights = (1 - alpha) * self._current_velocity_weights + alpha * weights

        return self._current_velocity_weights.copy()


    def sdf_rect(self,
                 points: torch.Tensor,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 theta: torch.Tensor,
                 width: torch.Tensor,
                 length: torch.Tensor) -> torch.Tensor:
        """
        Signed Distance Function for rectangular room boundary.

        Coordinate convention: x=lateral (width), y=forward (length)
        Room origin at CENTER: walls at x=±width/2, y=±length/2

        For LIDAR points that hit walls, the SDF should be ≈ 0.
        This measures how far transformed points are from the room boundary.

        Args:
            points: [N, 2] points in robot frame [dx, dy]
            x, y: Robot position (x=lateral, y=forward) in meters, origin at room center
            theta: Robot heading angle in radians (theta=0 is +y/forward direction)
            width: Room width in x direction (lateral extent) in meters
            length: Room length in y direction (forward extent) in meters

        Returns:
            [N] absolute distance to nearest wall (should be ≈ 0)
        """
        if len(points) == 0:
            return torch.tensor([], dtype=DTYPE, device=DEVICE)

        # Rotation matrix (robot frame to room frame)
        # Convention: theta=0 means robot faces +y (forward)
        # Robot frame: x+ = right, y+ = forward
        # Room frame: same convention
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Transform points: p_room = R @ p_robot + [x, y]
        # Standard 2D rotation for theta=0 → +y convention
        px = points[:, 0]  # robot x (right)
        py = points[:, 1]  # robot y (forward)

        room_x = cos_t * px - sin_t * py + x
        room_y = sin_t * px + cos_t * py + y

        # Half dimensions (walls at ±half_width, ±half_length)
        half_width = width / 2.0
        half_length = length / 2.0

        # Distance to each wall (absolute, unsigned)
        d_left = torch.abs(room_x + half_width)   # Distance to x = -width/2
        d_right = torch.abs(room_x - half_width)  # Distance to x = +width/2
        d_bottom = torch.abs(room_y + half_length)  # Distance to y = -length/2
        d_top = torch.abs(room_y - half_length)     # Distance to y = +length/2

        # Each point should be ON exactly one wall, so min distance should be ≈ 0
        distances = torch.stack([d_left, d_right, d_bottom, d_top], dim=1)  # [N, 4]
        min_dist, _ = torch.min(distances, dim=1)  # [N]

        return min_dist


    def update(self,
               robot_pose_gt: np.ndarray = None,
               robot_velocity: np.ndarray = None,
               angular_velocity: float = 0.0,
               dt: float = 0.1,
               lidar_points: np.ndarray = None) -> Dict:
        """
        Update room and pose estimates using Active Inference.

        Args:
            robot_pose_gt: Ground truth pose [x, y, theta] - used ONLY for initial approximation
            robot_velocity: Velocity command [vx, vy] in m/s (vx=lateral, vy=forward)
            angular_velocity: Angular velocity command in rad/s
            dt: Time step in seconds
            lidar_points: Real LIDAR points [N, 2] from sensor in meters, robot frame.

        Returns:
            Dict with estimation results including phase, errors, belief state, etc.
        """
        if robot_velocity is None:
            robot_velocity = np.zeros(2)
        if robot_pose_gt is None:
            robot_pose_gt = np.zeros(3)

        # Get LIDAR points
        if lidar_points is not None and len(lidar_points) > 0:
            lidar_points_t = torch.tensor(lidar_points, dtype=DTYPE, device=DEVICE)
        else:
            return {
                'phase': self.phase,
                'status': 'no_lidar',
                'sdf_error': 0,
                'belief_uncertainty': self.belief.uncertainty() if self.belief else 5.0
            }

        # Check if we should skip this frame (only in tracking phase)
        if self.phase == 'tracking' and self._should_skip_frame(robot_velocity, angular_velocity):
            # Return last known result without processing
            return {
                'phase': 'tracking',
                'status': 'skipped',
                'pose': self.belief.pose.copy() if self.belief else np.zeros(3),
                'pose_cov': self.belief.pose_cov if self.belief else np.eye(3),
                'sdf_error': self._last_sdf_error,
                'belief_uncertainty': self.belief.uncertainty() if self.belief else 5.0,
                'innovation': np.zeros(3),
                'prior_precision': getattr(self, '_adaptive_prior_precision', 0.1)
            }

        # Apply adaptive LIDAR subsampling
        lidar_points_t = self._subsample_lidar(lidar_points_t)

        if self.phase == 'init':
            return self._initialization_phase(lidar_points_t, robot_pose_gt)
        else:
            return self._tracking_phase(lidar_points_t, robot_velocity, angular_velocity, dt)

    def _estimate_initial_state_from_lidar(self,
                                           all_points: torch.Tensor,
                                           robot_pose_gt: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Estimate room dimensions and robot orientation from LIDAR points with Gaussian priors.

        Uses Bayesian fusion:
        - Prior for room: N(μ_width=6m, σ²=1) and N(μ_length=4m, σ²=1)
        - Prior for orientation: N(μ_theta=0, σ²=1)
        - Likelihood: estimated from LIDAR point geometry

        The orientation is estimated by finding the principal axes of the LIDAR points
        using PCA and aligning them with the room axes.

        Convention: width is the larger dimension (x-axis), length is smaller (y-axis)

        Args:
            all_points: [N, 2] LIDAR points in robot frame
            robot_pose_gt: [x, y, theta] approximate robot pose (used for position only)

        Returns:
            (est_width, est_length, width_var, length_var, est_theta, theta_var, est_pos_x, est_pos_y)
        """
        points_np = all_points.cpu().numpy()

        # Compute raw spreads in robot frame
        raw_x_spread = np.max(points_np[:, 0]) - np.min(points_np[:, 0])
        raw_y_spread = np.max(points_np[:, 1]) - np.min(points_np[:, 1])

        # Use PCA to find principal axes
        centroid = np.mean(points_np, axis=0)
        centered = points_np - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Major axis angle from y-axis
        major_axis = eigenvectors[:, 0]
        pca_angle = np.arctan2(major_axis[0], major_axis[1])

        # Estimate theta using prior from GT
        prior_theta = robot_pose_gt[2] if robot_pose_gt is not None else 0.0
        prior_theta_var = 0.3 ** 2
        lidar_theta_var = 0.5 ** 2

        theta_var = (prior_theta_var * lidar_theta_var) / (prior_theta_var + lidar_theta_var)

        w_prior = lidar_theta_var / (prior_theta_var + lidar_theta_var)
        w_lidar = prior_theta_var / (prior_theta_var + lidar_theta_var)

        sin_theta = w_prior * np.sin(prior_theta) + w_lidar * np.sin(pca_angle)
        cos_theta = w_prior * np.cos(prior_theta) + w_lidar * np.cos(pca_angle)
        est_theta = np.arctan2(sin_theta, cos_theta)

        # Compute dimensions in room-aligned frame
        cos_t = np.cos(-est_theta)
        sin_t = np.sin(-est_theta)
        aligned_x = cos_t * points_np[:, 0] - sin_t * points_np[:, 1]
        aligned_y = sin_t * points_np[:, 0] + cos_t * points_np[:, 1]

        max_x, min_x = np.max(aligned_x), np.min(aligned_x)
        max_y, min_y = np.max(aligned_y), np.min(aligned_y)

        lidar_dim_x = max_x - min_x + 0.2
        lidar_dim_y = max_y - min_y + 0.2

        # Assign dimensions ensuring width >= length
        prior_width = 6.0
        prior_length = 4.0
        prior_dim_var = 1.0 ** 2
        lidar_dim_var = 1.0 ** 2

        if lidar_dim_x >= lidar_dim_y:
            lidar_width = lidar_dim_x
            lidar_length = lidar_dim_y
            theta_adjustment = 0.0
        else:
            lidar_width = lidar_dim_y
            lidar_length = lidar_dim_x
            theta_adjustment = np.pi / 2
            est_theta += theta_adjustment
            est_theta = np.arctan2(np.sin(est_theta), np.cos(est_theta))

        # Bayesian fusion with priors
        width_var = (prior_dim_var * lidar_dim_var) / (prior_dim_var + lidar_dim_var)
        length_var = (prior_dim_var * lidar_dim_var) / (prior_dim_var + lidar_dim_var)

        est_width = (lidar_dim_var * prior_width + prior_dim_var * lidar_width) / (prior_dim_var + lidar_dim_var)
        est_length = (lidar_dim_var * prior_length + prior_dim_var * lidar_length) / (prior_dim_var + lidar_dim_var)

        est_width = np.clip(est_width, 3.0, 12.0)
        est_length = np.clip(est_length, 2.0, 8.0)

        # Estimate robot position
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0

        cos_t_inv = np.cos(est_theta)
        sin_t_inv = np.sin(est_theta)
        est_pos_x = cos_t_inv * center_x - sin_t_inv * center_y
        est_pos_y = sin_t_inv * center_x + cos_t_inv * center_y

        # Fuse with GT prior for position
        if robot_pose_gt is not None:
            pos_prior_var = 0.5
            pos_lidar_var = 2.0
            est_pos_x = (pos_lidar_var * robot_pose_gt[0] + pos_prior_var * est_pos_x) / (pos_prior_var + pos_lidar_var)
            est_pos_y = (pos_lidar_var * robot_pose_gt[1] + pos_prior_var * est_pos_y) / (pos_prior_var + pos_lidar_var)

        return est_width, est_length, width_var, length_var, est_theta, theta_var, est_pos_x, est_pos_y

    def _initialization_phase(self,
                              lidar_points: torch.Tensor,
                              robot_pose_gt: np.ndarray) -> Dict:
        """
        Phase 1: Estimate initial pose AND room dimensions from LIDAR.

        Strategy:
        1. Collect LIDAR points from multiple frames
        2. Estimate room dimensions and orientation using Bayesian fusion with Gaussian priors
        3. Optimize full state (x, y, theta, W, L) to minimize SDF error
        4. Once converged, freeze room dimensions and switch to tracking

        Uses PyTorch autograd for gradient computation.
        """
        # Collect points
        if len(lidar_points) > 0:
            self.init_buffer.append(lidar_points)

        # Need enough points
        total_points = sum(len(p) for p in self.init_buffer)
        if total_points < self.min_points_for_init:
            return {
                'phase': 'init',
                'status': 'collecting',
                'points_collected': total_points,
                'points_needed': self.min_points_for_init
            }

        # Initialize belief if needed
        if self.belief is None:
            all_points = torch.cat(self.init_buffer, dim=0)

            if self.use_known_room_dimensions:
                # Use known dimensions, only estimate pose
                est_width = self.true_width
                est_height = self.true_height
                width_var = 0.01
                height_var = 0.01
                theta_var = 0.5  # Default theta variance when using known dimensions
                init_x = robot_pose_gt[0]
                init_y = robot_pose_gt[1]
                init_theta = robot_pose_gt[2]
            else:
                # Estimate everything from LIDAR with Gaussian priors
                (est_width, est_height, width_var, height_var,
                 init_theta, theta_var, init_x, init_y) = self._estimate_initial_state_from_lidar(
                    all_points, robot_pose_gt
                )

            # Create initial belief
            self.belief = RoomBelief(
                x=init_x,
                y=init_y,
                theta=init_theta,
                width=est_width,
                length=est_height
            )

            # Set covariance reflecting uncertainty
            self.belief.cov = np.diag([
                1.0,        # x position uncertainty (1m std)
                1.0,        # y position uncertainty (1m std)
                theta_var,  # theta uncertainty
                width_var,
                height_var
            ])

        # Concatenate all points
        all_points = torch.cat(self.init_buffer, dim=0)

        # Create optimizable parameters
        if self.use_known_room_dimensions:
            # Only optimize pose, not room dimensions
            pose_params = torch.tensor([
                self.belief.x,
                self.belief.y,
                self.belief.theta,
            ], dtype=DTYPE, device=DEVICE, requires_grad=True)
            fixed_width = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
            fixed_length = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)
            optimizer = torch.optim.Adam([pose_params], lr=self.lr_init)
        else:
            params = torch.tensor([
                self.belief.x,
                self.belief.y,
                self.belief.theta,
                self.belief.width,
                self.belief.length
            ], dtype=DTYPE, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([params], lr=self.lr_init)

        # Store initial pose for regularization
        initial_pose = torch.tensor([self.belief.x, self.belief.y, self.belief.theta],
                                    dtype=DTYPE, device=DEVICE)

        # Early stopping variables
        prev_loss = float('inf')
        iterations_used = 0
        max_iters = self.max_iterations_init if self.adaptive_cpu else 100

        for iteration in range(max_iters):
            optimizer.zero_grad()

            if self.use_known_room_dimensions:
                # Use fixed dimensions
                sdf_values = self.sdf_rect(
                    all_points,
                    pose_params[0], pose_params[1], pose_params[2],
                    fixed_width, fixed_length
                )
                sdf_loss = torch.mean(sdf_values ** 2)

                # Regularization: keep pose close to initial guess (weak)
                pose_reg = 0.01 * (torch.sum((pose_params[:2] - initial_pose[:2]) ** 2) +
                                 0.01 * (pose_params[2] - initial_pose[2]) ** 2)
                loss = sdf_loss + pose_reg
            else:
                # Compute SDF for all points
                sdf_values = self.sdf_rect(
                    all_points,
                    params[0], params[1], params[2],  # x, y, theta
                    params[3], params[4]  # width, length
                )
                sdf_loss = torch.mean(sdf_values ** 2)

                # Regularization: penalize if dimensions go below 1m (soft constraint)
                size_reg = 0.001 * (torch.relu(1.0 - params[3]) ** 2 + torch.relu(1.0 - params[4]) ** 2)

                # Regularization: keep pose close to initial guess
                # Position regularization (weak - allow optimization)
                pos_reg = 0.01 * torch.sum((params[:2] - initial_pose[:2]) ** 2)

                # Angle regularization (stronger - prevent 180° flip due to room symmetry)
                # Use sin to handle angle wrapping properly
                angle_diff = params[2] - initial_pose[2]
                angle_reg = 0.1 * (torch.sin(angle_diff / 2) ** 2)

                loss = sdf_loss + size_reg + pos_reg + angle_reg

            iterations_used = iteration + 1

            # Early stopping check (after minimum iterations)
            if self.adaptive_cpu and iteration >= self.min_iterations:
                loss_val = loss.item()
                improvement = prev_loss - loss_val
                if improvement < self.early_stop_threshold and improvement >= 0:
                    break  # Converged
                prev_loss = loss_val

            loss.backward()

            # Clip gradients
            if self.use_known_room_dimensions:
                torch.nn.utils.clip_grad_norm_([pose_params], max_norm=100.0)
            else:
                torch.nn.utils.clip_grad_norm_([params], max_norm=100.0)

            optimizer.step()

            # Enforce constraints
            with torch.no_grad():
                if self.use_known_room_dimensions:
                    pose_params[2] = torch.atan2(torch.sin(pose_params[2]), torch.cos(pose_params[2]))
                else:
                    # Clamp room dimensions to reasonable bounds (in meters)
                    params[3] = torch.clamp(params[3], min=2.0, max=20.0)  # width: 2-20m
                    params[4] = torch.clamp(params[4], min=2.0, max=20.0)  # length: 2-20m
                    params[2] = torch.atan2(torch.sin(params[2]), torch.cos(params[2]))  # theta: [-π, π]

        # Extract final parameters
        # Update belief
        with torch.no_grad():
            if self.use_known_room_dimensions:
                self.belief.x = pose_params[0].item()
                self.belief.y = pose_params[1].item()
                self.belief.theta = pose_params[2].item()
                # Width and length stay fixed

                # Compute final SDF error
                sdf_values = self.sdf_rect(
                    all_points,
                    pose_params[0], pose_params[1], pose_params[2],
                    fixed_width, fixed_length
                )
            else:
                self.belief.x = params[0].item()
                self.belief.y = params[1].item()
                self.belief.theta = params[2].item()
                opt_width = params[3].item()
                opt_length = params[4].item()

                # ENFORCE width >= length constraint
                # Due to room symmetry, optimizer might swap them
                if opt_width >= opt_length:
                    self.belief.width = opt_width
                    self.belief.length = opt_length
                else:
                    # Swap dimensions and rotate theta by 90°
                    self.belief.width = opt_length
                    self.belief.length = opt_width
                    self.belief.theta += np.pi / 2
                    # Normalize theta to [-pi, pi]
                    while self.belief.theta > np.pi:
                        self.belief.theta -= 2 * np.pi
                    while self.belief.theta < -np.pi:
                        self.belief.theta += 2 * np.pi

                # Compute final SDF error using corrected values
                sdf_values = self.sdf_rect(
                    all_points,
                    torch.tensor(self.belief.x, dtype=DTYPE, device=DEVICE),
                    torch.tensor(self.belief.y, dtype=DTYPE, device=DEVICE),
                    torch.tensor(self.belief.theta, dtype=DTYPE, device=DEVICE),
                    torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE),
                    torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)
                )

            mean_sdf_error = torch.mean(torch.abs(sdf_values)).item()

        self.init_iterations += 1
        self.stats['sdf_error_history'].append(mean_sdf_error)
        self.stats['iterations_used'].append(iterations_used)
        self._last_sdf_error = mean_sdf_error  # Update for adaptive CPU

        # Check convergence (no ground truth available!)
        # We can only check SDF error and iteration count
        room_error = (abs(self.belief.width - self.true_width) +
                      abs(self.belief.length - self.true_height)) / 2  # For reporting only

        converged = (mean_sdf_error < self.init_convergence_threshold or
                     self.init_iterations >= self.max_init_iterations)

        if converged:
            self.phase = 'tracking'
            self.belief.converged = True

            # Reduce room uncertainty (room is now fixed)
            self.belief.cov[3, 3] = 0.01
            self.belief.cov[4, 4] = 0.01

            # Store last known pose for prediction
            self._last_pose = self.belief.pose[:2].copy()
            self._last_theta = self.belief.theta

            print(f"\n[ROOM ESTIMATION] Initialization complete")
            print(f"  Iterations: {self.init_iterations}")
            print(f"  Mean SDF error: {mean_sdf_error:.4f}m")
            print(f"  Estimated room: {self.belief.width:.2f} x {self.belief.length:.2f}m")
            print(f"  True room: {self.true_width:.2f} x {self.true_height:.2f}m")
            print(f"  Room error: {room_error:.3f}m")
            print(f"  Estimated pose: [{self.belief.x:.2f}, {self.belief.y:.2f}, {self.belief.theta:.2f}]\n")

        return {
            'phase': 'init',
            'status': 'converged' if converged else 'optimizing',
            'iteration': self.init_iterations,
            'sdf_error': mean_sdf_error,
            'pose': self.belief.pose.copy(),
            'room_params': self.belief.room_params.copy(),
            'room_error': room_error,
            'belief_uncertainty': self.belief.uncertainty()
        }

    def _tracking_phase(self,
                        lidar_points: torch.Tensor,
                        robot_velocity: np.ndarray,
                        angular_velocity: float,
                        dt: float) -> Dict:
        """
        Phase 2: Room fixed, update pose by minimizing Free Energy.

        NO GROUND TRUTH USED. Uses dead reckoning + LIDAR correction.
        """
        if len(lidar_points) == 0:
            return {
                'phase': 'tracking',
                'pose': self.belief.pose.copy(),
                'pose_cov': self.belief.pose_cov,
                'sdf_error': 0,
                'belief_uncertainty': self.belief.uncertainty()
            }

        # =====================================================================
        # PREDICTION STEP: Motion model
        # =====================================================================
        s_prev = np.array([self.belief.x, self.belief.y, self.belief.theta])
        Sigma_prev = self.belief.cov[:3, :3].copy()

        # Transform velocity from robot frame to world frame
        cos_theta = np.cos(s_prev[2])
        sin_theta = np.sin(s_prev[2])
        vx_world = robot_velocity[0] * cos_theta - robot_velocity[1] * sin_theta
        vy_world = robot_velocity[0] * sin_theta + robot_velocity[1] * cos_theta

        # Predicted state from motion model
        s_pred = np.array([
            s_prev[0] + vx_world * dt,
            s_prev[1] + vy_world * dt,
            s_prev[2] + angular_velocity * dt
        ])
        s_pred[2] = np.arctan2(np.sin(s_pred[2]), np.cos(s_pred[2]))

        # Jacobian of motion model
        F_t = np.eye(3)
        F_t[0, 2] = -vy_world * dt
        F_t[1, 2] = vx_world * dt

        # Process noise (increases with motion)
        speed = np.linalg.norm(robot_velocity)
        Q_scale = max(0.1, speed / 0.2)  # Scale with speed
        Q = self.Q_pose * dt * Q_scale

        # Predicted covariance
        Sigma_pred = F_t @ Sigma_prev @ F_t.T + Q

        # =====================================================================
        # CORRECTION STEP: SDF optimization with prior
        # Free Energy F = F_likelihood + F_prior
        # F_likelihood = sum(SDF²) / (2 * σ_sdf²)
        # F_prior = 0.5 * (s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)
        # =====================================================================
        width = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
        length = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)

        # Compute velocity-adaptive weights for [x, y, theta]
        velocity_weights = self._compute_velocity_adaptive_weights(robot_velocity, angular_velocity)
        velocity_weights_t = torch.tensor(velocity_weights, dtype=DTYPE, device=DEVICE)

        # Apply velocity weights to prediction covariance
        # Higher weight = lower uncertainty = more trust in that parameter's optimization
        # We scale the covariance inversely with weights (higher weight -> lower covariance)
        weight_scale = np.diag(1.0 / (velocity_weights + 1e-6))
        Sigma_pred_weighted = weight_scale @ Sigma_pred @ weight_scale

        # Initialize at PREDICTED pose
        pose_params = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE, requires_grad=True)
        s_pred_t = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE)

        # Prior information (inverse covariance) - now weighted
        Sigma_pred_inv = torch.tensor(np.linalg.inv(Sigma_pred_weighted + 1e-6 * np.eye(3)),
                                       dtype=DTYPE, device=DEVICE)

        # Prior precision (balances odometry vs LIDAR)
        # Uses adaptive precision from previous step, or default if first iteration
        if hasattr(self, '_adaptive_prior_precision'):
            prior_precision = self._adaptive_prior_precision
        else:
            prior_precision = 0.1  # Initial default

        optimizer = torch.optim.Adam([pose_params], lr=0.05)

        # Early stopping variables for tracking
        prev_loss = float('inf')
        iterations_used = 0
        max_iters = self.max_iterations_tracking if self.adaptive_cpu else 50

        for iteration in range(max_iters):
            optimizer.zero_grad()

            # Likelihood: SDF error
            sdf_values = self.sdf_rect(lidar_points, pose_params[0], pose_params[1],
                                       pose_params[2], width, length)
            F_likelihood = torch.mean(sdf_values ** 2)

            # Prior: deviation from motion model prediction
            pose_diff = pose_params - s_pred_t
            # Handle angle wrapping
            pose_diff_wrapped = pose_diff.clone()
            pose_diff_wrapped[2] = torch.atan2(torch.sin(pose_diff[2]), torch.cos(pose_diff[2]))
            F_prior = 0.5 * pose_diff_wrapped @ Sigma_pred_inv @ pose_diff_wrapped

            # Total Free Energy
            F = F_likelihood + prior_precision * F_prior

            iterations_used = iteration + 1

            # Early stopping check (after minimum iterations)
            if self.adaptive_cpu and iteration >= self.min_iterations:
                loss_val = F.item()
                improvement = prev_loss - loss_val
                if improvement < self.early_stop_threshold and improvement >= 0:
                    break  # Converged
                prev_loss = loss_val

            F.backward()

            # Apply velocity-adaptive gradient weighting
            # Scale gradients by velocity weights: higher weight = stronger update
            with torch.no_grad():
                if pose_params.grad is not None:
                    pose_params.grad *= velocity_weights_t

            optimizer.step()

            with torch.no_grad():
                pose_params[2] = torch.atan2(torch.sin(pose_params[2]), torch.cos(pose_params[2]))

        # Track iterations used
        self.stats['iterations_used'].append(iterations_used)

        optimized_pose = np.array([pose_params[0].item(), pose_params[1].item(), pose_params[2].item()])

        # =====================================================================
        # ADAPTIVE PRIOR PRECISION based on prediction error (innovation)
        # In Active Inference, precision = inverse variance = confidence
        # High precision → trust motion model more
        # Low precision → trust LIDAR observations more
        # =====================================================================
        innovation = optimized_pose - s_pred
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))

        # Mahalanobis distance of innovation
        innovation_mahal = np.sqrt(innovation @ np.linalg.inv(Sigma_pred + 1e-6 * np.eye(3)) @ innovation)

        # Adaptive precision formula:
        # - Base precision is low to prefer LIDAR (exteroceptive)
        # - Increase if innovation is small (motion model predicts well)
        # - max_precision caps proprioceptive confidence
        if not hasattr(self, '_adaptive_prior_precision'):
            self._adaptive_prior_precision = 0.1  # Initial value

        # Parameters for adaptive precision
        base_precision = 0.05   # Minimum (LIDAR always dominates somewhat)
        max_precision = 0.3     # Maximum (never trust odometry too much)
        scale = 0.5             # Sensitivity to innovation

        # Precision decreases exponentially with innovation
        target_precision = base_precision + (max_precision - base_precision) * np.exp(-innovation_mahal / scale)
        target_precision = np.clip(target_precision, base_precision, max_precision)

        # Exponential moving average for smooth adaptation
        alpha = 0.2
        self._adaptive_prior_precision = (1 - alpha) * self._adaptive_prior_precision + alpha * target_precision

        # =====================================================================
        # POSTERIOR COVARIANCE via Hessian of the cost function
        # Formula: Σ = σ² × H⁻¹ where σ² is the residual variance
        # This scales the geometric uncertainty (H⁻¹) by the actual fit quality (σ²)
        # =====================================================================
        N_obs = len(lidar_points)
        pose_opt = torch.tensor(optimized_pose, dtype=DTYPE, device=DEVICE, requires_grad=True)
        sdf_opt = self.sdf_rect(lidar_points, pose_opt[0], pose_opt[1], pose_opt[2], width, length)

        # Residual variance: how well the model fits the data
        residual_var = torch.mean(sdf_opt ** 2).item()

        # Use MEAN like C++ RoomLoss::compute_loss
        loss = torch.mean(sdf_opt ** 2)

        # Compute gradient
        grad = torch.autograd.grad(loss, pose_opt, create_graph=True)[0]

        # Compute Hessian
        hessian = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE)
        for i in range(3):
            hessian[i] = torch.autograd.grad(grad[i], pose_opt, retain_graph=True)[0]
        hessian = 0.5 * (hessian + hessian.T)  # Ensure symmetry

        try:
            hessian_np = hessian.cpu().numpy()

            # Robust inversion with regularization
            reg = 1e-6
            max_attempts = 6
            success = False

            for attempt in range(max_attempts):
                h_reg = hessian_np + reg * np.eye(3)
                try:
                    L = np.linalg.cholesky(h_reg)
                    H_inv = np.linalg.inv(h_reg)
                    success = True
                    break
                except np.linalg.LinAlgError:
                    reg *= 10.0

            if not success:
                H_inv = np.linalg.pinv(hessian_np + reg * np.eye(3))

            # Covariance = σ² × H⁻¹
            # This makes covariance small when residuals are small (good fit)
            # and large when residuals are large (poor fit)
            Sigma_post = residual_var * H_inv

        except Exception as e:
            # Fallback
            Sigma_post = np.diag([0.01, 0.01, 0.01])

        # Compute final SDF error for diagnostics
        mean_sdf_error = torch.mean(torch.abs(sdf_opt)).detach().item()
        self._last_sdf_error = mean_sdf_error  # Update for adaptive CPU


        # =====================================================================
        # UPDATE BELIEF
        # =====================================================================
        self.belief.x = optimized_pose[0]
        self.belief.y = optimized_pose[1]
        self.belief.theta = optimized_pose[2]
        self.belief.cov[:3, :3] = Sigma_post

        self.stats['sdf_error_history'].append(mean_sdf_error)

        return {
            'phase': 'tracking',
            'pose': self.belief.pose.copy(),
            'pose_cov': self.belief.pose_cov,
            'sdf_error': mean_sdf_error,
            'room_params': self.belief.room_params.copy(),
            'belief_uncertainty': self.belief.uncertainty(),
            'innovation': innovation,  # [dx, dy, dtheta] prediction error
            'prior_precision': self._adaptive_prior_precision,
            'iterations_used': iterations_used,
            'lidar_points_used': len(lidar_points),
            'velocity_weights': velocity_weights  # [w_x, w_y, w_theta] velocity-adaptive weights
        }

    def is_converged(self) -> bool:
        """Check if room estimation has converged."""
        return self.belief is not None and self.belief.converged

    def get_pose_and_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current pose estimate and covariance.

        Used by obstacle models for updating.
        """
        if self.belief is None:
            return np.zeros(3), np.eye(3) * 1000.0

        return self.belief.pose.copy(), self.belief.pose_cov

    def reset(self):
        """Reset estimator state."""
        self.belief = None
        self.phase = 'init'
        self.init_buffer = []
        self.init_iterations = 0
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': []
        }
