import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from src.pointcloud_center_estimator import PointcloudCenterEstimator, EstimatorConfig, OBB


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
        self.init_convergence_threshold = 0.10  # 10cm threshold - stricter for good initialization
        self.max_init_iterations = 100  # Max iterations per search attempt
        self.init_iterations = 0

        # Tracking parameters - adjusted for meters
        # sigma_sdf represents the effective uncertainty in wall distance measurements
        # This includes sensor noise + dynamic effects (motion, rotation, timing)
        # Needs to be large enough that prior (motion model) remains influential
        self.sigma_sdf = 0.15  # 15cm - accounts for dynamic uncertainty during motion

        # Process noise Q_pose: controls how fast uncertainty grows during prediction
        # These are BASE values that get scaled by dt and speed
        # Lower values = slower covariance growth, more trust in motion model
        # For dt=0.1s and stationary robot, effective Q ≈ Q_pose * 0.1 * 0.1 = Q_pose * 0.01
        self.Q_pose = np.diag([0.002, 0.002, 0.001])  # Reduced from [0.015, 0.015, 0.01]

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
        self.lr_init = 0.1  # Higher learning rate for faster convergence
        self.lr_tracking = 0.01  # Reduced to avoid oscillations (was 0.1, then 0.02)

        # Dynamic CPU optimization parameters
        self.adaptive_cpu = True  # Enable adaptive CPU optimization
        self.min_iterations = 25  # Increased to ensure convergence (was 5, then 15)
        self.max_iterations_init = 100  # Max iterations for init phase
        self.max_iterations_tracking = 50  # Max iterations for tracking phase
        self.early_stop_threshold = 1e-6  # Tighter threshold (was 1e-5)
        self.skip_frames_when_stable = True  # Skip processing when stable
        self.stability_threshold = 0.05  # SDF error below this = stable (5cm)
        self.max_skip_frames = 3  # Maximum frames to skip when stable
        self._skip_counter = 0   # Current skip counter
        self._last_sdf_error = float('inf')  # Last SDF error for stability check
        self.lidar_subsample_factor = 1  # Subsample LIDAR points (1 = no subsampling)
        self.adaptive_subsampling = False  # DISABLED - subsampling now done at LIDAR proxy level
        self.min_lidar_points = 100  # Minimum LIDAR points to keep

        # Prediction-based early exit: if prediction SDF < threshold, skip optimization
        # This saves CPU when motion model is accurate (smooth motion)
        self.prediction_early_exit = True  # Enable prediction-based early exit
        self.prediction_trust_factor = 0.5  # Trust threshold = sigma_sdf * factor
        self.min_tracking_steps_for_early_exit = 50  # Wait for room to stabilize
        self._tracking_step_count = 0  # Counter for tracking steps

        # Velocity auto-calibration based on innovation
        # If motion model consistently under/over-predicts, learn correction factor
        self.velocity_calibration_enabled = True
        self.velocity_scale_factor = 1.0  # Multiplicative correction: v_real = k * v_commanded
        self._innovation_accumulator = np.zeros(3)  # Accumulated innovation [x, y, theta]
        self._velocity_accumulator = np.zeros(3)    # Accumulated velocity * dt [vx*dt, vy*dt, omega*dt]
        self._calibration_samples = 0               # Number of samples collected
        self._calibration_window = 50               # Samples before updating calibration (reduced from 100)
        self._calibration_learning_rate = 0.2       # How fast to adapt (increased from 0.1)
        self._min_motion_for_calibration = 0.005    # Min displacement to count sample (m) (reduced from 0.01)

        # Initialization state
        self._hierarchical_search_done = False  # Track if hierarchical search has been performed
        self._init_restart_count = 0  # Track number of restarts

        # Statistics
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': [],
            'iterations_used': [],  # Track actual iterations per step
            'prediction_early_exits': 0,  # Track prediction-based early exits
            'frames_skipped': 0,     # Track skipped frames
            'velocity_scale_history': []  # Track velocity calibration factor
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

    def _update_velocity_calibration(self, innovation: np.ndarray, commanded_displacement: np.ndarray):
        """
        Update velocity calibration factor based on accumulated innovation.

        The innovation tells us how much the LIDAR-corrected pose differs from
        the motion model prediction. If the robot consistently moves more/less
        than predicted, we adjust the velocity scale factor.

        Method:
        - Accumulate innovation and commanded displacement over a window
        - Compute ratio: actual_motion / predicted_motion
        - Update scale factor with learning rate

        Args:
            innovation: [dx, dy, dtheta] difference between optimized and predicted pose
            commanded_displacement: [dx, dy, dtheta] predicted displacement from velocity * dt
        """
        # Only calibrate when there's significant motion
        displacement_magnitude = np.linalg.norm(commanded_displacement[:2])

        # Debug: only print when there IS motion (to avoid spam when robot is static)
        if displacement_magnitude >= self._min_motion_for_calibration:
            if self._calibration_samples % 20 == 0:
                print(f"[VelCal] Accumulating: disp={displacement_magnitude:.4f}m, "
                      f"samples={self._calibration_samples}/{self._calibration_window}")

        if displacement_magnitude < self._min_motion_for_calibration:
            return
            return
            return

        # Accumulate innovation and commanded motion
        self._innovation_accumulator += innovation
        self._velocity_accumulator += commanded_displacement
        self._calibration_samples += 1

        # Update calibration when window is full
        if self._calibration_samples >= self._calibration_window:
            # The innovation tells us: actual - predicted
            # predicted = k_old * commanded
            # actual = k_true * commanded
            # innovation = (k_true - k_old) * commanded
            # Therefore: k_true = k_old + innovation / commanded

            accumulated_cmd_magnitude = np.linalg.norm(self._velocity_accumulator[:2])

            if accumulated_cmd_magnitude > 0.05:  # Need significant motion to calibrate
                # Project innovation onto direction of motion
                motion_direction = self._velocity_accumulator[:2] / accumulated_cmd_magnitude
                innovation_along_motion = np.dot(self._innovation_accumulator[:2], motion_direction)

                # k_correction = innovation / commanded (both in same direction)
                k_correction = innovation_along_motion / accumulated_cmd_magnitude

                # New estimate of true scale factor
                k_estimated = self.velocity_scale_factor + k_correction

                # Clamp to reasonable range [0.5, 1.5]
                k_estimated = np.clip(k_estimated, 0.5, 1.5)

                # Update with learning rate (exponential moving average)
                old_k = self.velocity_scale_factor
                self.velocity_scale_factor = (
                    (1 - self._calibration_learning_rate) * self.velocity_scale_factor +
                    self._calibration_learning_rate * k_estimated
                )

                # Track history
                self.stats['velocity_scale_history'].append(self.velocity_scale_factor)

                # Print calibration update
                print(f"[Velocity Calibration] k: {old_k:.3f} → {self.velocity_scale_factor:.3f} "
                      f"(innov={innovation_along_motion:.4f}m, cmd={accumulated_cmd_magnitude:.4f}m, "
                      f"corr={k_correction:+.3f})")

            # Reset accumulators
            self._innovation_accumulator = np.zeros(3)
            self._velocity_accumulator = np.zeros(3)
            self._calibration_samples = 0


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
        half_width = width * 0.5
        half_length = length * 0.5

        # Distance to each wall (absolute, unsigned)
        d_left = torch.abs(room_x + half_width)   # Distance to x = -width/2
        d_right = torch.abs(room_x - half_width)  # Distance to x = +width/2
        d_bottom = torch.abs(room_y + half_length)  # Distance to y = -length/2
        d_top = torch.abs(room_y - half_length)     # Distance to y = +length/2

        # OPTIMIZED: Use torch.minimum chain instead of stack+min (avoids tensor allocation)
        min_dist = torch.minimum(torch.minimum(d_left, d_right),
                                  torch.minimum(d_bottom, d_top))

        return min_dist


    def update(self,
               robot_pose_gt: np.ndarray = None,
               robot_velocity: np.ndarray = None,
               angular_velocity: float = 0.0,
               dt: float = 0.1,
               lidar_points: np.ndarray = None) -> Dict:
        """
        Update room and pose estimates using Active Inference.

        This method implements GT-free estimation: ground truth is NOT used
        for state estimation. The robot_pose_gt parameter is kept only for
        API compatibility and performance evaluation.

        INIT phase: Uses PointcloudCenterEstimator (OBB-based) for initial state
        TRACKING phase: Uses motion model + LIDAR SDF correction

        Args:
            robot_pose_gt: Ground truth pose [x, y, theta] - NOT USED for estimation,
                           only for performance metrics and compatibility
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

        # Get LIDAR points - OPTIMIZED: use from_numpy for zero-copy when possible
        if lidar_points is not None and len(lidar_points) > 0:
            # Ensure contiguous float32 array for zero-copy transfer
            if lidar_points.dtype != np.float32:
                lidar_points = lidar_points.astype(np.float32)
            if not lidar_points.flags['C_CONTIGUOUS']:
                lidar_points = np.ascontiguousarray(lidar_points)
            lidar_points_t = torch.from_numpy(lidar_points).to(device=DEVICE)
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
                                           robot_pose_gt: np.ndarray = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Estimate room dimensions and robot pose from LIDAR points WITHOUT ground truth.

        Uses a HIERARCHICAL SEARCH approach for robust initialization:

        Level 1 (Coarse): OBB-based estimation for room dimensions and initial pose candidates
        Level 2 (Medium): Grid search around top candidates with multiple angles
        Level 3 (Fine): Local optimization from best candidates

        This method is GT-free: robot_pose_gt parameter is kept for API compatibility
        but is NOT used in the estimation.

        Mathematical basis (main.tex Sec. 5.2.1):
            The room geometry defines the generative model. By finding the configuration
            that minimizes SDF error across multiple hypotheses, we robustly infer
            the latent room state (W, L) and robot pose (x, y, θ).

        Args:
            all_points: [N, 2] LIDAR points in robot frame (torch tensor)
            robot_pose_gt: UNUSED - kept for API compatibility only

        Returns:
            (est_width, est_length, width_var, length_var, est_theta, theta_var, est_pos_x, est_pos_y)
        """
        points_np = all_points.cpu().numpy()

        # =====================================================================
        # LEVEL 1: Coarse estimation using OBB
        # =====================================================================
        estimator = PointcloudCenterEstimator(EstimatorConfig(
            min_range=0.3,
            max_range=10.0,
            num_sectors=72,
            min_valid_points=20,
            outlier_std_threshold=2.0
        ))

        obb = estimator.estimate(points_np)

        if obb is not None:
            # Initial estimates from OBB
            init_pos_x = -obb.center[0]
            init_pos_y = -obb.center[1]
            init_width = obb.width
            init_length = obb.height
            init_theta = -obb.rotation
            init_theta = np.arctan2(np.sin(init_theta), np.cos(init_theta))
        else:
            # Fallback: simple statistics
            print("[RoomEstimator] OBB failed, using statistical fallback")
            centroid = np.median(points_np, axis=0)
            init_pos_x = -centroid[0]
            init_pos_y = -centroid[1]
            init_width = np.max(points_np[:, 0]) - np.min(points_np[:, 0])
            init_length = np.max(points_np[:, 1]) - np.min(points_np[:, 1])
            if init_width < init_length:
                init_width, init_length = init_length, init_width
                init_theta = np.pi / 2
            else:
                init_theta = 0.0

        # Apply Gaussian priors for room dimensions
        prior_width = 6.0
        prior_length = 4.0
        prior_dim_var = 1.0
        n_points = len(points_np)
        base_var = 1.0 / np.sqrt(n_points) if n_points > 0 else 1.0
        width_var = base_var * 0.5
        length_var = base_var * 0.5

        # Bayesian fusion with priors
        est_width = (prior_dim_var * init_width + width_var * prior_width) / (prior_dim_var + width_var)
        est_length = (prior_dim_var * init_length + length_var * prior_length) / (prior_dim_var + length_var)
        est_width = np.clip(est_width, 3.0, 12.0)
        est_length = np.clip(est_length, 2.0, 8.0)

        # =====================================================================
        # LEVEL 2: Hierarchical grid search for pose
        # =====================================================================
        est_pos_x, est_pos_y, est_theta, best_score = self._hierarchical_pose_search(
            all_points, est_width, est_length, init_pos_x, init_pos_y, init_theta
        )

        # Variance estimates based on search quality
        theta_var = base_var * 0.3
        width_var = (prior_dim_var * width_var) / (prior_dim_var + width_var)
        length_var = (prior_dim_var * length_var) / (prior_dim_var + length_var)

        print(f"[Hierarchical Init] Best pose: ({est_pos_x:.2f}, {est_pos_y:.2f}, {np.degrees(est_theta):.1f}°), score: {best_score:.4f}")

        return est_width, est_length, width_var, length_var, est_theta, theta_var, est_pos_x, est_pos_y

    def _hierarchical_pose_search(self,
                                   all_points: torch.Tensor,
                                   width: float,
                                   length: float,
                                   init_x: float,
                                   init_y: float,
                                   init_theta: float) -> Tuple[float, float, float, float]:
        """
        Hierarchical multi-scale search for optimal robot pose.

        Three-level search:
        - Level 1 (Coarse): Wide grid with many angles (finds general region)
        - Level 2 (Medium): Refined grid around top candidates
        - Level 3 (Fine): Local optimization from best candidates

        Uses combined score: mean SDF + inlier ratio penalty

        Args:
            all_points: LIDAR points [N, 2] in robot frame
            width, length: Estimated room dimensions
            init_x, init_y, init_theta: Initial pose estimates from OBB

        Returns:
            (best_x, best_y, best_theta, best_score)
        """
        width_t = torch.tensor(width, dtype=DTYPE, device=DEVICE)
        length_t = torch.tensor(length, dtype=DTYPE, device=DEVICE)

        print(f"\n[Hierarchical Search] Starting with OBB estimate: ({init_x:.2f}, {init_y:.2f}, {np.degrees(init_theta):.1f}°)")
        print(f"[Hierarchical Search] Room dimensions: {width:.2f} x {length:.2f}m")

        # =====================================================================
        # LEVEL 1: Coarse grid search
        # =====================================================================
        # Define search bounds based on room dimensions
        # Robot must be inside the room
        max_x = width / 2 - 0.3   # Keep 30cm from walls
        max_y = length / 2 - 0.3

        # Coarse grid: 5x5 positions, 16 angles
        coarse_positions = []
        for dx in np.linspace(-0.8, 0.8, 5):  # ±0.8m around initial estimate
            for dy in np.linspace(-0.8, 0.8, 5):
                px = np.clip(init_x + dx, -max_x, max_x)
                py = np.clip(init_y + dy, -max_y, max_y)
                coarse_positions.append((px, py))

        # 16 angles covering full circle
        coarse_angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)

        # Evaluate all coarse candidates
        coarse_candidates = []
        for (px, py) in coarse_positions:
            for theta in coarse_angles:
                score, inlier_ratio = self._evaluate_pose_candidate(
                    all_points, px, py, theta, width_t, length_t
                )
                # Combined score: lower is better
                # Penalize low inlier ratio (many points far from walls)
                combined_score = score * (1.0 + 0.5 * (1.0 - inlier_ratio))
                coarse_candidates.append((px, py, theta, combined_score, inlier_ratio))

        # Sort by score and keep top-K
        coarse_candidates.sort(key=lambda x: x[3])
        top_k = min(10, len(coarse_candidates))
        top_coarse = coarse_candidates[:top_k]

        print(f"[Level 1 - Coarse] Evaluated {len(coarse_candidates)} candidates")
        print(f"[Level 1 - Coarse] Best: ({top_coarse[0][0]:.2f}, {top_coarse[0][1]:.2f}, {np.degrees(top_coarse[0][2]):.1f}°) score={top_coarse[0][3]:.4f}")

        # =====================================================================
        # LEVEL 2: Medium grid search around top candidates
        # =====================================================================
        medium_candidates = []
        for (cx, cy, ct, _, _) in top_coarse:
            # Finer grid: ±0.2m, ±15° around each top candidate
            for dx in np.linspace(-0.2, 0.2, 5):
                for dy in np.linspace(-0.2, 0.2, 5):
                    for dtheta in np.linspace(-0.26, 0.26, 5):  # ±15°
                        px = np.clip(cx + dx, -max_x, max_x)
                        py = np.clip(cy + dy, -max_y, max_y)
                        theta = ct + dtheta
                        theta = np.arctan2(np.sin(theta), np.cos(theta))

                        score, inlier_ratio = self._evaluate_pose_candidate(
                            all_points, px, py, theta, width_t, length_t
                        )
                        combined_score = score * (1.0 + 0.5 * (1.0 - inlier_ratio))
                        medium_candidates.append((px, py, theta, combined_score, inlier_ratio))

        # Sort and keep top-M
        medium_candidates.sort(key=lambda x: x[3])
        top_m = min(5, len(medium_candidates))
        top_medium = medium_candidates[:top_m]

        print(f"[Level 2 - Medium] Evaluated {len(medium_candidates)} candidates")
        print(f"[Level 2 - Medium] Best: ({top_medium[0][0]:.2f}, {top_medium[0][1]:.2f}, {np.degrees(top_medium[0][2]):.1f}°) score={top_medium[0][3]:.4f}")

        # =====================================================================
        # LEVEL 3: Fine local optimization from best candidates
        # =====================================================================
        final_candidates = []
        for (mx, my, mt, _, _) in top_medium:
            # Local optimization using gradient descent
            opt_x, opt_y, opt_theta, opt_score = self._local_pose_optimization(
                all_points, mx, my, mt, width_t, length_t, max_iters=30
            )
            final_candidates.append((opt_x, opt_y, opt_theta, opt_score))

        # Select best final candidate
        final_candidates.sort(key=lambda x: x[3])
        best = final_candidates[0]

        print(f"[Level 3 - Fine] Optimized {len(final_candidates)} candidates")
        print(f"[Level 3 - Fine] Best: ({best[0]:.2f}, {best[1]:.2f}, {np.degrees(best[2]):.1f}°) score={best[3]:.4f}")

        return best[0], best[1], best[2], best[3]

    def _evaluate_pose_candidate(self,
                                  points: torch.Tensor,
                                  x: float,
                                  y: float,
                                  theta: float,
                                  width: torch.Tensor,
                                  length: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate a pose candidate using SDF and inlier ratio.

        Args:
            points: LIDAR points [N, 2]
            x, y, theta: Candidate pose
            width, length: Room dimensions

        Returns:
            (mean_sdf_error, inlier_ratio)
        """
        x_t = torch.tensor(x, dtype=DTYPE, device=DEVICE)
        y_t = torch.tensor(y, dtype=DTYPE, device=DEVICE)
        theta_t = torch.tensor(theta, dtype=DTYPE, device=DEVICE)

        with torch.no_grad():
            sdf_values = self.sdf_rect(points, x_t, y_t, theta_t, width, length)
            mean_sdf = torch.mean(sdf_values).item()

            # Inlier ratio: points within 15cm of walls
            inliers = (sdf_values < 0.15).float()
            inlier_ratio = torch.mean(inliers).item()

        return mean_sdf, inlier_ratio

    def _local_pose_optimization(self,
                                  points: torch.Tensor,
                                  init_x: float,
                                  init_y: float,
                                  init_theta: float,
                                  width: torch.Tensor,
                                  length: torch.Tensor,
                                  max_iters: int = 30) -> Tuple[float, float, float, float]:
        """
        Local gradient-based optimization of pose.

        Uses Adam optimizer with early stopping.

        Args:
            points: LIDAR points [N, 2]
            init_x, init_y, init_theta: Initial pose
            width, length: Room dimensions (fixed)
            max_iters: Maximum optimization iterations

        Returns:
            (opt_x, opt_y, opt_theta, final_score)
        """
        # Create optimizable pose parameters
        pose = torch.tensor([init_x, init_y, init_theta],
                           dtype=DTYPE, device=DEVICE, requires_grad=True)

        optimizer = torch.optim.Adam([pose], lr=0.05)

        prev_loss = float('inf')
        for _ in range(max_iters):
            optimizer.zero_grad()

            sdf_values = self.sdf_rect(points, pose[0], pose[1], pose[2], width, length)
            loss = torch.mean(sdf_values ** 2)

            loss_val = loss.item()
            if abs(prev_loss - loss_val) < 1e-6:
                break  # Converged
            prev_loss = loss_val

            loss.backward()
            optimizer.step()

            # Normalize theta
            with torch.no_grad():
                pose[2] = torch.atan2(torch.sin(pose[2]), torch.cos(pose[2]))

        with torch.no_grad():
            final_sdf = self.sdf_rect(points, pose[0], pose[1], pose[2], width, length)
            final_score = torch.mean(final_sdf).item()

        return pose[0].item(), pose[1].item(), pose[2].item(), final_score

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
        # We need GOOD SDF error to converge, not just iteration count
        room_error = (abs(self.belief.width - self.true_width) +
                      abs(self.belief.length - self.true_height)) / 2  # For reporting only

        # Compute pose uncertainty for convergence check
        pose_uncertainty = self.belief.uncertainty()

        # STRICT convergence criteria:
        # 1. SDF error must be below threshold (good fit)
        # 2. Pose uncertainty should be reasonable
        sdf_good = mean_sdf_error < self.init_convergence_threshold  # < 0.15m
        uncertainty_ok = pose_uncertainty < 0.1  # Reasonable uncertainty

        # Only converge if quality is good enough
        converged = sdf_good and uncertainty_ok

        # If we've hit max iterations but haven't converged, try harder
        if self.init_iterations >= self.max_init_iterations and not converged:
            if mean_sdf_error > 0.2 and self._init_restart_count < 3:  # Very poor fit, allow up to 3 restarts
                print(f"\n[ROOM ESTIMATION] WARNING: Max iterations reached with poor fit!")
                print(f"  SDF error: {mean_sdf_error:.4f}m (threshold: {self.init_convergence_threshold}m)")
                print(f"  Uncertainty: {pose_uncertainty:.4f}")
                print(f"  Restart {self._init_restart_count + 1}/3: Restarting hierarchical search...")

                # Reset and try again with fresh search
                self.init_iterations = 0
                self._hierarchical_search_done = False
                self._init_restart_count += 1
                return {
                    'phase': 'init',
                    'status': 'restarting',
                    'iteration': self.init_iterations,
                    'sdf_error': mean_sdf_error,
                    'pose': self.belief.pose.copy(),
                    'room_params': self.belief.room_params.copy(),
                    'room_error': room_error,
                    'belief_uncertainty': pose_uncertainty
                }
            else:
                # Mediocre fit or max restarts reached - accept with warning
                print(f"\n[ROOM ESTIMATION] WARNING: Converging with suboptimal fit")
                print(f"  SDF error: {mean_sdf_error:.4f}m (threshold {self.init_convergence_threshold}m)")
                print(f"  Restarts used: {self._init_restart_count}")
                converged = True  # Accept anyway after max iterations/restarts

        if converged:
            self.phase = 'tracking'
            self.belief.converged = True

            # Reduce room uncertainty (room is now fixed)
            self.belief.cov[3, 3] = 0.01
            self.belief.cov[4, 4] = 0.01

            # Store last known pose for prediction
            self._last_pose = self.belief.pose[:2].copy()
            self._last_theta = self.belief.theta

            # Reset tracking step counter for early exit feature
            self._tracking_step_count = 0

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

        Implements PREDICTION-BASED EARLY EXIT: if the motion model prediction
        already has low SDF error, we trust it and skip optimization entirely.
        This saves significant CPU when the robot moves smoothly.
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
        # PREDICTION STEP: Motion model with velocity calibration
        # =====================================================================
        s_prev = np.array([self.belief.x, self.belief.y, self.belief.theta])
        Sigma_prev = self.belief.cov[:3, :3].copy()

        # Apply velocity calibration factor (learned from innovation)
        # v_calibrated = k * v_commanded
        calibrated_velocity = robot_velocity * self.velocity_scale_factor
        calibrated_angular = angular_velocity * self.velocity_scale_factor

        # Transform velocity from robot frame to world frame
        cos_theta = np.cos(s_prev[2])
        sin_theta = np.sin(s_prev[2])
        vx_world = calibrated_velocity[0] * cos_theta - calibrated_velocity[1] * sin_theta
        vy_world = calibrated_velocity[0] * sin_theta + calibrated_velocity[1] * cos_theta

        # Store RAW commanded motion for calibration (without factor)
        # This is what we COMMANDED, innovation tells us the error
        # commanded_displacement = v_raw * dt (in world frame)
        raw_vx_world = robot_velocity[0] * cos_theta - robot_velocity[1] * sin_theta
        raw_vy_world = robot_velocity[0] * sin_theta + robot_velocity[1] * cos_theta
        commanded_displacement = np.array([
            raw_vx_world * dt,
            raw_vy_world * dt,
            angular_velocity * dt
        ])

        # Predicted state from motion model
        s_pred = np.array([
            s_prev[0] + vx_world * dt,
            s_prev[1] + vy_world * dt,
            s_prev[2] + calibrated_angular * dt
        ])
        s_pred[2] = np.arctan2(np.sin(s_pred[2]), np.cos(s_pred[2]))

        # Jacobian of motion model
        F_t = np.eye(3)
        F_t[0, 2] = -vy_world * dt
        F_t[1, 2] = vx_world * dt

        # Process noise (increases with motion, but bounded)
        # Q_scale ranges from 0.1 (stationary) to ~2.0 (fast motion)
        # This reflects that faster motion has more odometry uncertainty
        speed = np.linalg.norm(robot_velocity)
        Q_scale = max(0.1, min(2.0, speed / 0.3))  # Capped scaling, slower growth
        Q = self.Q_pose * dt * Q_scale

        # Predicted covariance
        Sigma_pred = F_t @ Sigma_prev @ F_t.T + Q

        # =====================================================================
        # PREDICTION-BASED EARLY EXIT
        # If the predicted pose already has low SDF error, skip optimization.
        # This implements "trusting the prior" when it's already good.
        # =====================================================================

        # Cache room dimensions as tensors
        if not hasattr(self, '_cached_width_t') or self._cached_width != self.belief.width:
            self._cached_width = self.belief.width
            self._cached_width_t = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
        if not hasattr(self, '_cached_length_t') or self._cached_length != self.belief.length:
            self._cached_length = self.belief.length
            self._cached_length_t = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)
        width = self._cached_width_t
        length = self._cached_length_t

        # Evaluate SDF at predicted pose (no gradients needed)
        with torch.no_grad():
            s_pred_t = torch.tensor(s_pred.astype(np.float32), dtype=DTYPE, device=DEVICE)
            sdf_pred = self.sdf_rect(lidar_points, s_pred_t[0], s_pred_t[1],
                                     s_pred_t[2], width, length)
            mean_sdf_pred = torch.mean(sdf_pred).item()

        # Early exit threshold: if prediction SDF is very good, trust it
        # Use a configurable factor of sigma_sdf (default 0.5 → ~7.5cm)
        prediction_trust_threshold = self.sigma_sdf * self.prediction_trust_factor

        # Increment tracking step counter
        self._tracking_step_count += 1

        # Compute current pose uncertainty (trace of position covariance)
        pose_uncertainty = np.trace(Sigma_pred[:2, :2])  # Only x, y uncertainty
        max_uncertainty_for_early_exit = 0.1  # If uncertainty > 0.1, force optimization

        # Only allow early exit after room estimation has stabilized
        # AND uncertainty is below threshold (to prevent unbounded covariance growth)
        early_exit_allowed = (self.prediction_early_exit and
                              self.adaptive_cpu and
                              self._tracking_step_count > self.min_tracking_steps_for_early_exit and
                              pose_uncertainty < max_uncertainty_for_early_exit)

        if early_exit_allowed and mean_sdf_pred < prediction_trust_threshold:
            # Prediction is good enough - skip optimization entirely
            self.belief.x = s_pred[0]
            self.belief.y = s_pred[1]
            self.belief.theta = s_pred[2]
            self.belief.cov[:3, :3] = Sigma_pred

            self._last_sdf_error = mean_sdf_pred
            self.stats['iterations_used'].append(0)  # No iterations used
            self.stats['sdf_error_history'].append(mean_sdf_pred)

            # Track early exits for statistics
            if 'prediction_early_exits' not in self.stats:
                self.stats['prediction_early_exits'] = 0
            self.stats['prediction_early_exits'] += 1

            # Compute Free Energy at predicted pose (for UI consistency)
            f_likelihood_early = mean_sdf_pred ** 2 / (2 * self.sigma_sdf ** 2)

            return {
                'phase': 'tracking',
                'pose': self.belief.pose.copy(),
                'pose_cov': self.belief.pose_cov,
                'sdf_error': mean_sdf_pred,
                'belief_uncertainty': self.belief.uncertainty(),
                'innovation': np.zeros(3),  # No correction applied
                'prior_precision': 1.0,  # Perfect trust in prediction
                'f_likelihood': f_likelihood_early,  # Accuracy term
                'f_prior': 0.0,                      # No deviation from prior
                'vfe': f_likelihood_early,           # Total VFE
                'iterations_used': 0,
                'early_exit': True
            }

        # =====================================================================
        # CORRECTION STEP: SDF optimization with prior
        # Free Energy F = F_likelihood + F_prior
        # F_likelihood = sum(SDF²) / (2 * σ_sdf²)
        # F_prior = 0.5 * (s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)
        # =====================================================================

        # Compute velocity-adaptive weights for [x, y, theta]
        velocity_weights = self._compute_velocity_adaptive_weights(robot_velocity, angular_velocity)

        # OPTIMIZED: Use float32 numpy array for faster tensor conversion
        velocity_weights_f32 = velocity_weights.astype(np.float32)
        velocity_weights_t = torch.from_numpy(velocity_weights_f32).to(device=DEVICE)

        # Apply velocity weights to prediction covariance
        weight_scale = np.diag(1.0 / (velocity_weights + 1e-6))
        Sigma_pred_weighted = weight_scale @ Sigma_pred @ weight_scale

        # Initialize at PREDICTED pose
        s_pred_f32 = s_pred.astype(np.float32)
        pose_params = torch.tensor(s_pred_f32, dtype=DTYPE, device=DEVICE, requires_grad=True)
        s_pred_t = torch.from_numpy(s_pred_f32).to(device=DEVICE)

        # Prior information (inverse covariance) - pre-compute as float32
        Sigma_inv_np = np.linalg.inv(Sigma_pred_weighted + 1e-6 * np.eye(3)).astype(np.float32)
        Sigma_pred_inv = torch.from_numpy(Sigma_inv_np).to(device=DEVICE)

        # Use class learning rate parameter (was hardcoded lr=0.05)
        optimizer = torch.optim.Adam([pose_params], lr=self.lr_tracking)

        # Early stopping variables for tracking
        prev_loss = float('inf')
        iterations_used = 0
        max_iters = self.max_iterations_tracking if self.adaptive_cpu else 50

        # Pre-compute constants outside loop
        check_convergence = self.adaptive_cpu
        min_iters = self.min_iterations
        early_threshold = self.early_stop_threshold

        # Likelihood precision (inverse variance of SDF observations)
        # See main.tex Eq. obstacle_objective
        sigma_sdf_sq = self.sigma_sdf ** 2

        for iteration in range(max_iters):
            optimizer.zero_grad()

            # ============================================================
            # FREE ENERGY (main.tex Eq. obstacle_objective):
            # F(s) = (1/2σ²) Σᵢ dᵢ(s)² + (1/2)(s-μ_p)ᵀ Σ_p⁻¹ (s-μ_p)
            #      = F_likelihood + F_prior
            # The balance comes from σ² (sensor noise) and Σ_p (prior cov)
            # ============================================================

            # Likelihood: SDF error normalized by sensor variance
            # F_likelihood = (1/2σ²) × Σ dᵢ²  (summed, not averaged, for proper scaling)
            sdf_values = self.sdf_rect(lidar_points, pose_params[0], pose_params[1],
                                       pose_params[2], width, length)
            # Use sum for proper likelihood, divide by N for numerical stability
            N_pts = len(lidar_points)
            F_likelihood = torch.sum(sdf_values ** 2) / (2.0 * sigma_sdf_sq * N_pts)

            # Prior: deviation from motion model prediction
            # F_prior = (1/2)(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)
            pose_diff_x = pose_params[0] - s_pred_t[0]
            pose_diff_y = pose_params[1] - s_pred_t[1]
            pose_diff_theta = torch.atan2(torch.sin(pose_params[2] - s_pred_t[2]),
                                          torch.cos(pose_params[2] - s_pred_t[2]))

            # Compute F_prior (manual expansion for 3x3 to avoid tensor creation)
            F_prior = 0.5 * (
                pose_diff_x * (Sigma_pred_inv[0, 0] * pose_diff_x + Sigma_pred_inv[0, 1] * pose_diff_y + Sigma_pred_inv[0, 2] * pose_diff_theta) +
                pose_diff_y * (Sigma_pred_inv[1, 0] * pose_diff_x + Sigma_pred_inv[1, 1] * pose_diff_y + Sigma_pred_inv[1, 2] * pose_diff_theta) +
                pose_diff_theta * (Sigma_pred_inv[2, 0] * pose_diff_x + Sigma_pred_inv[2, 1] * pose_diff_y + Sigma_pred_inv[2, 2] * pose_diff_theta)
            )

            # Total Free Energy (no additional weighting - balance from σ² and Σ)
            F = F_likelihood + F_prior

            iterations_used = iteration + 1

            # Early stopping check (after minimum iterations)
            # OPTIMIZED: only call .item() when checking convergence
            if check_convergence and iteration >= min_iters:
                loss_val = F.item()
                if prev_loss - loss_val < early_threshold:
                    break  # Converged
                prev_loss = loss_val

            F.backward()

            # Apply velocity-adaptive gradient weighting
            with torch.no_grad():
                if pose_params.grad is not None:
                    pose_params.grad *= velocity_weights_t

            optimizer.step()

            # Normalize theta to [-pi, pi]
            with torch.no_grad():
                pose_params[2] = torch.atan2(torch.sin(pose_params[2]), torch.cos(pose_params[2]))

        # Track iterations used
        self.stats['iterations_used'].append(iterations_used)

        optimized_pose = np.array([pose_params[0].item(), pose_params[1].item(), pose_params[2].item()])

        # =====================================================================
        # INNOVATION AND DIAGNOSTIC PRECISION (for monitoring only)
        # The balance between likelihood and prior comes from:
        #   - sigma_sdf² (sensor noise variance)
        #   - Sigma_pred (motion model prediction covariance)
        # This diagnostic precision shows how well the motion model predicted
        # =====================================================================
        innovation = optimized_pose - s_pred
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))

        # =====================================================================
        # VELOCITY AUTO-CALIBRATION
        # If motion model consistently under/over-predicts, learn correction
        # Innovation tells us: actual_displacement - predicted_displacement
        # If innovation > 0 consistently → we're under-predicting → k should increase
        # =====================================================================
        if self.velocity_calibration_enabled:
            self._update_velocity_calibration(innovation, commanded_displacement)

        # Mahalanobis distance of innovation (diagnostic: how many sigmas off was prediction)
        innovation_mahal = np.sqrt(innovation @ np.linalg.inv(Sigma_pred + 1e-6 * np.eye(3)) @ innovation)
        innovation_mahal = np.sqrt(innovation @ np.linalg.inv(Sigma_pred + 1e-6 * np.eye(3)) @ innovation)

        # Diagnostic precision: indicates how well motion model predicted
        # High value = good prediction, Low value = poor prediction
        # This is reported to the UI but does NOT affect the optimization
        if not hasattr(self, '_adaptive_prior_precision'):
            self._adaptive_prior_precision = 1.0

        # Scale Mahalanobis to a 0-1 range for display (exp decay)
        # innovation_mahal ~ 0 → precision ~ 1.0 (good prediction)
        # innovation_mahal ~ 3 → precision ~ 0.05 (poor prediction, 3σ outlier)
        self._adaptive_prior_precision = np.exp(-innovation_mahal / 2.0)

        # =====================================================================
        # POSTERIOR COVARIANCE via Hessian of the cost function
        # Formula: Σ = σ² × H⁻¹ where σ² is the residual variance
        # =====================================================================
        N_obs = len(lidar_points)

        # OPTIMIZED: Reuse optimized pose tensor, enable grad temporarily
        pose_opt = torch.tensor(optimized_pose.astype(np.float32), dtype=DTYPE, device=DEVICE, requires_grad=True)
        sdf_opt = self.sdf_rect(lidar_points, pose_opt[0], pose_opt[1], pose_opt[2], width, length)

        # Residual variance and loss
        sdf_sq = sdf_opt ** 2
        residual_var = torch.mean(sdf_sq).item()
        loss = torch.mean(sdf_sq)

        # Compute gradient
        grad = torch.autograd.grad(loss, pose_opt, create_graph=True)[0]

        # OPTIMIZED: Compute Hessian rows more efficiently
        hessian = torch.stack([
            torch.autograd.grad(grad[i], pose_opt, retain_graph=(i < 2))[0]
            for i in range(3)
        ])
        hessian = 0.5 * (hessian + hessian.T)  # Ensure symmetry

        try:
            hessian_np = hessian.detach().cpu().numpy()

            # Robust inversion with progressive regularization
            reg = 1e-6
            H_inv = None
            for _ in range(5):
                try:
                    H_inv = np.linalg.inv(hessian_np + reg * np.eye(3))
                    # Verify it's positive definite
                    np.linalg.cholesky(H_inv + 1e-10 * np.eye(3))
                    break
                except np.linalg.LinAlgError:
                    reg *= 10.0

            if H_inv is None:
                H_inv = np.linalg.pinv(hessian_np + 1e-3 * np.eye(3))

            # Covariance = σ² × H⁻¹
            Sigma_post = residual_var * H_inv

        except Exception:
            Sigma_post = np.diag([0.01, 0.01, 0.01])

        # Compute final SDF error for diagnostics
        mean_sdf_error = torch.mean(torch.abs(sdf_opt)).detach().item()
        self._last_sdf_error = mean_sdf_error

        # =====================================================================
        # COMPUTE FINAL FREE ENERGY COMPONENTS (for visualization)
        # F = F_likelihood + F_prior (main.tex Eq. obstacle_objective)
        # =====================================================================
        sigma_sdf_sq = self.sigma_sdf ** 2
        N_pts = len(lidar_points)
        with torch.no_grad():
            # Final likelihood term (SDF fit, normalized)
            final_sdf = self.sdf_rect(lidar_points, pose_params[0], pose_params[1],
                                      pose_params[2], width, length)
            final_f_likelihood = (torch.sum(final_sdf ** 2) / (2.0 * sigma_sdf_sq * N_pts)).item()

            # Final prior term (deviation from motion model)
            final_pose_diff = pose_params - s_pred_t
            final_pose_diff[2] = torch.atan2(torch.sin(final_pose_diff[2]), torch.cos(final_pose_diff[2]))
            final_f_prior = (0.5 * final_pose_diff @ Sigma_pred_inv @ final_pose_diff).item()

            # Total VFE after optimization (no additional weighting)
            final_vfe = final_f_likelihood + final_f_prior

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
            'velocity_weights': velocity_weights,  # [w_x, w_y, w_theta] velocity-adaptive weights
            'velocity_scale': self.velocity_scale_factor,  # Calibrated velocity factor
            # Free Energy components
            'f_likelihood': final_f_likelihood,  # Accuracy term (SDF²)
            'f_prior': final_f_prior,            # Complexity term (motion model deviation)
            'vfe': final_vfe                     # Total Variational Free Energy
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
