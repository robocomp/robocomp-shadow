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

    Coordinate convention: x=lateral (left/right), y=forward (front/back), θ=rotation

    State vector: s = (x, y, θ, W, L)
    - (x, y, θ): robot pose in room frame (in METERS)
      * x: lateral position (left/right)
      * y: forward position (front/back)
      * θ: heading angle (radians)
    - (W, L): room width and length (in METERS)
      * W: width in x direction (lateral extent)
      * L: length in y direction (forward extent)

    Belief is Gaussian: q(s) = N(μ, Σ)
    """
    # Mean (in meters)
    x: float = 0.0  # lateral position
    y: float = 0.0  # forward position
    theta: float = 0.0  # heading angle
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
        self.init_convergence_threshold = 0.05  # 5cm threshold - more realistic
        self.max_init_iterations = 500
        self.init_iterations = 0

        # Tracking parameters - adjusted for meters
        # sigma_sdf represents the effective uncertainty in wall distance measurements
        # This includes sensor noise + dynamic effects (motion, rotation, timing)
        # Needs to be large enough that prior (motion model) remains influential
        self.sigma_sdf = 0.15  # 15cm - accounts for dynamic uncertainty during motion
        self.Q_pose = np.diag([0.01, 0.01, 0.05])  # Process noise in meters

        # Optimization parameters
        self.lr_init = 0.5  # Higher learning rate for faster convergence
        self.lr_tracking = 0.1  # Balanced for convergence with theta constraint


        # Statistics
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': []
        }


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
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Transform points: p_room = R @ p_robot + [x, y]
        px = points[:, 0]
        py = points[:, 1]

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

        # =====================================================================
        # STEP 1: Compute raw spreads in robot frame first
        # This gives us the apparent dimensions without rotation correction
        # =====================================================================
        raw_x_spread = np.max(points_np[:, 0]) - np.min(points_np[:, 0])
        raw_y_spread = np.max(points_np[:, 1]) - np.min(points_np[:, 1])

        print(f"[INIT] Raw LIDAR spread: x={raw_x_spread:.2f}m, y={raw_y_spread:.2f}m")

        # =====================================================================
        # STEP 2: Use PCA to find principal axes
        # =====================================================================
        centroid = np.mean(points_np, axis=0)
        centered = points_np - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Major axis (direction of largest variance)
        major_axis = eigenvectors[:, 0]

        # Angle of major axis: arctan2(x_component, y_component)
        # This is angle from y-axis (forward) to the major axis
        pca_angle = np.arctan2(major_axis[0], major_axis[1])

        print(f"[INIT] PCA major axis angle: {np.degrees(pca_angle):.1f}° from y-axis")

        # =====================================================================
        # STEP 3: Estimate theta using prior from GT
        # =====================================================================
        prior_theta = robot_pose_gt[2] if robot_pose_gt is not None else 0.0
        prior_theta_std = 0.3  # Trust GT more
        prior_theta_var = prior_theta_std ** 2
        lidar_theta_std = 0.5
        lidar_theta_var = lidar_theta_std ** 2

        theta_var = (prior_theta_var * lidar_theta_var) / (prior_theta_var + lidar_theta_var)

        # Weighted circular mean
        w_prior = lidar_theta_var / (prior_theta_var + lidar_theta_var)
        w_lidar = prior_theta_var / (prior_theta_var + lidar_theta_var)

        sin_theta = w_prior * np.sin(prior_theta) + w_lidar * np.sin(pca_angle)
        cos_theta = w_prior * np.cos(prior_theta) + w_lidar * np.cos(pca_angle)
        est_theta = np.arctan2(sin_theta, cos_theta)

        # =====================================================================
        # STEP 4: Compute dimensions in room-aligned frame
        # =====================================================================
        cos_t = np.cos(-est_theta)
        sin_t = np.sin(-est_theta)
        aligned_x = cos_t * points_np[:, 0] - sin_t * points_np[:, 1]
        aligned_y = sin_t * points_np[:, 0] + cos_t * points_np[:, 1]

        max_x, min_x = np.max(aligned_x), np.min(aligned_x)
        max_y, min_y = np.max(aligned_y), np.min(aligned_y)

        lidar_dim_x = max_x - min_x + 0.2  # margin for noise
        lidar_dim_y = max_y - min_y + 0.2

        # =====================================================================
        # STEP 5: Assign dimensions ensuring width >= length
        # Prior: width=6m (larger), length=4m (smaller)
        # =====================================================================
        prior_width = 6.0
        prior_length = 4.0
        prior_dim_std = 1.0
        prior_dim_var = prior_dim_std ** 2
        lidar_dim_std = 1.0
        lidar_dim_var = lidar_dim_std ** 2

        # Determine which LIDAR dimension is likely width (larger) vs length (smaller)
        if lidar_dim_x >= lidar_dim_y:
            # x is the larger dimension -> x = width, y = length
            lidar_width = lidar_dim_x
            lidar_length = lidar_dim_y
            # No rotation adjustment needed
            theta_adjustment = 0.0
        else:
            # y is the larger dimension -> need to swap
            # This means the room is rotated 90° from our initial estimate
            lidar_width = lidar_dim_y
            lidar_length = lidar_dim_x
            # Adjust theta by 90° to align width with x-axis
            theta_adjustment = np.pi / 2

        print(f"[INIT] Aligned dims: x={lidar_dim_x:.2f}m, y={lidar_dim_y:.2f}m")
        print(f"[INIT] Assigned: width={lidar_width:.2f}m, length={lidar_length:.2f}m")
        if abs(theta_adjustment) > 0.1:
            print(f"[INIT] Theta adjustment: {np.degrees(theta_adjustment):.1f}°")
            est_theta += theta_adjustment
            # Normalize theta to [-pi, pi]
            while est_theta > np.pi:
                est_theta -= 2 * np.pi
            while est_theta < -np.pi:
                est_theta += 2 * np.pi

        # Bayesian fusion with priors
        width_var = (prior_dim_var * lidar_dim_var) / (prior_dim_var + lidar_dim_var)
        length_var = (prior_dim_var * lidar_dim_var) / (prior_dim_var + lidar_dim_var)

        est_width = (lidar_dim_var * prior_width + prior_dim_var * lidar_width) / (prior_dim_var + lidar_dim_var)
        est_length = (lidar_dim_var * prior_length + prior_dim_var * lidar_length) / (prior_dim_var + lidar_dim_var)

        # Clamp to reasonable bounds
        est_width = np.clip(est_width, 3.0, 12.0)
        est_length = np.clip(est_length, 2.0, 8.0)

        # =====================================================================
        # STEP 6: Estimate robot position
        # =====================================================================
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0

        # Transform back to world frame using final theta
        cos_t_inv = np.cos(est_theta)
        sin_t_inv = np.sin(est_theta)
        est_pos_x = cos_t_inv * center_x - sin_t_inv * center_y
        est_pos_y = sin_t_inv * center_x + cos_t_inv * center_y

        # Fuse with GT prior for position
        if robot_pose_gt is not None:
            pos_prior_var = 0.5  # Trust GT more
            pos_lidar_var = 2.0
            est_pos_x = (pos_lidar_var * robot_pose_gt[0] + pos_prior_var * est_pos_x) / (pos_prior_var + pos_lidar_var)
            est_pos_y = (pos_lidar_var * robot_pose_gt[1] + pos_prior_var * est_pos_y) / (pos_prior_var + pos_lidar_var)

        # Debug output
        print(f"[INIT] Prior theta: {np.degrees(prior_theta):.1f}°")
        print(f"[INIT] Final theta: {np.degrees(est_theta):.1f}° (σ={np.degrees(np.sqrt(theta_var)):.1f}°)")
        print(f"[INIT] Prior dims: {prior_width:.2f}m x {prior_length:.2f}m")
        print(f"[INIT] Final dims: {est_width:.2f}m x {est_length:.2f}m (σ={np.sqrt(width_var):.2f}m)")
        print(f"[INIT] Estimated position: ({est_pos_x:.2f}, {est_pos_y:.2f})m")

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

            print(f"[INIT] Initial belief: room {self.belief.width:.2f}x{self.belief.length:.2f}m, "
                  f"pose [{self.belief.x:.2f}, {self.belief.y:.2f}, θ={np.degrees(self.belief.theta):.1f}°]")

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

        for iteration in range(100):  # More iterations for better convergence
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

            # Debug print occasionally
            if iteration % 50 == 0 and self.init_iterations < 5:
                print(f"  [OPT iter {iteration}] SDF loss: {sdf_loss.item():.4f}, total: {loss.item():.4f}")

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
                    print(f"[INIT] Swapping dimensions: {opt_width:.2f}x{opt_length:.2f} -> {opt_length:.2f}x{opt_width:.2f}")
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

        Paper Section 4.25 - Resulting Free Energy Objective:

        F[q(s), o] = F_likelihood + F_prior

        Where:
        - F_likelihood = Σ SDF(o_i, s)² / (2σ_sdf²)  : Observation error
        - F_prior = (1/2)(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred) : Prediction error

        s_pred = s_{t-1} + v * dt  (propagate previous pose with velocity)
        Σ_pred = Σ_{t-1} + Q        (process noise added)
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
        # PREDICTION STEP: Propagate previous posterior with motion model
        # Using differential drive kinematics (dead reckoning)
        # =====================================================================

        # Previous posterior (from last iteration)
        s_prev = np.array([self.belief.x, self.belief.y, self.belief.theta])
        Sigma_prev = self.belief.cov[:3, :3].copy()

        # Differential drive motion model:
        # x_new = x + vx * cos(theta) * dt - vy * sin(theta) * dt
        # y_new = y + vx * sin(theta) * dt + vy * cos(theta) * dt
        # theta_new = theta + omega * dt
        cos_theta = np.cos(s_prev[2])
        sin_theta = np.sin(s_prev[2])

        # Transform velocity from robot frame to world frame
        vx_world = robot_velocity[0] * cos_theta - robot_velocity[1] * sin_theta
        vy_world = robot_velocity[0] * sin_theta + robot_velocity[1] * cos_theta

        s_pred = np.array([
            s_prev[0] + vx_world * dt,
            s_prev[1] + vy_world * dt,
            s_prev[2] + angular_velocity * dt
        ])

        # Normalize predicted theta to [-π, π]
        s_pred[2] = np.arctan2(np.sin(s_pred[2]), np.cos(s_pred[2]))

        # Debug: print prediction occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        if self._debug_counter % 20 == 0:
            print(f"  [Tracking] Prediction: theta_prev={s_prev[2]:.3f}, omega={angular_velocity:.3f}, "
                  f"dt={dt:.3f}, theta_pred={s_pred[2]:.3f}")

        # Process noise proportional to velocity (more movement = more uncertainty)
        speed = np.linalg.norm(robot_velocity)
        velocity_noise_factor = max(0.1, speed / 0.5)
        Q_dynamic = self.Q_pose * dt * velocity_noise_factor

        # Add extra noise for rotation
        Q_dynamic[2, 2] += abs(angular_velocity) * dt * 0.1

        Sigma_pred = Sigma_prev + Q_dynamic

        # =====================================================================
        # CORRECTION STEP: Minimize F = F_likelihood + F_prior
        # Paper Section 4.25: Both terms have equal weight (factor 1/2 in each)
        # =====================================================================

        # Fixed room parameters
        width = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
        length = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)

        # Optimizable pose - initialize at predicted position
        pose_params = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE, requires_grad=True)

        # Prior tensors
        s_pred_t = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE)
        Sigma_pred_inv = torch.tensor(np.linalg.inv(Sigma_pred), dtype=DTYPE, device=DEVICE)

        # Optimization
        optimizer = torch.optim.Adam([pose_params], lr=self.lr_tracking)

        for iteration in range(15):  # More iterations for better convergence
            optimizer.zero_grad()

            # =================================================================
            # F_likelihood: -ln p(o|s) = (1/(2*sigma_w^2)) * ||d||^2
            # As per paper equation in Section 4.25 (Resulting Free Energy Objective)
            #
            # PRACTICAL CONSIDERATION: With M~4000 observations, the sum scales
            # linearly with M, giving observation precision ~ M/σ². For the prior
            # to remain influential, we use MEAN squared error which effectively
            # treats the observations as providing aggregate information rather than
            # M independent pieces of evidence. This is standard in batch optimization.
            # =================================================================
            sdf_values = self.sdf_rect(
                lidar_points,
                pose_params[0], pose_params[1], pose_params[2],
                width, length
            )
            N_obs = len(lidar_points)
            F_likelihood = torch.mean(sdf_values ** 2) / (2 * self.sigma_sdf ** 2)

            # =================================================================
            # F_prior: (1/2)(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)
            # As per paper equation in Section 4.25
            # This term keeps the estimate close to the motion-model prediction
            # =================================================================
            pose_diff = pose_params - s_pred_t
            F_prior = 0.5 * pose_diff @ Sigma_pred_inv @ pose_diff

            # Total Free Energy
            F = F_likelihood + F_prior
            F.backward()

            optimizer.step()

            # Normalize theta to [-π, π]
            with torch.no_grad():
                pose_params[2] = torch.atan2(torch.sin(pose_params[2]),
                                              torch.cos(pose_params[2]))

        # =====================================================================
        # UPDATE BELIEF with optimized posterior
        # =====================================================================
        with torch.no_grad():
            self.belief.x = pose_params[0].item()
            self.belief.y = pose_params[1].item()
            self.belief.theta = pose_params[2].item()

            # Debug: print correction
            if self._debug_counter % 20 == 0:
                theta_change = self.belief.theta - s_pred[2]
                print(f"  [Tracking] Correction: theta_pred={s_pred[2]:.3f}, "
                      f"theta_corrected={self.belief.theta:.3f}, change={theta_change:.3f} ({np.degrees(theta_change):.1f}°)")

            # Compute final SDF error for diagnostics
            sdf_values = self.sdf_rect(
                lidar_points,
                pose_params[0], pose_params[1], pose_params[2],
                width, length
            )
            mean_sdf_error = torch.mean(torch.abs(sdf_values)).item()

            # =====================================================================
            # COVARIANCE UPDATE (EKF-inspired Information Filter)
            #
            # Σ_post^{-1} = Σ_pred^{-1} + H^T R^{-1} H
            #
            # Simplified version:
            # - Σ_pred already includes process noise (grew during prediction)
            # - Observation reduces uncertainty proportional to number of good points
            # - Poor observations (high SDF) provide less information
            # =====================================================================

            # Number of "good" observations (SDF close to 0)
            good_obs_mask = torch.abs(sdf_values) < 0.1  # Points within 10cm of wall
            num_good_obs = good_obs_mask.sum().item()
            total_obs = len(sdf_values)

            # Information gain from observations
            R_effective = self.sigma_sdf ** 2 * (1.0 + mean_sdf_error / 0.1)

            # Observation information matrix (simplified diagonal)
            info_scale = 0.02
            if num_good_obs > 0:
                obs_info = info_scale * (num_good_obs / total_obs) / R_effective
            else:
                obs_info = 0.0

            # Information filter update
            try:
                Sigma_pred_inv_np = np.linalg.inv(Sigma_pred)
                Sigma_post_inv = Sigma_pred_inv_np + obs_info * np.eye(3)
                Sigma_post = np.linalg.inv(Sigma_post_inv)
            except np.linalg.LinAlgError:
                Sigma_post = Sigma_pred * 0.95

            self.belief.cov[:3, :3] = Sigma_post

            # Bounds on covariance diagonal
            for i in range(3):
                self.belief.cov[i, i] = np.clip(self.belief.cov[i, i], 0.01, 10.0)

        self.stats['sdf_error_history'].append(mean_sdf_error)

        return {
            'phase': 'tracking',
            'pose': self.belief.pose.copy(),
            'pose_cov': self.belief.pose_cov,
            'sdf_error': mean_sdf_error,
            'room_params': self.belief.room_params.copy(),
            'belief_uncertainty': self.belief.uncertainty()
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
