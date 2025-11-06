# params.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

# ---- 2D Pose + Size ----
@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float
    length: float
    width: float

# ---- Robot params ----
@dataclass(frozen=True)
class Robot:
    width: float = 0.5
    length: float = 0.5
    radius: float = 0.35
    max_vel: float = 1.0          # m/s
    max_angular_vel: float = 1.0  # rad/s
    loss_vel_restricion: float = 200.0  # multiplicative loss factor for velocity loss

# ---- Plane detector ----
@dataclass(frozen=True)
class PlaneParams:
    voxel_size: float = 0.05
    angle_tolerance_deg: float = 10.0
    ransac_threshold: float = 0.05     # to accept inliers
    ransac_n: int = 3
    ransac_iterations: int = 1000
    min_plane_points: int = 100
    nms_normal_dot_threshold: float = 0.99
    nms_distance_threshold: float = 0.10
    plane_thickness: float = 0.01

# ---- Initial room hypothesis (also used to build Particle) ----
@dataclass(frozen=True)
class RoomHypothesis:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    length: float = 5.0
    width: float = 9.0
    height: float = 2.5
    z_center: float = 1.25
    weight: float = 1.0

# ---- Particle Filter (PF) high-level knobs ----
@dataclass(frozen=True)
class PFParams:
    num_particles: int = 100
    device: str = "cuda"
    use_gradient_refinement: bool = True
    adaptive_particles: bool = False
    min_particles: int = 20
    max_particles: int = 300
    elite_count: int = 1
    min_loss_for_locking_ransac: float = 0.01
    # Optional tuning we set on the instance after construction
    trans_noise: float = 0.01
    rot_noise: float = 0.01
    trans_noise_stationary: float = 0.0001
    rot_noise_stationary: float = 0.0001
    ess_frac: float = 0.3
    # Gradient refinement
    lr: float = 0.02
    num_steps: int = 15
    top_n: int = 3
    pose_lambda: float = 1e-2
    size_lambda: float = 1e-3

# ---- Regionalized loss (segmentation) ----
@dataclass(frozen=True)
class RegionalLossParams:
    num_segments_per_side: int = 16
    band_outside: float = 0.40
    band_inside: float = 0.40
    huber_delta: float = 0.05
    device: str = "cuda"
    absence_alpha: float = 1.0
    absence_curve_k: float = 4.0

# ---- Hotzone overlays (Open3D patches) ----
@dataclass(frozen=True)
class HotzonesParams:
    percentile_clip: float = 50.0
    min_norm: float = 0.95
    topk: int = 10
    eps_in: float = 0.001
    eps_out: float = 0.001
    snap_mode: str = "plane"     # "plane" | "inside" | "outside" | "thick"
    thickness: float = 0.001
    min_support_ratio: float = 0.2
    # slab vertical look
    height: float = 2.2
    lift: float = 0.02
    inside_frac: float = 0.05
    gap_in: float = 0.001
    gap_out: float = 0.001

# ---- Visualizer & runtime timing ----
@dataclass(frozen=True)
class VizParams:
    window_size: Tuple[int, int] = (600, 600)
    view_front: Tuple[float, float, float] = (5.0, -5.0, 20.0)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 1.5)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    zoom: float = 0.01
    floor_thickness: float = 0.02
    # RGB triplets for walls/planes
    h_color: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    v_colors: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
    )
    o_color: Tuple[float, float, float] = (1.0, 0.0, 1.0)

@dataclass(frozen=True)
class TimingParams:
    period_ms: int = 100
    vel_alpha: float = 0.8
    latency_gain: float = 1.2  # multiply cycle time to integrate commands

# ---- App-level container bundling all sections ----
@dataclass(frozen=True)
class AppParams:
    plane: PlaneParams = PlaneParams()
    hypothesis: RoomHypothesis = RoomHypothesis()
    pf: PFParams = PFParams()
    regional: RegionalLossParams = RegionalLossParams()
    hotzones: HotzonesParams = HotzonesParams()
    viz: VizParams = VizParams()
    timing: TimingParams = TimingParams()
    robot: Robot = Robot()
