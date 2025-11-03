# regional_loss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
import math
import numpy as np
import torch

Side = Literal["+x", "-x", "+y", "-y"]

@dataclass
class SegmentScore:
    side: Side
    index: int
    s0: float
    s1: float
    center: Tuple[float, float]
    n_points: int
    loss: float            # mean Huber(|SDF|) on present points
    mean_abs_sdf: float
    expected_pts: float    # expected number of points (based on density)
    support_ratio: float   # n_points / expected_pts
    total_loss: float      # blended (SDF + absence) loss


class RegionalizedRectLoss:
    """
    Regionalized loss for a single rectangle particle.
    Divides each wall into N uniform segments and computes
    a per-segment loss combining:
      - local SDF discrepancy (Huber(|SDF|))
      - absence penalty if few or no points hit the segment band
    """

    def __init__(
        self,
        num_segments_per_side: int = 16,
        band_outside: float = 0.40,
        band_inside: float = 0.25,
        huber_delta: float = 0.03,
        sdf_margin: float = -0.05,
        device: str = "cpu",
        absence_alpha: float = 1.0,      # weight for absence penalty
        absence_curve_k: float = 4.0,    # shape parameter for logistic shortage curve
    ):
        assert num_segments_per_side >= 1
        self.N = int(num_segments_per_side)
        self.band_outside = float(band_outside)
        self.band_inside = float(band_inside)
        self.delta = float(huber_delta)
        self.device = device
        self.absence_alpha = float(absence_alpha)
        self.absence_curve_k = float(absence_curve_k)

    # ---------- helpers ----------
    @staticmethod
    def _room_transform(points_world: torch.Tensor, x: float, y: float, theta: float, device: str) -> torch.Tensor:
        """Transform Nx2 world points into the particle's local frame."""
        c, s = math.cos(-theta), math.sin(-theta)
        R = torch.tensor([[c, -s], [s,  c]], dtype=torch.float32, device=device)
        t = torch.tensor([x, y], dtype=torch.float32, device=device)
        return (points_world - t) @ R.T

    @staticmethod
    def _sdf_rect(points_room: torch.Tensor, L, W) -> torch.Tensor:
        """Signed distance to an axis-aligned rectangle centered at origin."""
        if isinstance(L, torch.Tensor):
            half = torch.stack([L / 2.0, W / 2.0])
        else:
            half = torch.tensor([L / 2.0, W / 2.0], device=points_room.device)
        q = torch.abs(points_room) - half
        outside = torch.clamp(q, min=0.0)
        dist_out = outside.norm(dim=1)
        dist_in = torch.clamp(torch.max(q[:, 0], q[:, 1]), max=0.0)
        return dist_out + dist_in  # negative inside

    def _huber(self, abs_r: torch.Tensor) -> torch.Tensor:
        """Huber loss on absolute residual."""
        quad = 0.5 * abs_r ** 2
        lin = self.delta * (abs_r - 0.5 * self.delta)
        return torch.where(abs_r <= self.delta, quad, lin)

    def _segment_params(self, L: float, W: float) -> Dict[Side, Tuple[np.ndarray, float]]:
        segs_W = np.linspace(0.0, W, self.N + 1)
        segs_L = np.linspace(0.0, L, self.N + 1)
        return {
            "+x": (segs_W, W),
            "-x": (segs_W, W),
            "+y": (segs_L, L),
            "-y": (segs_L, L),
        }

    def _segment_center_xy(self, side: Side, s0: float, s1: float, L: float, W: float) -> Tuple[float, float]:
        """Center of a segment in local rectangle coordinates."""
        if side == "+x":
            x = +L / 2.0
            y = -W / 2.0 + 0.5 * (s0 + s1)
        elif side == "-x":
            x = -L / 2.0
            y = -W / 2.0 + 0.5 * (s0 + s1)
        elif side == "+y":
            x = -L / 2.0 + 0.5 * (s0 + s1)
            y = +W / 2.0
        else:  # "-y"
            x = -L / 2.0 + 0.5 * (s0 + s1)
            y = -W / 2.0
        return (x, y)

    def _make_band_masks_and_t(self, points_room: torch.Tensor, L: float, W: float) -> Dict[Side, Tuple[torch.Tensor, torch.Tensor]]:
        """Build masks selecting points near each wall and tangential coordinates."""
        x = points_room[:, 0]
        y = points_room[:, 1]
        band_in = self.band_inside
        band_out = self.band_outside

        mask_pos_x = (x >=  L/2 - band_in) & (x <=  L/2 + band_out)
        mask_neg_x = (x <= -L/2 + band_in) & (x >= -L/2 - band_out)
        mask_pos_y = (y >=  W/2 - band_in) & (y <=  W/2 + band_out)
        mask_neg_y = (y <= -W/2 + band_in) & (y >= -W/2 - band_out)

        t_pos_x = y - (-W/2)
        t_neg_x = y - (-W/2)
        t_pos_y = x - (-L/2)
        t_neg_y = x - (-L/2)

        return {
            "+x": (mask_pos_x, t_pos_x),
            "-x": (mask_neg_x, t_neg_x),
            "+y": (mask_pos_y, t_pos_y),
            "-y": (mask_neg_y, t_neg_y),
        }

    # ---------- main ----------
    def evaluate_segments(
        self,
        particle,
        wall_points_world: torch.Tensor,
    ) -> Tuple[List[SegmentScore], Dict[Side, np.ndarray]]:
        """Compute per-segment loss including absence penalty."""
        assert wall_points_world.ndim == 2 and wall_points_world.shape[1] == 2

        # Transform to room coordinates
        pts_room = self._room_transform(
            wall_points_world, particle.x, particle.y, particle.theta, self.device
        )

        L = float(particle.length)
        W = float(particle.width)

        sdf = self._sdf_rect(pts_room, L, W)
        abs_sdf = torch.abs(sdf)

        side_masks_and_t = self._make_band_masks_and_t(pts_room, L, W)

        # ---- estimate global point density in wall bands ----
        band_width = self.band_inside + self.band_outside
        total_band_area = 2 * L * band_width + 2 * W * band_width
        union_mask = None
        for m, _ in side_masks_and_t.values():
            union_mask = m if union_mask is None else (union_mask | m)
        total_band_points = int(torch.count_nonzero(union_mask).item())
        density = (total_band_points / max(total_band_area, 1e-6)) if total_band_points > 0 else 0.0

        seg_edges = self._segment_params(L, W)
        results: List[SegmentScore] = []
        heat: Dict[Side, np.ndarray] = {}

        # ---- loop over sides ----
        for side in ["+x", "-x", "+y", "-y"]:
            edges, total_len = seg_edges[side]
            step = total_len / self.N
            mask, t = side_masks_and_t[side]
            t_clamped = torch.clamp(t, 0.0, total_len)
            idx = torch.nonzero(mask, as_tuple=False).flatten()

            side_loss = np.zeros(self.N, dtype=np.float32)
            side_count = np.zeros(self.N, dtype=np.int32)
            side_mean_abs = np.zeros(self.N, dtype=np.float32)

            if idx.numel() > 0:
                t_sel = t_clamped[idx]
                abs_sdf_sel = abs_sdf[idx]
                bin_idx = torch.clamp((t_sel / max(step, 1e-6)).long(), 0, self.N - 1)
                penalties = self._huber(abs_sdf_sel)

                for b in range(self.N):
                    b_mask = (bin_idx == b)
                    if torch.any(b_mask):
                        pen_b = penalties[b_mask]
                        asdf_b = abs_sdf_sel[b_mask]
                        side_loss[b] = float(pen_b.mean().item())
                        side_mean_abs[b] = float(asdf_b.mean().item())
                        side_count[b] = int(b_mask.sum().item())

            # expected points per segment (based on global density)
            seg_area = step * (self.band_inside + self.band_outside)
            expected_pts = float(density * seg_area)

            def shortage(n_obs: float, n_exp: float) -> float:
                """Logistic shortage penalty."""
                if n_exp <= 1e-6:
                    return 0.0
                ratio = n_obs / (n_exp + 1e-6)
                return 1.0 / (1.0 + math.exp(self.absence_curve_k * (ratio - 0.5)))

            side_total_loss = np.zeros(self.N, dtype=np.float32)

            for i in range(self.N):
                s0, s1 = float(edges[i]), float(edges[i+1])
                cx, cy = self._segment_center_xy(side, s0, s1, L, W)
                n_obs = float(side_count[i])
                n_exp = expected_pts
                shortage_pen = self.absence_alpha * shortage(n_obs, n_exp)
                total_loss = float(side_loss[i]) + shortage_pen

                results.append(SegmentScore(
                    side=side,
                    index=i,
                    s0=s0,
                    s1=s1,
                    center=(cx, cy),
                    n_points=int(n_obs),
                    loss=float(side_loss[i]),
                    mean_abs_sdf=float(side_mean_abs[i]),
                    expected_pts=float(n_exp),
                    support_ratio=(n_obs / (n_exp + 1e-6) if n_exp > 0 else 1.0),
                    total_loss=total_loss,
                ))
                side_total_loss[i] = total_loss

            heat[side] = side_total_loss

        return results, heat

    def evaluate_from_np(
        self,
        particle,
        wall_points_world_np: np.ndarray,
    ) -> Tuple[List[SegmentScore], Dict[Side, np.ndarray]]:
        """Numpy wrapper."""
        pw = torch.from_numpy(wall_points_world_np).float().to(self.device)
        return self.evaluate_segments(particle, pw)
