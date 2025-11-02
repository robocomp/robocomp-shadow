# open3d_hotzones.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import math
import numpy as np
import open3d as o3d

# ----------------- small helpers -----------------
def _room_to_world_xy(x: float, y: float, px: float, py: float, th: float) -> Tuple[float, float]:
    c, s = math.cos(th), math.sin(th)
    X = c * x - s * y + px
    Y = s * x + c * y + py
    return X, Y

def _heat_range(values: np.ndarray, percentile_clip: Optional[float]) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    if percentile_clip is None:
        lo, hi = float(values.min()), float(values.max())
    else:
        lo = float(np.percentile(values, 100 - percentile_clip)) if percentile_clip > 50 else float(values.min())
        hi = float(np.percentile(values, percentile_clip))
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1e-6
    return lo, hi

def _heat_to_rgb(v: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
    """Blue → Cyan → Yellow → Red."""
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    if t < 1/3:
        u = t * 3
        r, g, b = 0.0, u, 1.0
    elif t < 2/3:
        u = (t - 1/3) * 3
        r, g, b = u, 1.0, 1.0 - u
    else:
        u = (t - 2/3) * 3
        r, g, b = 1.0, 1.0 - u, 0.0
    return float(r), float(g), float(b)

# ----------------- geometry builders -----------------
def _segment_patch_room(
    side: str,
    s0: float,
    s1: float,
    L: float,
    W: float,
    thickness: float,
    outward_eps: float,    # kept for API compatibility
    *,
    height: float = 2.2,
    lift: float = 0.02,
    # snapping controls:
    snap_mode: str = "plane",   # "plane" | "inside" | "outside" | "thick"
    eps_in: float = 1e-4,       # meters, offset toward inside
    eps_out: float = 1e-4,      # meters, offset toward outside
    # legacy "thick" mode (ignored unless snap_mode=="thick")
    inside_frac: float = 0.05,
    gap_in: float = 0.001,
    gap_out: float = 0.001,
) -> List[Tuple[float, float, float]]:
    """
    Build a vertical slab aligned with the nominal wall plane.

    snap_mode:
      - "plane": faces straddle the exact plane by ±eps (best to look 'touching').
      - "inside": both faces shifted inside by eps_in (visible but never outside).
      - "outside": both faces shifted outside by eps_out.
      - "thick": legacy thick slab (uses inside_frac + gaps).

    Notes:
      Inside directions: +x wall inside is -x; -x wall inside is +x;
                         +y wall inside is -y; -y wall inside is +y.
    """
    z0, z1 = lift, lift + height

    def y_span():
        return -W/2 + s0, -W/2 + s1

    def x_span():
        return -L/2 + s0, -L/2 + s1

    if snap_mode == "thick":
        # legacy behavior (your existing parameters)
        inside_frac = max(0.0, min(1.0, float(inside_frac)))
        t_in = thickness * inside_frac
        t_out = thickness * (1.0 - inside_frac)

    if side == "+x":
        x_plane =  L/2.0
        y0, y1 = y_span()
        if snap_mode == "plane":
            x_in, x_out = x_plane - eps_in, x_plane + eps_out
        elif snap_mode == "inside":
            x_in = x_plane - eps_in
            x_out = x_in - 1e-6   # degenerate thin slab
        elif snap_mode == "outside":
            x_out = x_plane + eps_out
            x_in  = x_out + 1e-6
        else:  # thick
            x_in  = x_plane - gap_in - t_in
            x_out = x_plane + gap_out + t_out
        v = [(x_in,y0,z0),(x_in,y1,z0),(x_in,y0,z1),(x_in,y1,z1),
             (x_out,y0,z0),(x_out,y1,z0),(x_out,y0,z1),(x_out,y1,z1)]

    elif side == "-x":
        x_plane = -L/2.0
        y0, y1 = y_span()
        if snap_mode == "plane":
            x_in, x_out = x_plane + eps_in, x_plane - eps_out
        elif snap_mode == "inside":
            x_in = x_plane + eps_in
            x_out = x_in + 1e-6
        elif snap_mode == "outside":
            x_out = x_plane - eps_out
            x_in  = x_out - 1e-6
        else:
            x_in  = x_plane + gap_in + t_in
            x_out = x_plane - gap_out - t_out
        v = [(x_in,y0,z0),(x_in,y1,z0),(x_in,y0,z1),(x_in,y1,z1),
             (x_out,y0,z0),(x_out,y1,z0),(x_out,y0,z1),(x_out,y1,z1)]

    elif side == "+y":
        # wall plane at y = +W/2, inside of the room is toward -y
        y_plane = W / 2.0
        if snap_mode == "plane":
            # straddle the wall plane by ±eps, but push toward -y (inside)
            y_in = y_plane - eps_out  # slightly inside
            y_out = y_plane - eps_in  # even more inside
        elif snap_mode == "inside":
            y_in = y_plane - eps_in
            y_out = y_in - 1e-6
        elif snap_mode == "outside":
            y_out = y_plane + eps_out
            y_in = y_out + 1e-6
        else:
            y_in = y_plane - gap_in - thickness
            y_out = y_plane - gap_out
        x0, x1 = -L / 2 + s0, -L / 2 + s1
        v = [
            (x0, y_in, z0), (x1, y_in, z0), (x0, y_in, z1), (x1, y_in, z1),
            (x0, y_out, z0), (x1, y_out, z0), (x0, y_out, z1), (x1, y_out, z1),
        ]

    else:  # "-y"
        # wall plane at y = -W/2, inside of the room is toward +y
        y_plane = -W / 2.0
        if snap_mode == "plane":
            # straddle the wall plane by ±eps, but push toward +y (inside)
            y_in = y_plane + eps_out
            y_out = y_plane + eps_in
        elif snap_mode == "inside":
            y_in = y_plane + eps_in
            y_out = y_in + 1e-6
        elif snap_mode == "outside":
            y_out = y_plane - eps_out
            y_in = y_out - 1e-6
        else:
            y_in = y_plane + gap_in + thickness
            y_out = y_plane + gap_out
        x0, x1 = -L / 2 + s0, -L / 2 + s1
        v = [
            (x0, y_in, z0), (x1, y_in, z0), (x0, y_in, z1), (x1, y_in, z1),
            (x0, y_out, z0), (x1, y_out, z0), (x0, y_out, z1), (x1, y_out, z1),
        ]

    return v



def _make_quad_mesh_world(
    verts_room: List[Tuple[float, float, float]],
    px: float, py: float, th: float,
    color: Tuple[float, float, float]
) -> o3d.geometry.TriangleMesh:
    """
    verts_room: 8 vertices (two parallel quads). We create two vertical faces.
    """
    verts_world = []
    for (x, y, z) in verts_room:
        X, Y = _room_to_world_xy(x, y, px, py, th)
        verts_world.append([X, Y, z])

    tris = np.array([
        [0, 1, 2], [2, 1, 3],   # inner face
        [4, 5, 6], [6, 5, 7],   # outer face
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(verts_world, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

# ----------------- main API -----------------
def build_hot_patches_heatmap(
    seg_list: Iterable,
    particle,
    *,
    # selection
    threshold: Optional[float] = None,
    topk: Optional[int] = None,
    min_norm: float = 0.85,
    min_points: int = 0,
    # color scaling
    percentile_clip: Optional[float] = 90.0,
    vmin_override: Optional[float] = None,
    vmax_override: Optional[float] = None,
    # geometry
    thickness: float = 0.06,
    outward_eps: float = 0.0,
    height: float = 2.2,
    lift: float = 0.02,
    inside_frac: float = 0.05,
    gap_in: float = 0.001,
    gap_out: float = 0.001,
    # NEW:
    snap_mode: str = "plane",   # "plane" | "inside" | "outside" | "thick"
    eps_in: float = 1e-4,
    eps_out: float = 1e-4,
) -> List[o3d.geometry.TriangleMesh]:
    """
    Build colored overlay patches ONLY for hot segments and place them hugging the wall plane.
    Colors are mapped by (total_loss if present, else loss) using a blue→red heat map.
    """
    L, W = float(particle.length), float(particle.width)
    px, py, th = float(particle.x), float(particle.y), float(particle.theta)

    # collect scores
    items = []
    for s in seg_list:
        score = getattr(s, "total_loss", getattr(s, "loss", 0.0))
        items.append((s, float(score)))
    if not items:
        return []

    # color range on the pool BEFORE normalized filtering
    all_scores = np.array([sc for _, sc in items], dtype=np.float32)
    vmin, vmax = _heat_range(all_scores, percentile_clip)
    if vmin_override is not None: vmin = vmin_override
    if vmax_override is not None: vmax = vmax_override
    denom = max(vmax - vmin, 1e-9)

    # 1) absolute threshold
    if threshold is not None:
        items = [it for it in items if it[1] >= threshold]

    # 2) top-K
    if topk is not None and topk > 0:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:topk]

    # 3) normalized filter to hide blues
    filtered: List[tuple] = []
    for s, sc in items:
        if (s.n_points < min_points) and (min_points > 0):
            continue
        norm = (sc - vmin) / denom
        if norm >= min_norm:
            filtered.append((s, sc))
    items = filtered
    if not items:
        return []

    # build meshes
    meshes: List[o3d.geometry.TriangleMesh] = []
    for s, sc in items:
        color = _heat_to_rgb(sc, vmin, vmax)
        verts_room = _segment_patch_room(
            s.side, s.s0, s.s1, L, W, thickness, outward_eps,
            height=height, lift=lift,
            inside_frac=inside_frac, gap_in=gap_in, gap_out=gap_out,
            snap_mode=snap_mode, eps_in=eps_in, eps_out=eps_out,
        )
        mesh = _make_quad_mesh_world(verts_room, px, py, th, color)
        meshes.append(mesh)
    return meshes

# Convenience wrappers
def add_hot_patches_to_visualizer(
    vis: o3d.visualization.Visualizer,
    seg_list: Iterable,
    particle,
    **kwargs,
) -> List[o3d.geometry.TriangleMesh]:
    meshes = build_hot_patches_heatmap(seg_list, particle, **kwargs)
    for m in meshes:
        vis.add_geometry(m, reset_bounding_box=False)
    return meshes

def draw_hot_patches_once(seg_list: Iterable, particle, **kwargs):
    meshes = build_hot_patches_heatmap(seg_list, particle, **kwargs)
    if not meshes:
        print("No hot patches to draw.")
        return
    o3d.visualization.draw_geometries(meshes)
