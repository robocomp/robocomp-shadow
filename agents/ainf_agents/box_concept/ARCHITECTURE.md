# Object Inference Architecture

## File Structure

```
object_concept/
├── bin/
│   └── object_concept           # Executable entry point
│
├── src/
│   ├── specificworker.py        # Main compute loop, DSR integration
│   ├── belief_core.py           # Base Belief class, shared constants
│   ├── belief_manager.py        # Base BeliefManager (clustering, association, optimization)
│   ├── model_selector.py        # Bayesian Model Selection (MultiModelBelief)
│   ├── multi_model_manager.py   # Manager using BMS for multiple object types
│   ├── transforms.py            # Coordinate frame transformations
│   ├── object_sdf_prior.py      # Re-exports SDFs (compatibility layer)
│   ├── visualizer_3d.py         # 3D visualizer (Open3D) - standalone mode
│   ├── qt3d_viewer.py           # Qt3D visualizer (integrates with DSRViewer)
│   ├── dsr_gui.py               # DSRViewer for PySide6 (graph view + custom tabs)
│   │
│   └── cpp/                     # C++ components (reference implementations)
│       ├── dsr_gui.py           # Original DSRViewer (PySide2)
│       ├── qt3d_visualizer.cpp  # C++ Qt3D reference implementation
│       └── qt3d_visualizer.h
│   │
│   └── objects/                 # Object-specific implementations
│       ├── __init__.py          # OBJECT_REGISTRY, get_sdf_function()
│       │
│       ├── box/
│       │   ├── __init__.py
│       │   ├── sdf.py           # compute_box_sdf(), compute_box_priors()
│       │   ├── belief.py        # BoxBelief, BoxBeliefConfig
│       │   └── manager.py       # BoxManager
│       │
│       ├── table/
│       │   ├── __init__.py
│       │   ├── sdf.py           # compute_table_sdf() (box + 4 cylinders)
│       │   ├── belief.py        # TableBelief, TableBeliefConfig
│       │   └── manager.py       # TableManager
│       │
│       ├── chair/
│       │   ├── __init__.py
│       │   ├── sdf.py           # compute_chair_sdf() (seat + backrest)
│       │   ├── belief.py        # ChairBelief, ChairBeliefConfig
│       │   └── manager.py       # ChairManager
│       │
│       └── cylinder/
│           ├── __init__.py
│           └── sdf.py           # compute_cylinder_sdf()
│
├── OBJECT_INFERENCE_MATH.md     # Mathematical formulation
├── SESSION_NOTES.md             # Development notes
└── ARCHITECTURE.md              # This file
```

---

## Call Flow

### Single Model Mode (OBJECT_MODEL = 'table' | 'chair' | 'box')

```
bin/object_concept
    │
    └── specificworker.py::SpecificWorker
            │
            ├── __init__()
            │   ├── ObjectManager(g, agent_id)     # TableManager, ChairManager, or BoxManager
            │   └── ObjectConceptVisualizer()      # 3D visualizer (Open3D)
            │
            └── compute()  [called every Period ms]
                │
                ├── get_lidar_points()             # From Lidar3D proxy
                ├── get_room_dimensions()          # From DSR graph (room node)
                ├── get_robot_pose_and_cov()       # From DSR graph (shadow node)
                │
                └── object_manager.update(lidar_points, robot_pose, robot_cov, room_dims)
                        │
                        │  [BeliefManager.update() in belief_manager.py]
                        │
                        ├── transform_points_robot_to_room()
                        ├── _filter_wall_points()
                        ├── _cluster_points()              # DBSCAN
                        ├── _associate_clusters()          # Hungarian algorithm
                        │
                        ├── _update_matched_beliefs()
                        │   │
                        │   └── For each matched (cluster, belief):
                        │       ├── belief.get_historical_points_in_robot_frame()
                        │       ├── _optimize_belief()     # Gradient descent on VFE
                        │       │   │
                        │       │   └── For each iteration:
                        │       │       ├── belief.sdf(points)           # Object-specific SDF
                        │       │       ├── belief.compute_prior_term()  # State regularization
                        │       │       └── loss.backward() + gradient step
                        │       │
                        │       ├── belief.add_historical_points()
                        │       └── belief.update_rfe()    # Update TCE for historical points
                        │
                        ├── _create_new_beliefs()          # From unmatched clusters
                        │   └── Belief.from_cluster()      # PCA initialization
                        │
                        └── _decay_unmatched_beliefs()     # Confidence decay
```

---

### Multi-Model Mode (OBJECT_MODEL = 'multi')

```
bin/object_concept
    │
    └── specificworker.py::SpecificWorker
            │
            ├── __init__()
            │   ├── MultiModelManager(model_classes={...})
            │   │       │
            │   │       └── ModelSelector(model_classes, config)
            │   │
            │   └── ObjectConceptVisualizer()
            │
            └── compute()
                │
                └── multi_model_manager.update(lidar_points, robot_pose, robot_cov, room_dims)
                        │
                        ├── [Same as BeliefManager: transform, filter, cluster, associate]
                        │
                        ├── _update_matched_beliefs()
                        │   │
                        │   └── For each matched (cluster, multi_belief):
                        │       │
                        │       └── For each hypothesis (table, chair):
                        │           ├── _optimize_hypothesis()  → F_m (VFE)
                        │           ├── selector.update_hypothesis_vfe(multi_belief, model, F_m)
                        │           ├── belief.add_historical_points()
                        │           └── belief.update_rfe()
                        │       
                        │       └── multi_belief.update_posterior()
                        │           │
                        │           └── q(m) = softmax(-F_m) · p(m)
                        │
                        ├── _create_new_beliefs()
                        │   └── selector.create_multi_belief(cluster)
                        │       │
                        │       └── Creates TableBelief + ChairBelief in parallel
                        │
                        ├── _decay_unmatched_beliefs()
                        │
                        └── _process_commitments()
                            │
                            └── For each multi_belief:
                                └── selector.process_commitment()
                                    │
                                    └── If (H[q(m)] < threshold) AND (hysteresis_ok):
                                        └── multi_belief.commit(winner)
                                            └── Discard losing model
```

---

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `Belief` | belief_core.py | Base class: state (μ, Σ), historical points, lifecycle |
| `BoxBelief` | objects/box/belief.py | Box-specific: 6-param state, box SDF |
| `TableBelief` | objects/table/belief.py | Table-specific: 7-param state, table SDF |
| `ChairBelief` | objects/chair/belief.py | Chair-specific: 7-param state, chair SDF |
| `BeliefManager` | belief_manager.py | Single-model perception: cluster → associate → optimize |
| `MultiModelBelief` | model_selector.py | Parallel hypotheses + q(m) posterior |
| `ModelSelector` | model_selector.py | Factory + commitment logic |
| `MultiModelManager` | multi_model_manager.py | BMS-enabled perception manager |

---

## Data Flow

```
LIDAR (robot frame)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Transform to room frame                                    │
│  p_room = R(θ) · p_robot + [rx, ry]                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Filter wall points                                         │
│  Keep points > wall_margin from room boundaries            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DBSCAN clustering                                          │
│  eps=0.15-0.30m, min_pts=15                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Hungarian association (cluster ↔ belief)                  │
│  Cost = center_distance + mean(SDF²)                       │
└─────────────────────────────────────────────────────────────┘
    │
    ├── Matched → Update belief via VFE optimization
    ├── Unmatched cluster → Create new belief
    └── Unmatched belief → Decay confidence
```

---

## VFE Optimization (per belief)

```
F = F_present + F_past + F_prior

F_present = (1/N) Σ SDF(p_i, s)²           # Current LIDAR points
F_past    = Σ w_j · SDF(p_j, s)²           # Historical points (weighted by TCE)
F_prior   = λ_pos·||Δc||² + λ_size·||Δd||² + λ_θ·Δθ²   # State regularization

Gradient descent:
    s ← s - η · ∇F(s)
    
For 10 iterations with η=0.05, grad_clip=2.0
```

---

## Configuration (specificworker.py)

```python
OBJECT_MODEL = 'table'           # 'box', 'table', 'chair', or 'multi'

GT_CONFIG = {                    # Ground truth for debug
    'box':   {...},
    'table': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_w': 0.7, 'gt_h': 0.9, ...},
    'chair': {...},
}
```
