# Session Notes - 2026-02-06

## Current State

### What's Working
1. **Chair initialization** - Tests 8 angles (every 45°) and picks the one with lowest SDF
2. **Likelihood-only optimization** - SDF-based optimization
3. **Historical points** - Slow accumulation with warmup period
4. **Gradient handling** - Angle gradient multiplier (100x) with cap (0.5 rad)
5. **Visualization** - Open3D visualizer showing chair, points, historical points
6. **STATE PRIOR REINTRODUCED** - Frame mismatch fixed!

### Latest Results (2026-02-06)
```
Position:     Error=2.5cm   ✅ Excellent
SeatWidth:    Error=-0.9cm  ✅ Excellent  
SeatDepth:    Error=+0.1cm  ✅ Perfect
SeatHeight:   Error=+6.0cm  ⚠️ Acceptable (LiDAR coverage issue?)
BackHeight:   Error=+4.7cm  ⚠️ Acceptable
Angle:        Error=-90.6°  ❌ Initialization issue (known)
SDF mean:     0.023         ✅ Very low
Hist pts:     N=219
```

### Prior Implementation (2026-02-06)
The prior is now active with lambdas as soft regularizer:

```python
# In compute_prior_term() for all beliefs (box, table, chair):
lambda_pos = 0.05     # Position regularization
lambda_size = 0.02    # Size regularization
lambda_angle = 0.0    # Angle: disabled (causes 90° flips)
```

**Frame transformation fix:**
- `mu` (being optimized) is in **robot frame**
- `self.mu` (previous state) is in **room frame**
- Solution: Use `transform_object_to_robot_frame(self.mu, robot_pose)` to compare in same frame

### Key Files Modified
- `src/objects/chair/belief.py` - Prior enabled with frame transform
- `src/objects/table/belief.py` - Prior enabled with frame transform
- `src/objects/box/belief.py` - Prior enabled with frame transform
- `src/belief_manager.py` - Already passes `robot_pose` to `compute_prior_term()`

### Architecture
```
belief_manager.py
  └── _optimize_belief()
        └── belief.compute_prior_term(mu, robot_pose)
              └── transform_object_to_robot_frame(self.mu, robot_pose)
              └── Compare mu vs mu_prev_robot (both in robot frame)
```

### Key Parameters
```python
# In belief_manager.py
self.optimization_iters = 10
self.optimization_lr = 0.05
self.grad_clip = 2.0
angle_lr_multiplier = 100.0   # For chair only
max_angle_grad = 0.5          # Cap for chair angle

# In each belief's compute_prior_term():
lambda_pos = 0.05
lambda_size = 0.02
lambda_angle = 0.01  # Enabled (very weak)
```

### Math Reference (from ACTIVE_INFERENCE_MATH.md)
- VFE = Likelihood + Prior
- Likelihood = (1/2σ²) × Σ SDF(p_i, s)²
- Prior = (λ/2) × ||s - s_prev||² (for static objects)

### Known Issues
- **90° angle error** - Initialization sometimes picks wrong angle due to chair symmetry
- Possible fix: Use backrest detection more aggressively in initialization

### Next Steps
1. ✅ Prior working with frame transform
2. ✅ Lambda values tuned (0.05/0.02)
3. Fix angle initialization (backrest detection)
4. Consider angle prior once initialization is fixed

### Test Configuration
- GT for chair: seat 0.45x0.45, height 0.45m, angle 0°
- Chair position: (0, 0) in room frame
