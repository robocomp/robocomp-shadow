#!/usr/bin/env python3
"""Test script for PointcloudCenterEstimator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.pointcloud_center_estimator import PointcloudCenterEstimator, EstimatorConfig

def test_obb_estimation():
    """Test OBB estimation with simulated room data."""
    print("=" * 60)
    print("Testing PointcloudCenterEstimator (GT-free room estimation)")
    print("=" * 60)

    np.random.seed(42)

    # True room and robot parameters
    room_w, room_h = 6.0, 4.0
    robot_x, robot_y, robot_theta = 0.5, 0.3, 0.2

    # Generate wall points in room frame
    n_per_wall = 50
    left_wall = np.column_stack([np.full(n_per_wall, -room_w/2), np.linspace(-room_h/2, room_h/2, n_per_wall)])
    right_wall = np.column_stack([np.full(n_per_wall, room_w/2), np.linspace(-room_h/2, room_h/2, n_per_wall)])
    front_wall = np.column_stack([np.linspace(-room_w/2, room_w/2, n_per_wall), np.full(n_per_wall, room_h/2)])
    back_wall = np.column_stack([np.linspace(-room_w/2, room_w/2, n_per_wall), np.full(n_per_wall, -room_h/2)])

    room_points = np.vstack([left_wall, right_wall, front_wall, back_wall])

    # Add noise
    room_points += np.random.randn(*room_points.shape) * 0.05

    # Transform to robot frame
    cos_t, sin_t = np.cos(-robot_theta), np.sin(-robot_theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    robot_points = (R @ (room_points - np.array([robot_x, robot_y])).T).T

    print(f"\nInput: {len(robot_points)} points simulating {room_w}x{room_h}m room")
    print(f"True robot pose: x={robot_x:.3f}m, y={robot_y:.3f}m, theta={robot_theta:.3f}rad ({np.degrees(robot_theta):.1f}deg)")

    # Estimate
    estimator = PointcloudCenterEstimator()
    obb = estimator.estimate(robot_points)

    if obb:
        # Robot position = negative of OBB center
        est_x = -obb.center[0]
        est_y = -obb.center[1]
        est_theta = -obb.rotation

        pos_error = np.sqrt((est_x - robot_x)**2 + (est_y - robot_y)**2)
        angle_error = abs(est_theta - robot_theta)
        angle_error = min(angle_error, 2*np.pi - angle_error)  # Handle wrapping

        print(f"\n--- Results ---")
        print(f"Estimated pose:  x={est_x:.3f}m, y={est_y:.3f}m, theta={est_theta:.3f}rad ({np.degrees(est_theta):.1f}deg)")
        print(f"Position error:  {pos_error:.3f}m")
        print(f"Angle error:     {np.degrees(angle_error):.1f}deg")
        print(f"\nEstimated room:  {obb.width:.2f} x {obb.height:.2f}m")
        print(f"True room:       {room_w:.2f} x {room_h:.2f}m")
        print(f"Room size error: width={abs(obb.width - room_w):.2f}m, height={abs(obb.height - room_h):.2f}m")

        # Pass/fail criteria
        if pos_error < 0.2 and np.degrees(angle_error) < 10:
            print(f"\n✓ TEST PASSED: GT-free estimation working!")
            return True
        else:
            print(f"\n✗ TEST FAILED: Errors too large")
            return False
    else:
        print(f"\n✗ TEST FAILED: OBB estimation returned None")
        return False


if __name__ == "__main__":
    success = test_obb_estimation()
    sys.exit(0 if success else 1)
