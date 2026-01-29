#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import traceback
import numpy as np
from src.concept_room import RoomPoseEstimatorV2

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *
import torch



class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)
    
        if startup_check:
            self.startup_check()
        else:
            # Initialize total step counter
            self.total_steps = 0

            self.robot_pose_gt = [0.0, 0.0, 0.0]  # Ground truth robot pose [x, y, theta]
            self.last_cmd = [0.0, 0.0, 0.0]  # Last command issued to robot (advx, advz, rot)

            # Create room estimator - room is 6m x 4m
            self.room_estimator = RoomPoseEstimatorV2(
                room_width=6.0,   # width in x direction (lateral) - 6m
                room_height=4.0,  # length in y direction (forward) - 4m
                use_known_room_dimensions=False  # Estimate room dimensions
            )

            # Statistics tracking
            self.pose_errors = []
            self.sdf_errors = []
            self.stats_printed = False

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""


    @QtCore.Slot()
    def compute(self):
        """Main computation loop - called periodically"""

        # Get lidar data (comes in mm, convert to meters)
        lidar_points = self.get_lidar_points()

        # Execute exploration movement and get the command issued
        self.last_cmd = self.explore_room()  # Returns (advx, advz, rot)

        # Update room estimator using ONLY velocity commands (dead reckoning)
        # Ground truth is NOT passed to the estimator - only used for error reporting
        self.update_room_estimator(self.last_cmd, lidar_points)

        self.total_steps += 1
        return True

    def update_room_estimator(self, last_cmd, lidar_points):
        """Update the room estimator with current sensor data

        Uses Active Inference approach:
        - Phase INIT: Uses robot_pose_gt for initial approximation, estimates room dimensions
        - Phase TRACKING: Propagate pose with velocity (dead reckoning), correct with LIDAR

        Args:
            last_cmd: Last command issued (advx, advz, rot) in mm/s and rad/s
            lidar_points: LIDAR points in meters
        """
        if len(lidar_points) > 0:
            # Convert last command to velocity in m/s for the estimator
            # last_cmd = (advx, advz, rot) where advx=lateral, advz=forward in mm/s
            velocity = np.array([last_cmd[0] / 1000.0, last_cmd[1] / 1000.0])  # [vx, vy] in m/s
            angular_velocity = last_cmd[2]  # rad/s

            # Time step in seconds
            dt = self.Period / 1000.0

            # Update room estimator
            # Pass robot_pose_gt for initial approximation during INIT phase
            result = self.room_estimator.update(
                robot_pose_gt=self.robot_pose_gt,
                robot_velocity=velocity,
                angular_velocity=angular_velocity,
                dt=dt,
                lidar_points_real=lidar_points
            )

            # Print status every 20 steps
            if self.total_steps % 20 == 0:
                phase = result.get('phase', result.get('status', 'unknown'))
                sdf_err = result.get('sdf_error', 0)
                uncertainty = result.get('belief_uncertainty', 0)

                # Get estimated pose from belief
                est_x = self.room_estimator.belief.x if self.room_estimator.belief else 0
                est_y = self.room_estimator.belief.y if self.room_estimator.belief else 0
                est_theta = self.room_estimator.belief.theta if self.room_estimator.belief else 0

                # Compute error against ground truth (for evaluation only)
                gt_x, gt_y, gt_theta = self.robot_pose_gt
                pose_error = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)

                # Track errors for statistics
                self.pose_errors.append(pose_error)
                self.sdf_errors.append(sdf_err)

                print(f"[Step {self.total_steps}] Phase: {phase}, SDF: {sdf_err:.3f}, "
                      f"Pose_err: {pose_error:.3f}m, Uncertainty: {uncertainty:.3f}")

                # Ground truth (from simulator)
                print(f"  Ground truth: x={gt_x:.3f}m, y={gt_y:.3f}m, theta={gt_theta:.3f}rad")

                # Estimated pose (from Active Inference)
                print(f"  Estimated:    x={est_x:.3f}m, y={est_y:.3f}m, theta={est_theta:.3f}rad")

                # Command being issued
                print(f"  Command: vx={velocity[0]:.3f}m/s, vy={velocity[1]:.3f}m/s, rot={angular_velocity:.3f}rad/s")

                # Room estimation
                if self.room_estimator.belief:
                    print(f"  Estimated room: {self.room_estimator.belief.width:.2f} x "
                          f"{self.room_estimator.belief.length:.2f} m (true: {self.room_estimator.true_width:.1f} x {self.room_estimator.true_height:.1f} m)")

                print(f"  LIDAR points: {len(lidar_points)}")

    def print_statistics_summary(self):
        """Print a summary of statistics at the end of the run"""
        if self.stats_printed:
            return
        self.stats_printed = True

        print("\n" + "="*60)
        print("           ACTIVE INFERENCE ROOM ESTIMATOR - SUMMARY")
        print("="*60)

        # Room estimation
        if self.room_estimator.belief:
            est_w = self.room_estimator.belief.width
            est_l = self.room_estimator.belief.length
            true_w = self.room_estimator.true_width
            true_l = self.room_estimator.true_height
            room_err = np.sqrt((est_w - true_w)**2 + (est_l - true_l)**2)

            print(f"\n📐 ROOM ESTIMATION:")
            print(f"   True room:      {true_w:.2f} x {true_l:.2f} m")
            print(f"   Estimated room: {est_w:.2f} x {est_l:.2f} m")
            print(f"   Room error:     {room_err:.3f} m")

        # Pose errors
        if len(self.pose_errors) > 0:
            pose_arr = np.array(self.pose_errors)
            print(f"\n📍 POSE ERROR (vs Ground Truth):")
            print(f"   Mean:    {np.mean(pose_arr):.3f} m")
            print(f"   Std:     {np.std(pose_arr):.3f} m")
            print(f"   Min:     {np.min(pose_arr):.3f} m")
            print(f"   Max:     {np.max(pose_arr):.3f} m")
            print(f"   Final:   {pose_arr[-1]:.3f} m")

        # SDF errors
        if len(self.sdf_errors) > 0:
            sdf_arr = np.array(self.sdf_errors)
            print(f"\n📊 SDF ERROR (LIDAR fit):")
            print(f"   Mean:    {np.mean(sdf_arr):.3f} m")
            print(f"   Std:     {np.std(sdf_arr):.3f} m")
            print(f"   Min:     {np.min(sdf_arr):.3f} m")
            print(f"   Max:     {np.max(sdf_arr):.3f} m")
            print(f"   Final:   {sdf_arr[-1]:.3f} m")

        # Final pose comparison
        if self.room_estimator.belief:
            gt_x, gt_y, gt_theta = self.robot_pose_gt
            est_x = self.room_estimator.belief.x
            est_y = self.room_estimator.belief.y
            est_theta = self.room_estimator.belief.theta

            print(f"\n🤖 FINAL POSE:")
            print(f"   Ground Truth: x={gt_x:.3f}m, y={gt_y:.3f}m, θ={gt_theta:.3f}rad")
            print(f"   Estimated:    x={est_x:.3f}m, y={est_y:.3f}m, θ={est_theta:.3f}rad")
            print(f"   Position err: {np.sqrt((est_x-gt_x)**2 + (est_y-gt_y)**2):.3f} m")

            # Angle error (handle wrapping)
            angle_err = abs(est_theta - gt_theta)
            if angle_err > np.pi:
                angle_err = 2*np.pi - angle_err
            print(f"   Angle error:  {np.degrees(angle_err):.1f}°")

        print(f"\n⏱️  Total steps: {self.total_steps}")
        print("="*60 + "\n")


    #####################################################################33
    # def get_robot_state(self):
    #     try:
    #         robot_state = self.omnirobot_proxy.getBaseState()
    #         # Robot coordinates: x=lateral (left/right), y=forward (front), z=up
    #         # Estimator expects: [x, y] in meters (same convention)
    #         robot_pos = np.array([robot_state.x / 1000.0, robot_state.z / 1000.0])  # [lateral, forward]
    #         robot_theta = robot_state.alpha  # radians
    #
    #         # Debug print first few times
    #         if self.total_steps < 3:
    #             print(f"[DEBUG] robot_state: x={robot_state.x:.0f}, y={robot_state.z:.0f}, alpha={robot_state.alpha:.3f}")
    #             print(f"[DEBUG] robot_pos (meters): [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}]")
    #     except Ice.Exception as e:
    #         print(f"Error reading robot state: {e}")
    #         return np.array([0.0, 0.0]), 0.0
    #     return robot_pos, robot_theta

    def get_lidar_points(self) -> np.ndarray:
        """Get LIDAR points projected to 2D horizontal plane

        Coordinate convention: x+ = left, y+ = forward
        """
        try:
            # Get 2D LIDAR data (getLidarDataWithThreshold2d projects to horizontal plane)
            # Parameters: sensor_name, max_distance_mm, decimation_factor
            helios = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 8000, 4)

            # LIDAR points: p.x = left, p.y = forward
            # Estimator expects: [x, y] where x = left, y = forward
            # So use [p.x, p.y] directly (no swap needed)
            lidar_points = np.array([[p.x / 1000.0, p.y / 1000.0] for p in helios.points])

            # Debug: print LIDAR stats periodically
            if self.total_steps % 100 == 0 and len(lidar_points) > 0:
                x_range = lidar_points[:,0].max() - lidar_points[:,0].min()
                y_range = lidar_points[:,1].max() - lidar_points[:,1].min()
                print(f"[LIDAR] {len(lidar_points)} points")
                print(f"  x_left range: [{lidar_points[:,0].min():.2f}, {lidar_points[:,0].max():.2f}] = {x_range:.2f}m")
                print(f"  y_forward range: [{lidar_points[:,1].min():.2f}, {lidar_points[:,1].max():.2f}] = {y_range:.2f}m")

        except Ice.Exception as e:
            print(f"Error reading lidar: {e}")
            lidar_points = np.array([])

        return lidar_points

    def explore_room(self):
        """
        Exploration behavior using a trajectory definition system.

        Trajectory is defined as a list of actions:
        - ('turn', angle_deg): Turn by angle in degrees (positive=left, negative=right)
        - ('forward', distance_m): Move forward by distance in meters
        - ('backward', distance_m): Move backward by distance in meters
        - ('wait', seconds): Wait for specified time

        Returns:
            tuple: (advx, advz, rot) - the last command issued to the robot
        """
        # Initialize trajectory on first call
        if not hasattr(self, '_trajectory_initialized'):
            self._init_trajectory()

        return self._execute_trajectory()

    def _init_trajectory(self):
        """Initialize the trajectory system with a test pattern"""
        self._trajectory_initialized = True

        # Movement parameters
        self.advance_speed = 200.0    # mm/s
        self.rotation_speed = 0.3     # rad/s
        self.period_sec = self.Period / 1000.0

        # Define a longer trajectory for better statistics
        # This creates a pattern that explores more of the room
        self._trajectory = [
            # Start: explore forward
            ('forward', 1.5),
            ('turn', -90),      # Turn right
            ('forward', 1.0),
            ('turn', -90),      # Turn right
            ('forward', 1.5),
            ('turn', 180),      # Turn around
            ('forward', 1.5),
            ('turn', 90),       # Turn left
            ('forward', 1.0),
            ('turn', 90),       # Turn left
            ('forward', 1.5),
            ('turn', -90),      # Turn right
            ('forward', 0.5),
            ('turn', -90),      # Turn right
            ('forward', 1.0),
            ('turn', 180),      # Turn around
            ('forward', 1.0),
            ('wait', 1.0),      # Pause to collect data
        ]

        self._current_action_idx = 0
        self._action_step = 0
        self._action_total_steps = 0
        self._trajectory_complete = False

        print(f"[Explorer] Trajectory initialized with {len(self._trajectory)} actions")
        self._print_trajectory()

    def _print_trajectory(self):
        """Print the trajectory plan"""
        print("[Explorer] Trajectory plan:")
        total_distance = 0
        total_rotation = 0
        for i, action in enumerate(self._trajectory):
            action_type, value = action
            if action_type == 'forward':
                print(f"  {i+1}. Forward {value:.1f}m")
                total_distance += value
            elif action_type == 'backward':
                print(f"  {i+1}. Backward {value:.1f}m")
                total_distance += value
            elif action_type == 'turn':
                direction = "left" if value > 0 else "right"
                print(f"  {i+1}. Turn {abs(value):.0f}° {direction}")
                total_rotation += abs(value)
            elif action_type == 'wait':
                print(f"  {i+1}. Wait {value:.1f}s")
        print(f"  Total: {total_distance:.1f}m distance, {total_rotation:.0f}° rotation")

    def _execute_trajectory(self):
        """Execute the current action in the trajectory"""
        cmd = (0.0, 0.0, 0.0)

        if self._trajectory_complete:
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            return cmd

        if self._current_action_idx >= len(self._trajectory):
            # Trajectory complete
            if not self._trajectory_complete:
                self._trajectory_complete = True
                self.omnirobot_proxy.setSpeedBase(0, 0, 0)
                print(f"[Explorer] Trajectory complete - robot stopped")
                self.print_statistics_summary()
            return cmd

        # Get current action
        action_type, value = self._trajectory[self._current_action_idx]

        # Calculate steps needed for this action (on first step)
        if self._action_step == 0:
            self._action_total_steps = self._calculate_action_steps(action_type, value)
            print(f"[Explorer] Action {self._current_action_idx + 1}/{len(self._trajectory)}: "
                  f"{action_type} {value} ({self._action_total_steps} steps)")

        # Execute action
        if action_type == 'forward':
            cmd = (0, self.advance_speed, 0)
        elif action_type == 'backward':
            cmd = (0, -self.advance_speed, 0)
        elif action_type == 'turn':
            # Positive angle = left (positive rotation), negative = right
            rot_dir = 1 if value > 0 else -1
            cmd = (0, 0, rot_dir * self.rotation_speed)
        elif action_type == 'wait':
            cmd = (0, 0, 0)

        self.omnirobot_proxy.setSpeedBase(*cmd)
        self._action_step += 1

        # Check if action complete
        if self._action_step >= self._action_total_steps:
            self._current_action_idx += 1
            self._action_step = 0

        return cmd

    def _calculate_action_steps(self, action_type, value):
        """Calculate number of steps needed for an action"""
        if action_type in ('forward', 'backward'):
            # value is distance in meters
            time_needed = abs(value) / (self.advance_speed / 1000.0)
            return int(time_needed / self.period_sec)
        elif action_type == 'turn':
            # value is angle in degrees
            angle_rad = abs(value) * np.pi / 180.0
            time_needed = angle_rad / self.rotation_speed
            return int(time_needed / self.period_sec)
        elif action_type == 'wait':
            # value is time in seconds
            return int(value / self.period_sec)
        return 0

    ##########################################################################################
    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        print(f"Testing RoboCompLidar3D.TDataCategory from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataCategory()
        print(f"Testing RoboCompLidar3D.TColorCloudData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TColorCloudData()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        QTimer.singleShot(200, QApplication.instance().quit)


    ######################
    # From the RoboCompLidar3D you can call this methods:
    # RoboCompLidar3D.TColorCloudData self.lidar3d_proxy.getColorCloudData()
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarData(str name, float start, float len, int decimationDegreeFactor)
    # RoboCompLidar3D.TDataImage self.lidar3d_proxy.getLidarDataArrayProyectedInImage(str name)
    # RoboCompLidar3D.TDataCategory self.lidar3d_proxy.getLidarDataByCategory(TCategories categories, long timestamp)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataProyectedInImage(str name)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataWithThreshold2d(str name, float distance, int decimationDegreeFactor)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # ifaces.RoboCompLidar3D.TPoint
    # ifaces.RoboCompLidar3D.TDataImage
    # ifaces.RoboCompLidar3D.TData
    # ifaces.RoboCompLidar3D.TDataCategory
    # ifaces.RoboCompLidar3D.TColorCloudData

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # RoboCompOmniRobot.void self.omnirobot_proxy.correctOdometer(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBasePose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBaseState(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.resetOdometer()
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometer(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometerPose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setSpeedBase(float advx, float advz, float rot)
    # RoboCompOmniRobot.void self.omnirobot_proxy.stopBase()

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # ifaces.RoboCompOmniRobot.TMechParams


    #
    # SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
    #
    def FullPoseEstimationPub_newFullPose(self, pose):
        # Debug: print raw values occasionally to understand coordinate convention
        if hasattr(self, 'total_steps') and self.total_steps % 100 == 0:
            print(f"[GT RAW] pose.x={pose.x:.3f}, pose.y={pose.y:.3f}, pose.rz={pose.rz:.3f}")

        # Coordinate convention: x+ = left, y+ = forward
        # Simulator uses same convention - no transformation needed
        self.robot_pose_gt = [pose.x, pose.y, pose.rz]


    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):

        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
