#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
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
from time import sleep, time

import pybullet as p
import numpy as np
import locale
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]
        self.g = DSRGraph(0, str(configData["Agent"]["name"]), int(configData["Agent"]["id"]))

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

        locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(cameraDistance=2.7, cameraYaw=0, cameraPitch=-15,
                                     cameraTargetPosition=[-1.3, -0.5, 0.2])

        # Load floor in the simulation
        self.plane = p.loadURDF("../../etc/URDFs/plane/plane.urdf", basePosition=[0, 0, 0])

        # Load robot in the simulation
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF("../../etc/URDFs/shadow/shadow.urdf", [-3.7, -0.7, 0.01], flags=flags)
        sleep(2)
        self.init_pos, self.init_orn = p.getBasePositionAndOrientation(self.robot)

        self.joint_names = self.get_joints_info(self.robot) # Get the joint names and IDs
        print(self.joint_names)

        self.imu_data_inner = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": [], "prev_lin_vel": np.zeros(3), "pos": []}
        self.imu_data_inner["time"].append(0)
        self.imu_data_inner["lin_acc_x"].append(0)
        self.imu_data_inner["lin_acc_y"].append(0)
        self.imu_data_inner["lin_acc_z"].append(0)
        self.imu_data_inner["pos"].append(p.getBasePositionAndOrientation(self.robot)[0])

        self.saved_state = p.saveState()

        # self.bump_100x5cm = p.loadURDF("./URDFs/bump/bump_100x5cm.urdf", [0, -0.33, 0.001], flags=flags)
        # self.bump_1000x10cm = p.loadURDF("./URDFs/bump/bump_100x10cm.urdf", [0, -0.33, 0.001], flags=flags)
        # self.cylinder_bump_10m = p.loadURDF("./URDFs/bump/cylinder_bump_10m.urdf", [0, -0.8, 0.001], p.getQuaternionFromEuler([0, 0, np.pi/2]), flags=flags)

        self.joints_name = self.get_joints_info(self.robot)
        self.links_name = self.get_link_info(self.robot)

        self.motors = ["frame_back_right2motor_back_right", "frame_back_left2motor_back_left", "frame_front_right2motor_front_right", "frame_front_left2motor_front_left"]
        self.wheels_radius = 0.1
        self.distance_between_wheels = 0.44
        self.distance_from_center_to_wheels = self.distance_between_wheels / 2

        self.target_velocity = np.array([0, 0, 0]) # forward, angular, side
        self.old_velocity = np.array([0, 0, 0]) # forward, angular, side
        
        # Variables for the temporal analysis
        self.inner_cont = None
        self.mean_period = None
        self.use_mean_period = False

        self.start_time = time()
        

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        p.disconnect()


    @QtCore.Slot()
    def compute(self):
        if  not np.array_equal(self.target_velocity, self.old_velocity):
            self.old_velocity = np.copy(self.target_velocity)
            wheels_velocities = self.get_wheels_velocity_from_forward_velocity_and_angular_velocity(
                        self.old_velocity[0], self.old_velocity[1])
            for motor_name in self.motors:
                p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL,
                                        targetVelocity=wheels_velocities[motor_name])
            
        # read acceleration from PyBullet
        self.imu_data_inner = self.get_imu_data_inner(self.robot)
        
        
        p.stepSimulation()
        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)


    def get_joints_info(self, robot_id):
        """
        Get joint names and IDs from a robot model
        :param robot_id: ID of the robot model in the simulation
        :return: Dictionary with joint names as keys and joint IDs as values
        """
        joint_name_to_id = {}
        # Get number of joints in the model
        num_joints = p.getNumJoints(robot_id)
        # print("Num joints:", num_joints)

        # Populate the dictionary with joint names and IDs
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_name_to_id[joint_name] = i
            jid = joint_info[0]
            jtype = joint_info[2]
            if jtype == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(bodyUniqueId=robot_id,
                                        jointIndex=jid,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=0,
                                        force=0)
        return joint_name_to_id

    def get_link_info(self, robot_id):
        """
        Get link names and IDs from a robot model
        :param robot_id: ID of the robot model in the simulation
        :return: Dictionary with link names as keys and link IDs as values
        """
        link_name_to_id = {}
        # Get number of joints in the model
        num_links = p.getNumJoints(robot_id)
        # print("Num links:", num_links)

        # Populate the dictionary with link names and IDs
        for i in range(num_links):
            link_info = p.getJointInfo(robot_id, i)
            link_name = link_info[12].decode("utf-8")
            link_name_to_id[link_name] = i
        return link_name_to_id

    def get_forward_velocity(self):
        """
        Get the forward velocity of the robot
        :return: Forward velocity
        """
        wheel_velocities = {}
        for motor_name in self.motors:
            wheel_velocities[motor_name] = p.getJointState(self.robot, self.joints_name[motor_name])[1]
        forward_velocity = (wheel_velocities["frame_front_left2motor_front_left"] +
                            wheel_velocities["frame_front_right2motor_front_right"] +
                            wheel_velocities["frame_back_left2motor_back_left"] +
                            wheel_velocities["frame_back_right2motor_back_right"]) * self.wheels_radius / 4
        return forward_velocity

    def get_angular_velocity(self):
        """
        Get the angular velocity of the robot
        :return: Angular velocity
        """
        wheel_velocities = {}
        for motor_name in self.motors:
            wheel_velocities[motor_name] = p.getJointState(self.robot, self.joints_name[motor_name])[1]
        angular_velocity = ((wheel_velocities["frame_front_right2motor_front_right"] +
                            wheel_velocities["frame_back_right2motor_back_right"] -
                            wheel_velocities["frame_front_left2motor_front_left"] -
                            wheel_velocities["frame_back_left2motor_back_left"]) * self.wheels_radius /
                            2 * self.distance_between_wheels)
        return angular_velocity

    def get_wheels_velocity_from_forward_velocity_and_angular_velocity(self, forward_velocity=0, angular_velocity=0):
        """
        Get the velocity of each wheel from the forward velocity of the robot
        :param forward_velocity: Forward velocity of the robot
        :param angular_velocity: Angular velocity of the robot
        :return: Dictionary with the velocity of each wheel
        """
        wheels_velocity = {
            "frame_front_left2motor_front_left": forward_velocity / self.wheels_radius - self.distance_from_center_to_wheels * angular_velocity / self.wheels_radius,
            "frame_front_right2motor_front_right": forward_velocity / self.wheels_radius + self.distance_from_center_to_wheels * angular_velocity / self.wheels_radius,
            "frame_back_left2motor_back_left": forward_velocity / self.wheels_radius - self.distance_from_center_to_wheels * angular_velocity / self.wheels_radius,
            "frame_back_right2motor_back_right": forward_velocity / self.wheels_radius + self.distance_from_center_to_wheels * angular_velocity / self.wheels_radius}
        return wheels_velocity

    def get_imu_data_inner(self, body_id):
        """
        Get IMU data from a body in the PyBullet simulation
        :param body_id: ID of the body in the simulation
        :return: Dictionary with IMU data
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        lin_vel, ang_vel = p.getBaseVelocity(body_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        prev_lin_vel = np.array(self.imu_data_inner.get("prev_lin_vel"))
        t = time() - self.start_time

        self.imu_data_inner["prev_lin_vel"] = lin_vel

        if self.use_mean_period:
            timeStep = self.mean_period
        else :
            timeStep = t - self.imu_data_inner["time"][-1]

        lin_acc = (np.array(lin_vel, dtype=np.float32) - prev_lin_vel) / timeStep

        # Store data
        self.imu_data_inner["time"].append(t)
        self.imu_data_inner["lin_acc_x"].append(lin_acc[0])
        self.imu_data_inner["lin_acc_y"].append(lin_acc[1])
        self.imu_data_inner["lin_acc_z"].append(lin_acc[2])
        self.imu_data_inner["pos"].append(pos)
        # self.imu_data["ang_vel"].append(ang_vel)
        # self.imu_data["orientation"].append((roll, pitch, yaw))

        return self.imu_data_inner

    def reset_world(self):
        """
        Reset the PyBullet simulation world
        """
        p.resetBasePositionAndOrientation(self.robot, self.init_pos, self.init_orn)
        time.sleep(0.1)
        self.imu_data_inner = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": [], "prev_lin_vel": np.zeros(3), "pos": []}
        self.imu_data_inner["time"].append(0)
        self.imu_data_inner["lin_acc_x"].append(0)
        self.imu_data_inner["lin_acc_y"].append(0)
        self.imu_data_inner["lin_acc_z"].append(0)
        self.imu_data_inner["pos"].append(p.getBasePositionAndOrientation(self.robot)[0])






    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        # console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')
        match id:
            case 200:
                if "robot_ref_adv_speed" in attribute_names:
                    node = self.g.get_node(id)
                    self.target_velocity[0] = node.attrs["robot_ref_adv_speed"].value
                elif "robot_ref_rot_speed" in attribute_names:
                    node = self.g.get_node(id)
                    self.target_velocity[1] = node.attrs["robot_ref_rot_speed"].value
                elif "robot_ref_side_speed" in attribute_names:
                    node = self.g.get_node(id)
                    self.target_velocity[2] = node.attrs["robot_ref_side_speed"].value
                    
        

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
