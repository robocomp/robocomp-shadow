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
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import pybullet as p
import numpy as np
import locale
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50

        locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35,
                                     cameraTargetPosition=[0, 0, 0.5])

        # Load floor in the simulation
        self.plane = p.loadURDF("./URDFs/plane/plane.urdf", basePosition=[0, 0, 0])

        # Load robot in the simulation
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF("./URDFs/shadow/shadow.urdf", [0, 0, 0], flags=flags)

        self.bump = p.loadURDF("./URDFs/bump/bump_100x5.urdf", [1.8, 0, 0.01], flags=flags)

        self.joints_name = self.get_joints_info(self.robot)
        self.links_name = self.get_link_info(self.robot)

        print("Joints name:", self.joints_name)
        print("Links name:", self.links_name)

        self.motors = ["frame_back_right2motor_back_right", "frame_back_left2motor_back_left", "frame_front_right2motor_front_right", "frame_front_left2motor_front_left"]
        self.wheels_radius = 0.1
        self.distance_between_wheels = 0.44
        self.distance_from_center_to_wheels = self.distance_between_wheels / 2

        self.forward_velocity = 0
        self.angular_velocity = 0

        self.saved_state = p.saveState()

        self.joystickControl = True

        self.state = "moving"
        self.states = ["idle", "moving", "bump"]

        # Initialize IMU data
        self.imu_data = {"time": [], "lin_acc": [], "ang_vel": [], "orientation": []}
        self.start_time = time.time()

        # Initialize plot
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 6))
        self.labels = ["X", "Y", "Z"]
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50)

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        match self.state:
            case "idle":
                pass
            case "moving":
                # self.forward_velocity = 0
                # self.angular_velocity = 0

                self.omnirobot_proxy.setSpeedBase(0, self.forward_velocity * 1000, self.angular_velocity * 1000)

                # Get the velocity of the wheels from the forward and angular velocity of the robot
                wheels_velocities = self.get_wheels_velocity_from_forward_velocity_and_angular_velocity(self.forward_velocity, self.angular_velocity)

                # Set the velocity of the motors
                for motor_name in self.motors:
                    p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL,
                                        targetVelocity=wheels_velocities[motor_name])

                plt.show()

            case "bump":

                # imu_data = self.get_imu_data(self.robot)
                # output = (
                #     f"\n--- IMU Data ---\n"
                #     f"Linear Acceleration: {imu_data['lin_acc']}\n"
                #     f"Angular Velocity   : {imu_data['ang_vel']}\n"
                #     f"Orientation (Euler): Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}\n"
                #     f"Orientation (Quat) : {imu_data['orientation_quat']}\n"
                #     f"------------------"
                # )
                # print(output)

                plt.show()
                pass


    def startup_check(self):
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompJoystickAdapter.AxisParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.AxisParams()
        print(f"Testing RoboCompJoystickAdapter.ButtonParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.ButtonParams()
        print(f"Testing RoboCompJoystickAdapter.TData from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.TData()
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
        print("Num joints:", num_joints)

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
        print("Num links:", num_links)

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

    def get_imu_data(self, body_id):
        """
        Get IMU data from a body in the simulation
        :param body_id: ID of the body in the simulation
        :return: Dictionary with IMU data
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        lin_vel, ang_vel = p.getBaseVelocity(body_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        self.imu_data["prev_lin_vel"] = self.imu_data.get("lin_vel", lin_vel)
        lin_acc = np.array(lin_vel) - np.array(self.imu_data["prev_lin_vel"])

        t = time.time() - self.start_time

        self.imu_data["time"].append(t)
        self.imu_data["lin_acc"].append(lin_acc.tolist())
        self.imu_data["ang_vel"].append(ang_vel)
        self.imu_data["orientation"].append((roll, pitch, yaw))

    def update_plot(self, frame):
        """
        Update the plot with the latest IMU data
        :param frame: Frame number
        """
        self.get_imu_data(self.robot)

        t_vals = self.imu_data["time"][-50:]
        lin_acc_vals = np.array(self.imu_data["lin_acc"][-50:])
        ang_vel_vals = np.array(self.imu_data["ang_vel"][-50:])
        orientation_vals = np.array(self.imu_data["orientation"][-50:])

        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()

        for i in range(3):
            self.axs[0].plot(t_vals, lin_acc_vals[:, i], label=f"Acc {self.labels[i]}")
        self.axs[0].set_title("Aceleración Lineal")
        self.axs[0].legend()

        for i in range(3):
            self.axs[1].plot(t_vals, ang_vel_vals[:, i], label=f"Vel Ang {self.labels[i]}")
        self.axs[1].set_title("Velocidad Angular")
        self.axs[1].legend()

        for i in range(3):
            self.axs[2].plot(t_vals, orientation_vals[:, i], label=f"Orient {self.labels[i]}")
        self.axs[2].set_title("Orientación (Roll, Pitch, Yaw)")
        self.axs[2].legend()

        plt.tight_layout()


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    #
    def JoystickAdapter_sendData(self, data):
        for b in data.buttons:
            if b.name == "block":
                if b.step == 1:
                    p.restoreState(self.saved_state)
            if b.name == "joystick_control":
                if b.step == 1:
                    self.joystickControl = not self.joystickControl
                    if not self.joystickControl:
                        self.angular_velocity = 0
                        self.forward_velocity = 0

                    print("Joystick control: ", self.joystickControl)
            else:
                pass  # print(b.name, "PULASDOR NO AJUSTADO")

        if self.joystickControl:
            for a in data.axes:
                if a.name == "rotate":
                    self.angular_velocity = a.value
                elif a.name == "advance":
                    self.forward_velocity = a.value * 0.0003
                else:
                    pass  # print(a.name, "JOYSTICK NO AJUSTADO")

    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # self.omnirobot_proxy.correctOdometer(...)
    # self.omnirobot_proxy.getBasePose(...)
    # self.omnirobot_proxy.getBaseState(...)
    # self.omnirobot_proxy.resetOdometer(...)
    # self.omnirobot_proxy.setOdometer(...)
    # self.omnirobot_proxy.setOdometerPose(...)
    # self.omnirobot_proxy.setSpeedBase(...)
    # self.omnirobot_proxy.stopBase(...)

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


