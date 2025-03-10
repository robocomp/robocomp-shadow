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
import matplotlib

matplotlib.use("TkAgg")


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1

        locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(cameraDistance=2.7, cameraYaw=0, cameraPitch=-15,
                                     cameraTargetPosition=[-1.3, -0.5, 0.2])

        # Load floor in the simulation
        self.plane = p.loadURDF("./URDFs/plane/plane.urdf", basePosition=[0, 0, 0])

        # Load robot in the simulation
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF("./URDFs/shadow/shadow.urdf", [-3.7, -0.7, 0], flags=flags)

        # self.bump = p.loadURDF("./URDFs/bump/bump_100x5.urdf", [0, -0.33, 0.001], flags=flags)

        self.joints_name = self.get_joints_info(self.robot)
        self.links_name = self.get_link_info(self.robot)

        self.motors = ["frame_back_right2motor_back_right", "frame_back_left2motor_back_left", "frame_front_right2motor_front_right", "frame_front_left2motor_front_left"]
        self.wheels_radius = 0.1
        self.distance_between_wheels = 0.44
        self.distance_from_center_to_wheels = self.distance_between_wheels / 2

        self.forward_velocity = 0
        self.angular_velocity = 0

        self.omnirobot_proxy.setSpeedBase(0, self.forward_velocity * 1000, self.angular_velocity * 1000)

        self.saved_state = p.saveState()

        self.joystickControl = True

        self.state = "moving"
        self.states = ["idle", "moving", "bump"]

        # Initialize IMU data
        # self.imu_data = {"time": [], "lin_acc": [], "ang_vel": [], "orientation": [], "prev_lin_vel": np.zeros(3)}
        self.imu_data = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": [], "prev_lin_vel": np.zeros(3)}
        self.imu_data["time"].append(0)
        self.imu_data["lin_acc_x"].append(0)
        self.imu_data["lin_acc_y"].append(0)
        self.imu_data["lin_acc_z"].append(0)

        # Initialize IMU data of webots
        self.imu_data_webots = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": []}
        self.imu_data_webots["time"].append(0)
        self.imu_data_webots["lin_acc_x"].append(0)
        self.imu_data_webots["lin_acc_y"].append(0)
        self.imu_data_webots["lin_acc_z"].append(0)

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 10))

        # self.plot_timer = QTimer()
        # self.plot_timer.timeout.connect(self.update_plot)
        # self.plot_timer.start(50)

        self.update_plot(update_legend=True)

        self.start_time = time.time()

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
            case "moving":        #move until a perceptive event occurs

                temp_data = self.imu_proxy.getAcceleration()
                self.imu_data_webots["time"].append(time.time() - self.start_time)
                self.imu_data_webots["lin_acc_x"].append(temp_data.XAcc)
                self.imu_data_webots["lin_acc_y"].append(temp_data.YAcc)
                self.imu_data_webots["lin_acc_z"].append(temp_data.ZAcc)

                self.get_imu_data(self.robot)

                self.forward_velocity = 0.2
                self.angular_velocity = 0

                self.omnirobot_proxy.setSpeedBase(0, self.forward_velocity * 1000, self.angular_velocity * 1000)

                # Get the velocity of the wheels from the forward and angular velocity of the robot
                wheels_velocities = self.get_wheels_velocity_from_forward_velocity_and_angular_velocity(
                    self.forward_velocity, self.angular_velocity)

                # Set the velocity of the motors
                for motor_name in self.motors:
                    p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL,
                                        targetVelocity=wheels_velocities[motor_name])

                self.update_plot()

            case "bump":       # stop the robot and replay the trajectory in Pybullet until a match

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
                pass



    def startup_check(self):
        print(f"Testing RoboCompIMU.Acceleration from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Acceleration()
        print(f"Testing RoboCompIMU.Gyroscope from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Gyroscope()
        print(f"Testing RoboCompIMU.Magnetic from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Magnetic()
        print(f"Testing RoboCompIMU.Orientation from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.Orientation()
        print(f"Testing RoboCompIMU.DataImu from ifaces.RoboCompIMU")
        test = ifaces.RoboCompIMU.DataImu()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompJoystickAdapter.axisParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.axisParams()
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

    def get_imu_data(self, body_id):
        """
        Get IMU data from a body in the simulation
        :param body_id: ID of the body in the simulation
        :return: Dictionary with IMU data
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        lin_vel, ang_vel = p.getBaseVelocity(body_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        prev_lin_vel = np.array(self.imu_data.get("prev_lin_vel"))
        t = time.time() - self.start_time

        self.imu_data["prev_lin_vel"] = lin_vel

        timeStep = t - self.imu_data["time"][-1]

        lin_acc = (np.array(lin_vel, dtype=np.float32) - prev_lin_vel) / timeStep

        # Store data
        self.imu_data["time"].append(t)
        self.imu_data["lin_acc_x"].append(lin_acc[0])
        self.imu_data["lin_acc_y"].append(lin_acc[1])
        self.imu_data["lin_acc_z"].append(lin_acc[2])
        # self.imu_data["ang_vel"].append(ang_vel)
        # self.imu_data["orientation"].append((roll, pitch, yaw))

    def update_plot(self, update_legend=False):
        """
        Update the plot with the latest IMU data
        """
        self.ax[0].plot(self.imu_data["time"], self.imu_data["lin_acc_x"], label="Pybullet X", marker="o", color="blue")
        self.ax[0].plot(self.imu_data_webots["time"], self.imu_data_webots["lin_acc_x"], label="Webots X", marker="s",
                        color="red")
        self.ax[0].set_ylabel("Acc X")
        self.ax[0].set_title("Linear X Acceleration")
        self.ax[0].set_ylim(-2, 2)
        if update_legend: self.ax[0].legend()
        self.ax[0].grid(True)

        # Graficar eje Y
        self.ax[1].plot(self.imu_data["time"], self.imu_data["lin_acc_y"], label="Pybullet Y", marker="o", color="green")
        self.ax[1].plot(self.imu_data_webots["time"], self.imu_data_webots["lin_acc_y"], label="Webots Y", marker="s",
                        color="orange")
        self.ax[1].set_ylabel("Acc Y")
        self.ax[1].set_title("Linear Y Acceleration")
        self.ax[1].set_ylim(-2, 2)
        if update_legend: self.ax[1].legend()
        self.ax[1].grid(True)

        # Pausar para visualizar la actualización
        plt.pause(0.1)


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
                    self.forward_velocity = a.value * 0.001
                else:
                    pass  # print(a.name, "JOYSTICK NO AJUSTADO")

    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompIMU you can call this methods:
    # self.imu_proxy.getAcceleration(...)
    # self.imu_proxy.getAngularVel(...)
    # self.imu_proxy.getDataImu(...)
    # self.imu_proxy.getMagneticFields(...)
    # self.imu_proxy.getOrientation(...)
    # self.imu_proxy.resetImu(...)

    ######################
    # From the RoboCompIMU you can use this types:
    # RoboCompIMU.Acceleration
    # RoboCompIMU.Gyroscope
    # RoboCompIMU.Magnetic
    # RoboCompIMU.Orientation
    # RoboCompIMU.DataImu

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
    # RoboCompJoystickAdapter.axisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


