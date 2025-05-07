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
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
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
        self.plane = p.loadURDF("../../etc/URDFs/plane/plane.urdf", basePosition=[0, 0, 0])

        # Load robot in the simulation
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF("../../etc/URDFs/shadow/shadow.urdf", [-3.7, -0.7, 0.01], flags=flags)
        time.sleep(2)
        self.init_pos, self.init_orn = p.getBasePositionAndOrientation(self.robot)

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

        self.forward_velocity = 0
        self.angular_velocity = 0
        try:
            self.omnirobot_proxy.setSpeedBase(0, self.forward_velocity * 1000, self.angular_velocity * 1000)
        except Ice.ConnectionRefusedException:
            pass
        self.joystickControl = True

        # Set the state of the robot
        self.state = "start_moving"
        self.states = ["idle", "start_moving", "moving", "bump"]

        # Variables for the bump simulation
        self.generate_bumps = True
        self.bump_path = "./URDFs/bump/bump_100x10cm.urdf"
        self.number_of_bumps = 300
        self.bumps_coordinates = []
        self.bumps_cont = 0

        # Initialize IMU data
        # self.imu_data = {"time": [], "lin_acc": [], "ang_vel": [], "orientation": [], "prev_lin_vel": np.zeros(3)}
        self.imu_data_inner = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": [], "prev_lin_vel": np.zeros(3), "pos": []}
        self.imu_data_inner["time"].append(0)
        self.imu_data_inner["lin_acc_x"].append(0)
        self.imu_data_inner["lin_acc_y"].append(0)
        self.imu_data_inner["lin_acc_z"].append(0)
        self.imu_data_inner["pos"].append(p.getBasePositionAndOrientation(self.robot)[0])

        # Initialize IMU data of webots
        self.imu_data_webots = {"time": [], "lin_acc_x": [], "lin_acc_y": [], "lin_acc_z": [], "pos": []}
        self.imu_data_webots["time"].append(0)
        self.imu_data_webots["lin_acc_x"].append(0)
        self.imu_data_webots["lin_acc_y"].append(0)
        self.imu_data_webots["lin_acc_z"].append(0)

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1, figsize=(8, 10))
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(100)
        self.update_plot(update_legend=True)

        # Variables for the temporal analysis
        self.inner_cont = None
        self.mean_period = None
        self.use_mean_period = False

        # Get the initial time to compute the time of the episode
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
                # self.state = "start_moving"
                pass
            case "start_moving":
                # set webots robot speed
                self.forward_velocity = 0.5
                self.angular_velocity = 0
                try:
                    self.omnirobot_proxy.setSpeedBase(0, self.forward_velocity * 1000, self.angular_velocity * 1000)
                except Ice.ConnectionRefusedException:
                    pass
                # set inner robot speed
                # Get the velocity of the wheels from the forward and angular velocity of the robot
                wheels_velocities = self.get_wheels_velocity_from_forward_velocity_and_angular_velocity(
                    self.forward_velocity, self.angular_velocity)
                for motor_name in self.motors:
                    p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL,
                                            targetVelocity=wheels_velocities[motor_name])
                self.state = "moving"

            case "moving":        #move until a perceptive event occurs

                # read acceleration from Webots
                self.imu_data_webots = self.get_imu_data_webots()

                # read acceleration from PyBullet
                self.imu_data_inner = self.get_imu_data_inner(self.robot)

                # print last elements of self.imu_data_inner and self.imu_data_webots
                #print(f"Pybullet: {self.imu_data_inner['lin_acc_x'][-1]}, Webots: {self.imu_data_webots['lin_acc_x'][-1]}")
                #print(f"Pybullet: {self.imu_data_inner['lin_acc_y'][-1]}, Webots: {self.imu_data_webots['lin_acc_y'][-1]}")
                #print(f"Pybullet: {self.imu_data_inner['lin_acc_z'][-1]}, Webots: {self.imu_data_webots['lin_acc_z'][-1]}")

                #check if the difference is greater than a threshold
                print("Checking difference between Pybullet and Webots")
                print(f"Diff X: {abs(self.imu_data_inner['lin_acc_x'][-1] - self.imu_data_webots['lin_acc_x'][-1])}")
                print(f"Diff Y: {abs(self.imu_data_inner['lin_acc_z'][-1] - self.imu_data_webots['lin_acc_y'][-1])}")
                print(f"Diff Z: {abs(self.imu_data_inner['lin_acc_y'][-1] - self.imu_data_webots['lin_acc_z'][-1])}")

                thresh = 1
                if self.imu_data_webots['lin_acc_y'][-1] > thresh :
                    # stop the robots
                    print("Bump detected  -- STOPPING ROBOTS")
                    self.webot_stop_robot()
                    self.inner_stop_robot()
                    self.state = "bump_detected"

            case "bump_detected":       # stop the robot and replay the trajectory in Pybullet until a match
                if self.bumps_cont == self.number_of_bumps:
                    print("All bumps simulated. No match found")
                    self.reset_world()
                    self.inner_stop_robot()
                    self.state = "idle"

                else:
                    self.timer.stop()
                    self.plot_timer.stop()

                    # create a range of positions for the bump close to the final position of the episode
                    if self.generate_bumps:
                        print("Generating bumps")
                        last_coord = self.imu_data_inner["pos"][-1]
                        bump_x = last_coord[0] + np.random.uniform(-0.5, 0.5, self.number_of_bumps)
                        bump_y = last_coord[1] + np.random.uniform(-0.5, 0.5, self.number_of_bumps)
                        bump_z = np.full(self.number_of_bumps, 0.01)
                        self.bumps_coordinates = np.column_stack((bump_x, bump_y, bump_z))
                        self.generate_bumps = False

                    # reset the robot to the initial position of the episode
                    print("Resetting robot to initial position")
                    self.reset_world()

                    # introduce the bump in the scene
                    print("Introducing bump number ", self.bumps_cont)
                    self.bump = p.loadURDF(self.bump_path, self.bumps_coordinates[self.bumps_cont], flags=p.URDF_USE_INERTIA_FROM_FILE)

                    self.bumps_cont += 1

                    # set inner robot speed
                    # Get the velocity of the wheels from the forward and angular velocity of the robot
                    wheels_velocities = self.get_wheels_velocity_from_forward_velocity_and_angular_velocity(
                        self.forward_velocity, self.angular_velocity)
                    for motor_name in self.motors:
                        p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL,
                                                targetVelocity=wheels_velocities[motor_name])

                    # stadistics from the episode
                    print("Episode statistics")
                    periods = self.imu_data_webots["time"]
                    self.mean_period = np.mean(np.diff(periods))
                    self.use_mean_period = True
                    var = np.var(np.diff(periods))
                    print(f"Mean period: {self.mean_period}, Variance: {var}")

                    # setting the pybullet simulation velocity to the same as the recorded
                    p.setTimeStep(self.mean_period)

                    # self.state = "idle"
                    self.state = "simulate_bump"
                    self.inner_cont = 0
                    self.timer.start(1)

                # while not match:
                    # send the robot in pybullet to the initial position of the episode
                    # introduce the bump in the scene
                    # set same velocity as recorded
                    # read acceleration from PyBullet
                    # check if the peak occurs at the same time mark than the registered in the episode
                    # if match: exit the loop

                # print a sentence with the cause of the accident

            case "simulate_bump":
                # read acceleration from PyBullet
                self.imu_data_inner = self.get_imu_data_inner(self.robot)

                #check if the difference is greater than a threshold
                self.inner_cont += 1

                if len(self.imu_data_inner['time']) > len(self.imu_data_webots['time']):
                    print("Goood match")
                    print("Bump coordinates: ", self.bumps_coordinates[self.bumps_cont - 1])
                    self.reset_world()
                    self.inner_stop_robot()
                    self.state = "idle"
                else:
                    print("###############################Checking difference between Pybullet and Webots ########################################")
                    print(f"Diff Y: {abs(self.imu_data_inner['lin_acc_z'][self.inner_cont] - self.imu_data_webots['lin_acc_y'][self.inner_cont])}")
                    print(f"Pybullet Acc Y: {self.imu_data_inner['lin_acc_z'][self.inner_cont]}, Webots Acc Y: {self.imu_data_webots['lin_acc_y'][self.inner_cont]}")

                    p.stepSimulation()

                    thresh = 1
                    if abs(self.imu_data_inner['lin_acc_z'][self.inner_cont] - self.imu_data_webots['lin_acc_y'][self.inner_cont]) > thresh :
                        print("This bump is not the same as the one in the episode")
                        self.inner_stop_robot()
                        p.removeBody(self.bump)
                        self.state = "bump_detected"




    ###############################################################3
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

    def webot_stop_robot(self):
        """
        Stop the robot in Webots
        """
        try:
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
        except Exception as e:
            print(e)

    def inner_stop_robot(self):
        """
        Stop the robot in PyBullet
        """
        for motor_name in self.motors:
            p.setJointMotorControl2(self.robot, self.joints_name[motor_name], p.VELOCITY_CONTROL, targetVelocity=0)

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
        t = time.time() - self.start_time

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

    def get_imu_data_webots(self):
        try:
            temp_data = self.imu_proxy.getAcceleration()
            self.imu_data_webots["time"].append(time.time() - self.start_time)
            self.imu_data_webots["lin_acc_x"].append(temp_data.XAcc)
            self.imu_data_webots["lin_acc_y"].append(temp_data.YAcc)
            self.imu_data_webots["lin_acc_z"].append(temp_data.ZAcc)
            return self.imu_data_webots
        except Exception as e:
            print(e)

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

    def update_plot(self, update_legend=False):
        """
        Update the plot with the latest IMU data
        """

        if self.imu_data_inner is None or self.imu_data_webots is None:
            print("No IMU data")
            return

        # Graphing X axis
        self.ax[0].plot(self.imu_data_inner["time"], self.imu_data_inner["lin_acc_x"], label="Pybullet X", marker="o", color="blue")
        self.ax[0].plot(self.imu_data_webots["time"], self.imu_data_webots["lin_acc_x"], label="Webots X", marker="s",
                        color="red")
        self.ax[0].set_ylabel("Acc X")
        self.ax[0].set_title("Linear X Acceleration")
        self.ax[0].set_ylim(-5, 5)
        if update_legend: self.ax[0].legend()
        self.ax[0].grid(True)

        # Graphing Y axis
        self.ax[1].plot(self.imu_data_inner["time"], self.imu_data_inner["lin_acc_y"], label="Pybullet Y", marker="o", color="green")
        self.ax[1].plot(self.imu_data_webots["time"], self.imu_data_webots["lin_acc_y"], label="Webots Y", marker="s",
                        color="orange")
        self.ax[1].set_ylabel("Acc Y")
        self.ax[1].set_title("Linear Y Acceleration")
        self.ax[1].set_ylim(-5, 5)
        if update_legend: self.ax[1].legend()
        self.ax[1].grid(True)

        # Graphing Z axis
        self.ax[2].plot(self.imu_data_inner["time"], self.imu_data_inner["lin_acc_z"], label="Pybullet Z", marker="o", color="purple")
        self.ax[2].plot(self.imu_data_webots["time"], self.imu_data_webots["lin_acc_z"], label="Webots Z", marker="s",
                        color="black")
        self.ax[2].set_ylabel("Acc Z")
        self.ax[2].set_title("Linear Z Acceleration")
        self.ax[2].set_ylim(-5, 5)
        if update_legend: self.ax[2].legend()
        self.ax[2].grid(True)

        # Pause the plot for a short time to allow the plot to update
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
    # RoboCompIMU.Acceleration self.imu_proxy.getAcceleration()
    # RoboCompIMU.Gyroscope self.imu_proxy.getAngularVel()
    # RoboCompIMU.DataImu self.imu_proxy.getDataImu()
    # RoboCompIMU.Magnetic self.imu_proxy.getMagneticFields()
    # RoboCompIMU.Orientation self.imu_proxy.getOrientation()
    # RoboCompIMU.void self.imu_proxy.resetImu()

    ######################
    # From the RoboCompIMU you can use this types:
    # RoboCompIMU.Acceleration
    # RoboCompIMU.Gyroscope
    # RoboCompIMU.Magnetic
    # RoboCompIMU.Orientation
    # RoboCompIMU.DataImu

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
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


