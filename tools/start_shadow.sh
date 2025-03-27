#!/bin/bash

# WEBOTS
TAB_NAME="webots"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "rcnode&"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "webots&"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id


# BRIDGE
TAB_NAME="bridge"
DIRECTORY_PATH="/home/robocomp/robocomp/components/webots-bridge" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/Webots2Robocomp etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# ROBOT
TAB_NAME="robot_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/shadow_agent" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/shadow_agent etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# CAMERA
TAB_NAME="camera"
DIRECTORY_PATH="~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/RicohOmni etc/config_wb"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# LIDAR3D-HELIOS
TAB_NAME="helios"
DIRECTORY_PATH="~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/Lidar3D etc/config_helios_webots"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# LIDAR3D-BPEARL
TAB_NAME="pearl"
DIRECTORY_PATH="~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/Lidar3D etc/config_pearl_webots"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# RGBD
TAB_NAME="rgbd"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/insect/RGBD_360" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/RGBD_360 etc/config_wb"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# joystick pub
TAB_NAME="joy"
DIRECTORY_PATH="~/robocomp/components/robocomp-robolab/components/hardware/external_control/python_xbox_controller" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/python_xbox_controller.py etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# gridder
TAB_NAME="gridder"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/insect/gridder" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/gridder etc/config_wb"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# bumper
TAB_NAME="bumper"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/insect/bumper" # replace with your desired path
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/bumper etc/config_wb"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# yolo
#TAB_NAME="yolo"
#DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/insect/environment_object_perception"
#session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
#qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/environment_object_perception.py etc/config_wb"
#qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
#TAB_NAME="odom"
#DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/insect/lidar_odometry"
#session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
#qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/lidar_odometry etc/config_wb"
#qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="base_controller_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/base_controller_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/base_controller_agent etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="g2o_cpp"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/components/g2o_cpp"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/g2o_cpp etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="g2o_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/g2o_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/g2o_agent.py etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="room_detector_bt"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/room_detector_bt"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/room_detector_bt etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="scheduler"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/scheduler"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/scheduler etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# lidar odometry
TAB_NAME="long_term_spatial_memory_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/long_term_spatial_memory_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/long_term_spatial_memory_agent.py etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id



# lidar odometry
#TAB_NAME="fridge_concept"
#DIRECTORY_PATH="~/robocomp/components/beta_robotica_class_private/room_detector"
#session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
#qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/room_detector etc/config"
#qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id
