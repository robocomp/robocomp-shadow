#!/bin/bash

sleep 2      

TERMINAL_ID_0=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 0)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 0 "Webots"

SESSION_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 1)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 1 "Wb-bridge"

SESSION_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 2)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 2 "RicohOmni"

SESSION_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 3)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 3 "Lidar H"

SESSION_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 4)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 4 "Lidar P"

SESSION_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 5)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 5 "360RGBD"

SESSION_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 6)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 6 "Tracking"

SESSION_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 7)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 7 "Controller"

SESSION_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 8)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 8 "Gridder"

SESSION_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 9)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 9 "MPC"

SESSION_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 10)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 10 "Bumper"

SESSION_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 11)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 11 "SM-Grid"

SESSION_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 12)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 12 "Obj-br"

SESSION_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 13)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 13 "LidarOdo"

SESSION_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 14)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 14 "Intention"

SESSION_ID_15=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_15=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 15)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 15 "Forcefield"

SESSION_ID_16=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_16=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 16)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 16 "Bulletsim"


SESSION_ID_17=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_17=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 17)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 17 "Hazard"

SESSION_ID_18=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_18=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 18)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 18 "Bullet-Hazard"

SESSION_ID_18=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_18=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 19)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 19 "Human Drivers"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 python3"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "rcnode &"

# Webots

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "webots &"

#Bridge

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cd ~/robocomp/components/webots-bridge"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "bin/Webots2Robocomp etc/config"

# Camera

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cd ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "bin/RicohOmni etc/config_wb"

sleep 3

#Lidar H

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "bin/Lidar3D etc/config_helios_webots"

# Lidar P

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "bin/Lidar3D etc/config_pearl_webots"

#360RGBD

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cd ~/robocomp/components/robocomp-shadow/insect/RGBD_360"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "bin/RGBD_360 etc/config_wb"

#object

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cd ~/robocomp/components/robocomp-shadow/insect/object_tracking_new"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "src/object_tracking.py etc/config_wb"
# Controller

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "cd ~/robocomp/components/robocomp-shadow/insect/controller_web"
#qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "src/controller.py etc/config"

#Grid

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cd ~/robocomp/components/robocomp-shadow/insect/gridder"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "bin/gridder etc/config_wb"
#MPC

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "cd ~/robocomp/components/robocomp-shadow/components/mpc"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "bin/MPC etc/config_wb"

#Bumper

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "cd ~/robocomp/components/robocomp-shadow/insect/bumper"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "bin/bumper etc/config_wb"

#SM_Grid
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "cd ~/robocomp/components/robocomp-shadow/agents/sm_grid_planner_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "bin/sm_grid_planner_agent etc/config_wb"

#Model
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "cd ~/robocomp/components/robocomp-shadow/agents/objects_bridge_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "bin/objects_bridge_agent etc/config"

#Lidar Odometry
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "cd ~/robocomp/components/robocomp-shadow/insect/lidar_odometry"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "bin/lidar_odometry etc/config_wb"

#Intention
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "cd ~/robocomp/components/robocomp-shadow/agents/intention_predictor_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "bin/intention_prediction_agent etc/config"

#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "cd ~/robocomp/components/robocomp-shadow/insect/forcefield"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "bin/forcefield etc/config_wb"

#Bulletsim
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 16 "cd ~/robocomp/components/robocomp-shadow/components/bulletsim"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 16 "src/bulletsim.py etc/config"

#Hazard
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 17 "cd ~/robocomp/components/robocomp-shadow/agents/hazard_detector"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 17 "bin/hazard_detector etc/config"

#Bulletsim-hazard
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 18 "cd ~/robocomp/components/robocomp-shadow/components/bulletsim"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 18 "src/bulletsim.py etc/config_hazard"

#Human drivers
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 18 "cd ~/robocomp/components/robocomp-shadow/components/human_driver_wb"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 18 "bin/human_driver_wb etc/config"

