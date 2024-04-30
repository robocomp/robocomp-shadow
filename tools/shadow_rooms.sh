#!/bin/bash

sleep 2      

TERMINAL_ID_0=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 0)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 0 "Webots"

SESSION_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 1)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 1 "RicohOmni"

SESSION_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 2)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 2 "Lidar H"

SESSION_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 3)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 3 "Lidar P"

SESSION_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 4)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 4 "360RGBD"

SESSION_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 5)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 5 "Tracking"

SESSION_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 6)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 6 "Gridder"

SESSION_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 7)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 7 "MPC"

SESSION_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 8)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 8 "Lidar_odo"

SESSION_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 9)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 9 "Bumper"

SESSION_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 10)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 10 "Webots_bridge"

SESSION_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 11)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 11 "Forcefield_agent"

SESSION_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 12)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 12 "People_bridge"

SESSION_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 13)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 13 "SM_gridder_agent"

SESSION_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 14)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 14 "Room_planner"


qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 python3"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "rcnode &"

#Webots
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "webots &"

#Bridge
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "cd ~/robocomp/components/robocomp-shadow/agents/webots_bridge_iros"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "bin/Webots2Robocomp etc/config"

#Camera
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cd ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "bin/RicohOmni etc/config_wb"

sleep 3

#Lidar H
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "bin/Lidar3D etc/config_helios_webots"

# Lidar P
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "bin/Lidar3D etc/config_pearl_webots"

sleep 3

#360RGBD
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cd ~/robocomp/components/robocomp-shadow/insect/RGBD_360"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "bin/RGBD_360 etc/config_wb"

sleep 3

#Object_tracking
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cd ~/robocomp/components/robocomp-shadow/insect/object_tracking_new"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "src/object_tracking.py etc/config_wb_front"

#Gridder
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cd ~/robocomp/components/robocomp-shadow/insect/gridder"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "bin/gridder etc/config_wb"

#MPC
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "cd ~/robocomp/components/robocomp-shadow/components/mpc"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "bin/MPC etc/config_wb"

#Lidar_odometry
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cd ~/robocomp/components/robocomp-shadow/insect/lidar_odometry"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "bin/lidar_odometry etc/config_wb"

#Bumper
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "cd ~/robocomp/components/robocomp-shadow/insect/bumper"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "bin/bumper etc/config_wb"

# Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "cd ~/robocomp/components/robocomp-shadow/agents/forcefield_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "bin/forcefield_agent etc/config"

#People_bridge
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "cd ~/robocomp/components/robocomp-shadow/agents/people_bridge"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "bin/people_bridge_agent etc/config_wb"

#SM_Grid
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "cd ~/robocomp/components/robocomp-shadow/agents/sm_grid_planner_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "bin/sm_grid_planner_agent etc/config_wb"

#Room_planner
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "cd ~/robocomp/components/robocomp-shadow/agents/room_planner"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "bin/room_planner_agent etc/config"

