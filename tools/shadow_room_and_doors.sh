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
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 4 "gridder"

SESSION_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 5)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 5 "lidar_odom"

SESSION_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_6=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 6)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 6 "bumper"

SESSION_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_7=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 7)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 7 "joystick publish"

SESSION_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_8=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 8)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 8 "WebotsBridge"

SESSION_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_9=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 9)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 9 "base_controller"

SESSION_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_10=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 10)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 10 "g2o_cpp"

SESSION_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_11=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 11)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 11 "g2o_agent"

SESSION_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_12=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 12)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 12 "room_detector"

SESSION_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_13=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 13)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 13 "door_detector"

SESSION_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_14=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 14)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 14 "scheduler"

SESSION_ID_15=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_15=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 15)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 15 "ltsma"

# SESSION_ID_16=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
# TERMINAL_ID_16=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 16)
# qdbus org.kde.yakuake /yakuake/tabs setTabTitle 16 "BulletSim"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 python3"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "rcnode &"
sleep 1
# Webots

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "webots &"
sleep 5
#Bridge
# Camera

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cd ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "bin/RicohOmni etc/config_wb"



#Lidar H

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "bin/Lidar3D etc/config_helios_webots"

# Lidar P

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "bin/Lidar3D etc/config_pearl_webots"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cd ~/robocomp/components/robocomp-shadow/insect/gridder"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "bin/gridder etc/config_wb"
#MPC

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cd ~/robocomp/components/robocomp-shadow/insect/lidar_odometry"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "bin/lidar_odometry etc/config_wb"

#Bumper

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cd ~/robocomp/components/robocomp-shadow/insect/bumper"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "bin/bumper etc/config_wb"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "cd ~/robocomp/components/robocomp-robolab/components/hardware/external_control/joystickpublish"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "bin/JoystickPublish etc/config_shadow"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cd ~/robocomp/components/robocomp-shadow/agents/webots_bridge_iros"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "bin/Webots2Robocomp etc/config"


#SM_Grid
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "cd ~/robocomp/components/robocomp-shadow/agents/base_controller_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "bin/base_controller_agent etc/config"

#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "cd ~/robocomp/components/robocomp-shadow/components/g2o_cpp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "bin/g2o_cpp etc/config"

sleep 4

#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "cd ~/robocomp/components/robocomp-shadow/agents/g2o_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "src/g2o_agent.py etc/config"

# #Model
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "cd ~/robocomp/components/robocomp-shadow/agents/room_detector_bt"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "bin/room_detector_bt etc/config"

#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "cd ~/robocomp/components/robocomp-shadow/agents/door_detector"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "bin/door_detector etc/config"

#Scheduler
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "cd ~/robocomp/components/robocomp-shadow/agents/scheduler"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "bin/scheduler etc/config"

#ltsma
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "cd ~/robocomp/components/robocomp-shadow/agents/long_term_spatial_memory_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "src/long_term_spatial_memory_agent.py etc/config"


