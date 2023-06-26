#!/bin/bash

sleep 2      

TERMINAL_ID_0=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 0)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 0 "Coppelia"

 
 
SESSION_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 1)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 1 "LIDAR"



SESSION_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 2)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 2 "CAM"

SESSION_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 3)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 3 "YOLO"



SESSION_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 4)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 4 "HASH"

SESSION_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 5)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 5 "GRID"



SESSION_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_4=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 6)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 6 "DWA"


SESSION_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 7)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 7 "BUMPER"

SESSION_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_5=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 8)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 8 "CONTROL"




#RCNODE
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 python3"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "cd /home/robocomp/robocomp/components/"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "bash rcnode.sh &"


#Coppelia

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "cd /home/robocomp/robocomp/components/robocomp-shadow/pyrep/shadow/"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "bash run_rooms_path.sh"


sleep 10


#Lidar
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cd robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "bin/Lidar3D etc/config_helios"


#Camera
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cd robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "bin/RicohOmni etc/config_copp"


#YOLO
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cd ~/robocomp/components/robocomp-shadow/insect/yolov8_360"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "src/yolov8_360.py etc/config"

#HASH
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cd ~/robocomp/components/robocomp-shadow/insect/hash_tracker"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "src/hash_tracker.py etc/config_yolo"


#GRID
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cd ~/robocomp/components/robocomp-shadow/insect/grid_planner"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "bin/grid_planner etc/config"

#DWA
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "cd ~/robocomp/components/robocomp-shadow/insect/dwa"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 6 "src/dwa.py etc/config"

#BUMPER
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "cd ~/robocomp/components/robocomp-shadow/insect/bumper"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 7 "bin/bumper etc/config"


#CONTROL
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "cd ~/robocomp/components/robocomp-shadow/insect/controller"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "src/controller.py etc/config 















