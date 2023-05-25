#!/bin/bash

#---------------------CAMERA

TERMINAL_ID_0=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 0)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 0 "CAMERA"

#---------------------mask2former

SESSION_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_1=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 1)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 1 "mask2former"

qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.splitTerminalTopBottom "$TERMINAL_ID_1"


#---------------------YOLO

SESSION_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_2=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 2)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 2 "YOLO"

qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.splitTerminalTopBottom "$TERMINAL_ID_2"

#---------------------CONTROLLER

SESSION_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
TERMINAL_ID_3=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.terminalIdsForSessionId 3)
qdbus org.kde.yakuake /yakuake/tabs setTabTitle 3 "CONTROLLER"

qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.splitTerminalTopBottom "$TERMINAL_ID_3"


sleep 2
#----------------------------------------------------------------------------------------------------#

#-------------------------------CAMERA-------------------------
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 python3&"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "cd ~/robocomp/robocomp_tools/rcnode"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "bash rcnode.sh&"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "cd ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "bin/RicohOmni etc/config"

#-------------------------------mask2former-------------------------

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "cd ~/robocomp/components/robocomp-shadow/insect/semantic_segmentation"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 1 "src/semantic_segmentation.py etc/config"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "cd ~/robocomp/components/robocomp-shadow/insect/byte_tracker"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 2 "src/byte_tracker.py etc/config_m2f"

#-------------------------------YOLO-------------------------
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "cd ~/robocomp/components/robocomp-shadow/insect/yolov8_360"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 3 "src/yolov8_360.py etc/config"

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "cd ~/robocomp/components/robocomp-shadow/insect/byte_tracker"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 4 "src/byte_tracker.py etc/config_yolo"

#-------------------------------CONTROLLER-------------------------
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "cd ~/robocomp/components/robocomp-shadow/insect/controller"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 5 "src/controller.py etc/config"




