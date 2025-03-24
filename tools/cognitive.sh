#!/bin/bash

# Kill all running processes
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 Webots2Robocomp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 base_controller"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 g2o_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 room_detector_bt"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 scheduler"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 long_term_spatial_memory"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 fridge_concept"

# base_controller_agent
TAB_NAME="base_controller_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/base_controller_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/base_controller_agent etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# g20_agent
TAB_NAME="g2o_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/g2o_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/g2o_agent.py etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# room_detector_bt
TAB_NAME="room_detector_bt"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/room_detector_bt"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/room_detector_bt etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# scheduler
TAB_NAME="scheduler"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/scheduler"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "bin/scheduler etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

## long_term_spatial_memory_agent
TAB_NAME="long_term_spatial_memory_agent"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/long_term_spatial_memory_agent"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/long_term_spatial_memory_agent.py etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

# fridge_concept
TAB_NAME="fridge_concept"
DIRECTORY_PATH="~/robocomp/components/robocomp-shadow/agents/fridge_concept_python"
session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake . && make -j32"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "src/fridge_concept etc/config"
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id

