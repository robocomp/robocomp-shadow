# Kill process

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 Webots2Robocomp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 base_controller"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 g2o_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 g2o_cpp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 room_detector_b"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 door_detector"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 scheduler"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 0 "killall -9 long_term_spatial_memory"

sleep 1

#RELAUNCH
#Bridge
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "bin/Webots2Robocomp etc/config"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "bin/base_controller_agent etc/config"

#SM_Grid#Scheduler
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "bin/scheduler etc/config"

#ltsma
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "src/long_term_spatial_memory_agent.py etc/config"



#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "bin/g2o_cpp etc/config"

#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "src/g2o_agent.py etc/config"

sleep 2
# #Model
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "bin/room_detector_bt etc/config"
#Forcefield
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "bin/door_detector etc/config"




