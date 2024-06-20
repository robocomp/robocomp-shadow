# Kill process

qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 8 "killall -9 Webots2Robocomp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 9 "killall -9 base_controller"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 10 "killall -9 g2o_agent"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 11 "killall -9 g2o_cpp"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 12 "killall -9 room_detector_b"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 13 "killall -9 door_detector"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 14 "killall -9 scheduler"
qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal 15 "killall -9 long_term_spatial_memory"

sleep 1


