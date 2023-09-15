#!/bin/bash

# Lidar3D
gnome-terminal -- bash -c "~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/bin/Lidar3D ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/etc/config_helios_jetson; exec bash"
gnome-terminal -- bash -c "~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/bin/Lidar3D ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/etc/config_pearl_jetson; exec bash"

# RicohOmni
gnome-terminal -- bash -c "~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/bin/RicohOmni ~/robocomp/components/robocomp-robolab/components/hardware/camera/ricoh_omni/etc/config; exec bash"

sleep 2

# yolov8_360
gnome-terminal -- bash -c "python3 ~/robocomp/components/robocomp-shadow/insect/yolov8_360/src/yolov8_360.py ~/robocomp/components/robocomp-shadow/insect/yolov8_360/etc/config; exec bash"

# hash_tracker
gnome-terminal -- bash -c "python3 ~/robocomp/components/robocomp-shadow/insect/hash_tracker/src/hash_tracker.py ~/robocomp/components/robocomp-shadow/insect/hash_tracker/etc/config_yolo; exec bash"

# grid_planner
gnome-terminal -- bash -c "~/robocomp/components/robocomp-shadow/insect/grid_planner/bin/grid_planner ~/robocomp/components/robocomp-shadow/insect/grid_planner/etc/config; exec bash"

# dwa
gnome-terminal -- bash -c "python3 ~/robocomp/components/robocomp-shadow/insect/dwa/src/dwa.py ~/robocomp/components/robocomp-shadow/insect/dwa/etc/config; exec bash"

# bumper
gnome-terminal -- bash -c "~/robocomp/components/robocomp-shadow/insect/bumper/bin/bumper ~/robocomp/components/robocomp-shadow/insect/bumper/etc/config; exec bash"

# controller
gnome-terminal -- bash -c "python3 ~/robocomp/components/robocomp-shadow/insect/controller/src/controller.py ~/robocomp/components/robocomp-shadow/insect/controller/etc/config; exec bash"

