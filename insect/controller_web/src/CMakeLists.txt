cmake_minimum_required(VERSION 2.6)
PROJECT(  controller )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB Camera360RGB GenericBase Lidar3D MPC MaskElements OmniRobot SegmentatorTrackingPub VisualElementsPub VisualElementsPub )
