cmake_minimum_required(VERSION 2.6)
PROJECT(  dwa )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase GridPlanner Lidar3D OmniRobot SegmentatorTrackingPub VisualElements Person Camera360RGB)
