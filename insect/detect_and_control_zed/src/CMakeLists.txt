cmake_minimum_required(VERSION 2.6)
PROJECT(  detect_and_control_zed )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase GridPlanner OmniRobot SegmentatorTrackingPub Speech VisualElementsPub WhisperStream )
