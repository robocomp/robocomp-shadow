cmake_minimum_required(VERSION 2.6)
PROJECT(  controller )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB GenericBase MPC MaskElements OmniRobot SegmentatorTrackingPub VisualElements VisualElements )
