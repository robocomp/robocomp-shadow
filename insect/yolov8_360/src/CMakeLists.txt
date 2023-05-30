cmake_minimum_required(VERSION 2.6)
PROJECT(  yolov8_360 )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB Person SegmentatorTrackingPub VisualElements VisualElements )
