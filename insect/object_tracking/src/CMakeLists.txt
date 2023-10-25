cmake_minimum_required(VERSION 2.6)
PROJECT(  object_tracking )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB Camera360RGB Lidar3D Person SegmentatorTrackingPub VisualElements VisualElements )
