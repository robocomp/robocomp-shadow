cmake_minimum_required(VERSION 2.6)
PROJECT(  model_pybullet )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Camera360RGB Person )
