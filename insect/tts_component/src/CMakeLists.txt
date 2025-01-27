cmake_minimum_required(VERSION 2.6)
PROJECT(  tts_component )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior Speech )
