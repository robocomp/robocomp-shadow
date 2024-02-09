cmake_minimum_required(VERSION 2.6)
PROJECT(  bulletsim )

INCLUDE( /home/robocomp/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior BulletSim Gridder )
