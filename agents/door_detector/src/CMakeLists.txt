# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/Hungarian.cpp
  ../src/door_detector.cpp
  ../src/nodes.cpp
  ../src/custom_widget.h
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
)


INCLUDE( $ENV{ROBOCOMP}/cmake/modules/opencv4.cmake )
add_definitions(-g  -fmax-errors=1 -std=c++2a -fno-char8_t -I/usr/include/x86_64-linux-gnu/qt6/QtOpenGLWidgets/)
SET (LIBS ${LIBS}   dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets behaviortree_cpp Clipper2)

option(USE_QT6 "Use Qt6" ON) # OFF por defecto, puedes cambiarlo a ON si deseas usar Qt6

if( USE_QT6)
    LIST(APPEND SOURCES "/home/robocomp/robocomp/classes/abstract_graphic_viewer/abstract_graphic_viewer.cpp")
    LIST(APPEND HEADERS "/home/robocomp/robocomp/classes/abstract_graphic_viewer/abstract_graphic_viewer.h")
    LIST(APPEND LIBS QGLViewer-qt6 Qt6OpenGLWidgets Qt6PrintSupport)
  else()
    list(APPEND SOURCES "/home/robocomp/robocomp/classes/abstract_graphic_viewer_qt5/abstract_graphic_viewer.cpp")
    list(APPEND HEADERS "/home/robocomp/robocomp/classes/abstract_graphic_viewer_qt5/abstract_graphic_viewer.h")
    LIST(APPEND LIBS QGLViewer-qt5)
endif()

