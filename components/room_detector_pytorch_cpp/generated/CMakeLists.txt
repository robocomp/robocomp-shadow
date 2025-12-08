cmake_minimum_required(VERSION 3.10)

# Opción A: fijar la política explícitamente (recomendado)
if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

#Select the compiler standard
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "Version of g++ is lower than 11, g++ 11 or higher is required.")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    message(STATUS "Using g++ 11, setting C++20 standard.")
    set(CXX_STANDARD 20)
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(STATUS "Using g++ 12, setting C++23 standard.")
    set(CXX_STANDARD 23)
else()
    message(STATUS "Using g++ 13 or higher, setting the C++23 standard.")
    set(CXX_STANDARD 23)
endif()


IF(NOT EXISTS $ENV{ROBOCOMP}/cmake)
  MESSAGE(FATAL_ERROR "Couldn't find RoboComp in $ENV{ROBOCOMP}! Exiting...")
ENDIF(NOT EXISTS $ENV{ROBOCOMP}/cmake)

INCLUDE($ENV{ROBOCOMP}/cmake/robocomp.cmake)
INCLUDE($ENV{ROBOCOMP}/cmake/modules/qt.cmake)
INCLUDE (../src/CMakeLists.txt)

ADD_DEFINITIONS ("-DICE_CPP11_MAPPING")
FIND_PACKAGE( Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)

# Sources set
SET (SOURCES
  ${SOURCES}
  $ENV{ROBOCOMP}/classes/rapplication/rapplication.cpp
  $ENV{ROBOCOMP}/classes/sigwatch/sigwatch.cpp
  $ENV{ROBOCOMP}/classes/grafcetStep/GRAFCETStep.cpp
  $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp
  main.cpp
  genericworker.cpp
  joystickadapterI.cpp
)


#ROBOCOMP
ROBOCOMP_INITIALIZE( $ENV{ROBOCOMP}/ )
ROBOCOMP_IDSL_TO_ICE(Camera360RGBD Lidar3D OmniRobot GenericBase Camera360RGB JoystickAdapter)
ROBOCOMP_ICE_TO_SRC(Camera360RGBD Lidar3D OmniRobot GenericBase Camera360RGB JoystickAdapter)
SET (EXECUTABLE_OUTPUT_PATH ${RC_COMPONENT_DEVEL_PATH}/bin)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Specify construction and link process
ADD_EXECUTABLE( room_detector_pytorch_cpp ${SOURCES} ${MOC_SOURCES} ${RC_SOURCES} ${UI_HEADERS} )
# ===================================================================
# OPTIMIZACIÓN DE COMPILACIÓN: CABECERAS PRECOMPILADAS (PCH)
# ===================================================================
# Esto compila las cabeceras pesadas de 'src/pch.h' una sola vez.
target_precompile_headers(room_detector_pytorch_cpp
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/pch.h
)
TARGET_LINK_LIBRARIES( room_detector_pytorch_cpp ${LIBS}  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES} )
SET_TARGET_PROPERTIES(room_detector_pytorch_cpp
        PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )


INSTALL(FILES ${EXECUTABLE_OUTPUT_PATH}/room_detector_pytorch_cpp DESTINATION ${RC_COMPONENT_INSTALL_PATH}/bin/ PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
