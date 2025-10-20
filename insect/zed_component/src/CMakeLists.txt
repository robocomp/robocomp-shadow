# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
)


INCLUDE( $ENV{ROBOCOMP}/cmake/modules/opencv4.cmake )

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    LIST(APPEND LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

LIST(APPEND LIBS ${LIBS} ${ZED_LIBS} )

set(CXX_STANDARD 17)

