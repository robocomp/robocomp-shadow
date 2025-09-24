# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/GTSAMGraph.cpp
  ../src/custom_widget.h
../src/fixedsizedeque.h
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
)

include_directories(/usr/local/include)
LIST(APPEND LIBS ${LIBS} gtsam_unstable gtsam Clipper2 tbb)

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-fmax-errors=1 -fno-char8_t)

