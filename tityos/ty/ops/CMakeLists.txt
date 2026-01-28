target_sources(
    tityos
    PUBLIC
     ${CMAKE_CURRENT_LIST_DIR}/add.cpp
     ${CMAKE_CURRENT_LIST_DIR}/bmm.cpp
     ${CMAKE_CURRENT_LIST_DIR}/contiguous.cpp
     ${CMAKE_CURRENT_LIST_DIR}/expand.cpp
     ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/cpu/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/cuda/CMakeLists.txt)