target_sources(
    tityos
    PRIVATE
     ${CMAKE_CURRENT_LIST_DIR}/add.cpp
     ${CMAKE_CURRENT_LIST_DIR}/expand.cpp
     ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/cpu/CMakeLists.txt)