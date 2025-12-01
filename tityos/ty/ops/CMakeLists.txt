target_sources(
    tityos
    PRIVATE
     ${CMAKE_CURRENT_LIST_DIR}/add.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/cpu/CMakeLists.txt)