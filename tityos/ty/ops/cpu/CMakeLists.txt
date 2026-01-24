target_sources(
    tityos_cpu
    PUBLIC
     ${CMAKE_CURRENT_LIST_DIR}/add.cpp
     ${CMAKE_CURRENT_LIST_DIR}/avx2/add.cpp
     ${CMAKE_CURRENT_LIST_DIR}/bmm.cpp
)