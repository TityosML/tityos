if (TITYOS_BUILD_CUDA)
    target_sources(
        tityos_cuda
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/add.cpp
        ${CMAKE_CURRENT_LIST_DIR}/add.cu
        ${CMAKE_CURRENT_LIST_DIR}/bmm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/bmm.cu
        ${CMAKE_CURRENT_LIST_DIR}/contiguous.cpp
        ${CMAKE_CURRENT_LIST_DIR}/contiguous.cu        
        ${CMAKE_CURRENT_LIST_DIR}/toCpu.cpp
    )
endif()