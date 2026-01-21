target_sources(
    tityos
    PUBLIC
     ${CMAKE_CURRENT_LIST_DIR}/Backend.cpp
)

target_sources(
    tityos_cpu
    PUBLIC
     ${CMAKE_CURRENT_LIST_DIR}/CPUBackend.cpp
)

if (TITYOS_BUILD_CUDA)
    target_sources(
        tityos_cuda
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/CUDABackend.cpp
    )
endif()