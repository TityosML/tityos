target_sources(
    tityos
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/TensorStorage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ShapeStrides.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Tensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/BaseTensor.cpp
)

target_sources(
    tityos_cpu
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/cpu/TensorStorage.cpp
)

if (TITYOS_BUILD_CUDA)
    target_sources(
        tityos_cuda
        PUBLIC
            ${CMAKE_CURRENT_LIST_DIR}/cuda/TensorStorage.cpp
    ) 
endif()