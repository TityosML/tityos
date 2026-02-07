file(GLOB_RECURSE CORE_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/*.cpp"
)

set(TEST_SOURCES ${CORE_TEST_SOURCES})

if (TITYOS_BUILD_CUDA)
    file(GLOB_RECURSE CUDA_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/cuda/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/cuda/*.cpp"
    )
    list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif()

if (TITYOS_ENABLE_PERFORMANCE_TESTS)
    file(GLOB_RECURSE CORE_PERFORMANCE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/performance/core/*.cpp"
    )
    list(APPEND TEST_SOURCES ${CORE_PERFORMANCE_SOURCES})

    if (TITYOS_BUILD_CUDA)
        file(GLOB_RECURSE CUDA_PERFORMANCE_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/performance/cuda/*.cpp"
        )
        list(APPEND TEST_SOURCES ${CUDA_PERFORMANCE_SOURCES})
    endif()
endif()

add_executable(tityos_tests ${TEST_SOURCES})

target_include_directories(tityos_tests PUBLIC ${TY_ROOT_DIR})

if (TITYOS_BUILD_CUDA)
    target_include_directories(tityos_tests PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(tityos_tests PUBLIC CUDA::cudart)
endif()

# TODO: Change this
target_link_directories(tityos_tests PUBLIC "../build")

target_link_libraries(tityos_tests PUBLIC Catch2::Catch2WithMain tityos)

include(Catch)
catch_discover_tests(tityos_tests)
