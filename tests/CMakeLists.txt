file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp"
)

if (TITYOS_ENABLE_PERFORMANCE_TESTS)
    file(GLOB_RECURSE PERFORMANCE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/performance/*.cpp"
    )
    list(APPEND TEST_SOURCES ${PERFORMANCE_SOURCES})
endif()

add_executable(tityos_tests ${TEST_SOURCES})

target_include_directories(
    tityos_tests
    PUBLIC
        ${TY_ROOT_DIR}
)

if (TITYOS_BUILD_CUDA)
    target_include_directories(
        tityos_tests
        PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(tityos_tests PUBLIC CUDA::cudart)
endif()


target_link_directories(tityos_tests PRIVATE "../build")

target_link_libraries(tityos_tests PUBLIC Catch2::Catch2WithMain tityos)

include(Catch)
catch_discover_tests(tityos_tests)