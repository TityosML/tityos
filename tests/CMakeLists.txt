file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_BaseTensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_TensorStorage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ShapeStrides.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_Tensor.cpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp"
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


target_link_directories(tityos_tests PRIVATE "../build")

target_link_libraries(tityos_tests
    PRIVATE Catch2::Catch2WithMain tityos
)

include(Catch)
catch_discover_tests(tityos_tests)