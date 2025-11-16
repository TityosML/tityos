file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/*.cpp"
)

add_executable(tityos_tests ${TEST_SOURCES})

target_link_libraries(tityos_tests PRIVATE Catch2::Catch2WithMain tityos)

include(Catch)
catch_discover_tests(tityos_tests)