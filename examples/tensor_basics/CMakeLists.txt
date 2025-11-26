cmake_minimum_required(VERSION 3.15)
project(Examples)

set(TITYOS_LIB_DIR "${CMAKE_SOURCE_DIR}/../../build")
set(TITYOS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../..")

if(WIN32)
    set(TITYOS_LIB_FILE "${TITYOS_LIB_DIR}/tityos.lib")
else()
    set(TITYOS_LIB_FILE "${TITYOS_LIB_DIR}/libtityos.a")
endif()

add_executable(tensor_basics main.cpp)

target_link_libraries(tensor_basics PRIVATE ${TITYOS_LIB_FILE})

target_include_directories(
    tensor_basics
    PUBLIC
        ${TITYOS_INCLUDE_DIR}
)