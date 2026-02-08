cmake_minimum_required(VERSION 3.15)
project(TityosExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

find_package(tityos CONFIG REQUIRED)

add_executable(bmm_example main.cpp)

target_link_libraries(bmm_example PRIVATE tityos::tityos)