cmake_minimum_required(VERSION 3.15)

project(
    tityos
    VERSION 1.0.0 
    LANGUAGES CXX
)

set(TY_ROOT_DIR ${PROJECT_SOURCE_DIR}/tityos)

# ================================================================
# Compiler configuration
# ================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DEBUG_POSTFIX debug)

if(CMAKE_SYSTEM_NAME MATCHES "CYGWIN" OR CMAKE_SYSTEM_NAME MATCHES "MSYS" OR CMAKE_SYSTEM_NAME MATCHES "MINGW")
    set(CMAKE_CXX_EXTENSIONS ON)
endif()

# ================================================================
# Build Options
# ================================================================

# Build Shared/Static
option(TITYOS_BUILD_SHARED "Build shared" OFF)

# Sanitizing options
option(TITYOS_SANITIZE_ADDRESS "Enable address sanitizer in tests" OFF)
option(TITYOS_SANITIZE_MEMORY "Enable memroy sanitizer in tests" OFF)
if (NOT(CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND TITYOS_SANITIZE_MEMORY)
    message(FATAL_ERROR "TITYOS_SANITIZE_MEMORY can only be enabled with Clang") 
endif()
if(TITYOS_SANITIZE_ADDRESS AND TITYOS_SANITIZE_MEMORY)
    message(FATAL_ERROR "Only one of TITYOS_SANITIZE_ADDRESS and TITYOS_SANITIZE_MEMORY can be enabled")
endif()

# Compiler Warning options
option(TITYOS_BUILD_WARNINGS "Enable compiler warnings" OFF)

# Testing options
option(TITYOS_BUILD_TESTS "Build tests" OFF)

# CUDA options
option(TITYOS_USE_CUDA "Build to allow cuda" OFF)

# CPU SIMD options
option(TITYOS_USE_AVX "Build to allow AVX" OFF)
option(TITYOS_USE_AVX2 "Build to allow AVX2" OFF)

# ================================================================
#  Shared or Static library
# ================================================================

include(CheckCXXCompilerFlag)

if(TITYOS_BUILD_SHARED)
    add_library(tityos SHARED)
    target_compile_definitions(tityos PRIVATE TITYOS_BUILD_SHARED)
else()
    add_library(tityos STATIC)
endif()

target_include_directories(
    tityos
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA
if (TITYOS_USE_CUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
        message(STATUS "Compiling using CUDA")
        target_compile_definitions(tityos PRIVATE TITYOS_USE_CUDA)
        target_link_libraries(tityos PRIVATE CUDA::cudart)
    else()
        message(WARNING "Cuda not found - Compiling using CPU")
        set(TITYOS_USE_CUDA OFF)
    endif()
endif()

# CPU SIMD
if (TITYOS_USE_AVX2)
    message(STATUS "Checking for AVX2 support")
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_ALLOWS_AVX2)
        if (COMPILER_ALLOWS_AVX2)
            target_compile_options(tityos PRIVATE /arch:AVX2)
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_ALLOWS_AVX2)
        if (COMPILER_ALLOWS_AVX2)
            target_compile_options(tityos PRIVATE -mavx2)
        endif()
    endif()

    if(NOT COMPILER_ALLOWS_AVX2)
        message(WARNING "Checking for AVX2 support - Failed")
        message(WARNING "AVX2 not supported - continuing without AVX2")
    else()
        message(STATUS "Checking for AVX2 support - Success")
    endif()
endif()

# Warnings
if(TITYOS_BUILD_WARNINGS)
    if(MSVC)
        target_compile_options(tityos PRIVATE /W4 /WX)
    else()
        target_compile_options(tityos PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

# Sanitizer
if(TITYOS_SANITIZE_ADDRESS)
    target_compile_options(tityos PRIVATE -fsanitize=address -g)
    target_link_options(tityos PRIVATE -fsanitize=address)
elseif(TITYOS_SANITIZE_MEMORY)
    target_compile_options(tityos PRIVATE -fsanitize=memory -g)
    target_link_options(tityos PRIVATE -fsanitize=memory)
endif()

# Hiding for non Win32
if(NOT WIN32)
    set_target_properties(tityos PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN TRUE
    )
endif()

include(${TY_ROOT_DIR}/ty/CMakeLists.txt)

# ================================================================
#  Tests
# ================================================================

if(TITYOS_BUILD_TESTS)
    include(CTest)
    include(FetchContent)

    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
    )
    FetchContent_MakeAvailable(catch2)

    enable_testing()
    add_subdirectory(tests)
endif()