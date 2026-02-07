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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DEBUG_POSTFIX debug)

if(CMAKE_SYSTEM_NAME MATCHES "CYGWIN" OR CMAKE_SYSTEM_NAME MATCHES "MSYS" OR CMAKE_SYSTEM_NAME MATCHES "MINGW")
set(CMAKE_CXX_EXTENSIONS ON)
endif()


# ================================================================
# Build Options
# ================================================================

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
option(TITYOS_ENABLE_PERFORMANCE_TESTS "Enable performance tests" OFF)

# CUDA options
option(TITYOS_BUILD_CUDA "Build to allow cuda" OFF)


# ================================================================
#  Library
# ================================================================

include(CheckCXXCompilerFlag)

add_library(tityos SHARED)
add_library(tityos_cpu SHARED)
target_link_libraries(tityos_cpu PRIVATE tityos)

# TODO: Change this later
set_target_properties(tityos PROPERTIES INSTALL_RPATH "$ORIGIN")

target_include_directories(tityos PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(tityos_cpu PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Compiler flags
if(MSVC)
    target_compile_options(tityos PRIVATE /O2)
    target_compile_options(tityos_cpu PRIVATE /O2)
else()
    target_compile_options(tityos PRIVATE -O2)
    target_compile_options(tityos_cpu PRIVATE -O2)
endif()


# CUDA
if (TITYOS_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    message(STATUS "Building with CUDA")

    add_library(tityos_cuda SHARED)
    target_link_libraries(tityos_cuda PRIVATE tityos)

    set_target_properties(tityos_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tityos_cuda PUBLIC CUDA::cudart)
    target_include_directories(tityos_cuda PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>
    )
endif()

# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(tityos_cpu PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Checking for OpenMP - Success")
else()
    message(FATAL_ERROR "Checking for OpenMP - Failed")
endif()

# CPU SIMD
if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_ALLOWS_AVX2)
    if (COMPILER_ALLOWS_AVX2)
        target_compile_options(tityos_cpu PRIVATE /arch:AVX2)
    endif()
else()
    check_cxx_compiler_flag("-mavx2" COMPILER_ALLOWS_AVX2)
    if (COMPILER_ALLOWS_AVX2)
        target_compile_options(tityos_cpu PRIVATE -mavx2 -mfma)
    endif()
endif()

if(NOT COMPILER_ALLOWS_AVX2)
    message(FATAL_ERROR "Checking for AVX2 support - Failed")
else()
    message(STATUS "Checking for AVX2 support - Success")
endif()

# Warnings
if(TITYOS_BUILD_WARNINGS)
    if(MSVC)
        target_compile_options(tityos PRIVATE /W4 /WX)
        target_compile_options(tityos_cpu PRIVATE /W4 /WX)
    else()
        target_compile_options(tityos PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(tityos_cpu PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(TITYOS_BUILD_CUDA AND TITYOS_BUILD_WARNINGS)
    target_compile_options(tityos_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall
            -Xcompiler=-Wextra
            -Xcompiler=-Wpedantic
            -Xcompiler=-Werror
        >
    )

    target_compile_options(tityos_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --Werror=all-warnings
        >
    )
endif()

# Sanitizer
if(TITYOS_SANITIZE_ADDRESS)
    target_compile_options(tityos PRIVATE -fsanitize=address -g)
    target_link_options(tityos PRIVATE -fsanitize=address)
elseif(TITYOS_SANITIZE_MEMORY)
    target_compile_options(tityos PRIVATE -fsanitize=memory -g)
    target_link_options(tityos PRIVATE -fsanitize=memory)
endif()

# TODO: Change this so hidden by default, need to add TITYOS_API
# Hiding for non Win32
# if(NOT WIN32)
#     set_target_properties(tityos PROPERTIES
#         CXX_VISIBILITY_PRESET "hidden"
#         VISIBILITY_INLINES_HIDDEN TRUE
#     )

#     set_target_properties(tityos_cpu PROPERTIES
#         CXX_VISIBILITY_PRESET "hidden"
#         VISIBILITY_INLINES_HIDDEN TRUE
#     )
# endif()

include(${TY_ROOT_DIR}/ty/CMakeLists.txt)

# ================================================================
#  Install
# ================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(INSTALL_TARGETS tityos tityos_cpu)
if(TITYOS_BUILD_CUDA)
    list(APPEND INSTALL_TARGETS tityos_cuda)
endif()

install(TARGETS ${INSTALL_TARGETS}
    EXPORT TityosTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tityos/"
    DESTINATION include/tityos
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT TityosTargets
    FILE TityosTargets.cmake
    NAMESPACE tityos::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tityos
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tityosConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tityosConfig.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(OpenMP)
include(\"\${CMAKE_CURRENT_LIST_DIR}/TityosTargets.cmake\")
")

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tityosConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tityosConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tityos
)

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

