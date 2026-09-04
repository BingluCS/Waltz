cmake_minimum_required(VERSION 3.21)

project(
    Waltz
    VERSION 0.1.0
    DESCRIPTION "GPU lossy compression for 3D scientific floating-point fields"
    LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build Waltz as a shared library" ON)
option(WALTZ_BUILD_EXAMPLES "Build the command-line driver and tuner" ON)
option(WALTZ_BUILD_TOOLS "Build optional research/diagnostic tools" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(waltz_shared src/waltz.cu src/lossless/lc.cu src/lossless/wzp.cu)
add_library(Waltz::waltz ALIAS waltz_shared)

set_target_properties(waltz_shared PROPERTIES OUTPUT_NAME waltz EXPORT_NAME waltz)
target_compile_features(waltz_shared PUBLIC cxx_std_17 PRIVATE cuda_std_17)
target_compile_definitions(
    waltz_shared
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
    waltz_shared
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda;--expt-relaxed-constexpr;-Wno-deprecated-declarations>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_include_directories(
    waltz_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/lossless/lc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    waltz_shared
    PUBLIC CUDA::cudart
    PRIVATE CUDA::cuda_driver CUDA::cublas)

if(WALTZ_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(WALTZ_BUILD_TOOLS)
    add_executable(wzp_gpu_bench tools/wzp_gpu_bench.cu)
    target_link_libraries(wzp_gpu_bench PRIVATE Waltz::waltz CUDA::cudart)
endif()

install(
    TARGETS waltz_shared
    EXPORT WaltzTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
    cmake/WaltzConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/WaltzConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Waltz)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/WaltzConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    EXPORT WaltzTargets
    FILE WaltzTargets.cmake
    NAMESPACE Waltz::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Waltz)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/WaltzConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/WaltzConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Waltz)
