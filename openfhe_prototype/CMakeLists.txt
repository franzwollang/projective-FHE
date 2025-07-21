cmake_minimum_required(VERSION 3.5.1)
project(OpenFHE_Prototype)

# Build configuration options
option(ENABLE_DIAGNOSTICS "Build with NoiseMonitor and verbose logging (Debug mode)" ON)
option(USE_OPENFHE_OMP "Enable OpenFHE OpenMP support for performance" OFF)
option(USE_OPENFHE_GPU "Link against OpenFHE CUDA GPU backend" OFF)

# If GPU requested, require CUDA toolkit
if(USE_OPENFHE_GPU)
    find_package(CUDA REQUIRED)
    if (CUDA_FOUND)
        message(STATUS "CUDA toolkit found: ${CUDA_VERSION}")
        add_compile_definitions(OPENFHE_USE_GPU=1)
    else()
        message(FATAL_ERROR "CUDA toolkit not found but USE_OPENFHE_GPU=ON")
    endif()
endif()

# Set default build type to Release for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Diagnostics enabled: ${ENABLE_DIAGNOSTICS}")
message(STATUS "OpenMP enabled: ${USE_OPENFHE_OMP}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find system-wide OpenFHE first, fallback to local installation
find_package(OpenFHE)

if (OpenFHE_FOUND)
    message(STATUS "OpenFHE found")
    # Use system-wide OpenFHE includes and libraries
    set(OPENFHE_INCLUDES ${OPENFHE_INCLUDE})
    set(OPENFHE_LIBS ${OPENFHE_SHARED_LIBRARIES})
    
    # Debug: Print OpenFHE include directories
    message(STATUS "OpenFHE include directories: ${OPENFHE_INCLUDE}")
    message(STATUS "OpenFHE libraries: ${OPENFHE_SHARED_LIBRARIES}")
    
    # Configure OpenFHE with OpenMP if requested
    if(USE_OPENFHE_OMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "OpenMP found and enabled")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        else()
            message(WARNING "OpenMP requested but not found")
        endif()
    endif()
else()
    # Fallback to local installation if system-wide not found
    set(OPENFHE_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openfhe-install")
    if(EXISTS "${OPENFHE_INSTALL_DIR}/include/openfhe/openfhe.h")
        message(STATUS "Using local OpenFHE installation")
        set(OPENFHE_INCLUDES 
            ${OPENFHE_INSTALL_DIR}/include
            ${OPENFHE_INSTALL_DIR}/include/openfhe
            ${OPENFHE_INSTALL_DIR}/include/openfhe/core
            ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
            ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe)
        link_directories(${OPENFHE_INSTALL_DIR}/lib)
        set(OPENFHE_LIBS OPENFHEpke OPENFHEcore)
    else()
        message(STATUS "OpenFHE not found, using mock implementation")
        add_compile_definitions(MOCK_OPENFHE=1)
        set(OPENFHE_INCLUDES "")
        set(OPENFHE_LIBS "")
    endif()
endif()

# Configure diagnostics mode
if(ENABLE_DIAGNOSTICS)
    add_compile_definitions(ENABLE_DIAGNOSTICS=1)
    message(STATUS "Diagnostics mode: Full instrumentation enabled")
else()
    message(STATUS "Production mode: Diagnostics disabled for maximum performance")
endif()

# Find Eigen3 for SVD-based pseudoinverse
find_package(Eigen3 REQUIRED)

# Compiler flags for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Set up RPATH for runtime library loading
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR};/usr/local/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add shared library for utils and core classes
add_library(openfhe_utils SHARED utils.cpp encoding.cpp projection.cpp monitor.cpp)
target_include_directories(openfhe_utils PRIVATE ${OPENFHE_INCLUDES})
target_link_libraries(openfhe_utils OPENFHEpke OPENFHEcore Eigen3::Eigen)

# Add executables
add_executable(params params.cpp)
target_include_directories(params PRIVATE ${OPENFHE_INCLUDES})
target_link_libraries(params openfhe_utils OPENFHEpke OPENFHEcore)

# encoding and projection are now part of openfhe_utils library

# Pipeline executable
# Build FHEPipeline as object library to avoid duplicate main symbols
add_library(fhe_pipeline OBJECT pipeline.cpp)
target_include_directories(fhe_pipeline PRIVATE ${OPENFHE_INCLUDES})
target_compile_definitions(fhe_pipeline PRIVATE BUILD_AS_LIBRARY=1)

# Pipeline standalone executable
add_executable(pipeline pipeline.cpp)
target_include_directories(pipeline PRIVATE ${OPENFHE_INCLUDES})
target_link_libraries(pipeline openfhe_utils OPENFHEpke OPENFHEcore
    $<$<BOOL:${USE_OPENFHE_GPU}>:OPENFHEgpu>
)
target_compile_definitions(pipeline PRIVATE BUILD_AS_LIBRARY=0)

# Parity test
add_executable(test_parity test_parity.cpp)
target_include_directories(test_parity PRIVATE ${OPENFHE_INCLUDES})
target_link_libraries(test_parity openfhe_utils OPENFHEpke OPENFHEcore Eigen3::Eigen)

# Syntax test (can use mock)
add_executable(syntax_test syntax_test.cpp)
target_link_libraries(syntax_test openfhe_utils)

# Benchmark modes
add_executable(benchmark_modes benchmark_modes.cpp $<TARGET_OBJECTS:fhe_pipeline>)
target_include_directories(benchmark_modes PRIVATE ${OPENFHE_INCLUDES})
target_link_libraries(benchmark_modes openfhe_utils OPENFHEpke OPENFHEcore
    $<$<BOOL:${USE_OPENFHE_GPU}>:OPENFHEgpu>
) 