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
    # Use comprehensive library list for proper linking
    set(OPENFHE_LIBS ${OpenFHE_SHARED_LIBRARIES})
    if(NOT OPENFHE_LIBS)
        # Fallback to manual library specification for complete linking
        set(OPENFHE_LIBS "OPENFHEcore;OPENFHEpke;OPENFHEbinfhe")
        message(STATUS "Using manual OpenFHE library list: ${OPENFHE_LIBS}")
    endif()
    
    # Debug: Print OpenFHE include directories
    message(STATUS "OpenFHE include directories: ${OPENFHE_INCLUDE}")
    message(STATUS "OpenFHE shared libraries: ${OpenFHE_SHARED_LIBRARIES}")
    message(STATUS "Final OPENFHE_LIBS: ${OPENFHE_LIBS}")
    
    # Ensure we have the correct include directories with all OpenFHE modules
    if(NOT OPENFHE_INCLUDES)
        set(OPENFHE_INCLUDES "/usr/local/include;/usr/local/include/openfhe;/usr/local/include/openfhe/core;/usr/local/include/openfhe/pke;/usr/local/include/openfhe/binfhe")
        message(STATUS "Using complete OpenFHE include paths: ${OPENFHE_INCLUDES}")
    else()
        # Add all openfhe subdirectories to include path
        list(APPEND OPENFHE_INCLUDES "/usr/local/include/openfhe")
        list(APPEND OPENFHE_INCLUDES "/usr/local/include/openfhe/core")
        list(APPEND OPENFHE_INCLUDES "/usr/local/include/openfhe/pke")
        list(APPEND OPENFHE_INCLUDES "/usr/local/include/openfhe/binfhe")
        message(STATUS "Enhanced complete include paths: ${OPENFHE_INCLUDES}")
    endif()
    
    # Add global include directories for all targets
    foreach(INCLUDE_PATH ${OPENFHE_INCLUDES})
        include_directories(${INCLUDE_PATH})
    endforeach()
    
    # Add OpenFHE library directories
    if(OpenFHE_LIBDIR)
        link_directories(${OpenFHE_LIBDIR})
        message(STATUS "Added OpenFHE library directory: ${OpenFHE_LIBDIR}")
    else()
        link_directories("/usr/local/lib")
        message(STATUS "Using default library directory: /usr/local/lib")
    endif()
    
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
if(OpenFHE_FOUND)
    target_link_libraries(openfhe_utils ${OPENFHE_LIBS} Eigen3::Eigen)
else()
    target_link_libraries(openfhe_utils Eigen3::Eigen)
endif()

# Add executables
add_executable(params params.cpp)
target_include_directories(params PRIVATE ${OPENFHE_INCLUDES})
if(OpenFHE_FOUND)
    target_link_libraries(params openfhe_utils ${OPENFHE_LIBS})
else()
    target_link_libraries(params openfhe_utils)
endif()

# Pipeline executable
# Build FHEPipeline as object library to avoid duplicate main symbols
add_library(fhe_pipeline OBJECT pipeline.cpp)
target_include_directories(fhe_pipeline PRIVATE ${OPENFHE_INCLUDES})
if(OpenFHE_FOUND)
    target_link_libraries(fhe_pipeline ${OPENFHE_LIBS})
endif()
target_compile_definitions(fhe_pipeline PRIVATE BUILD_AS_LIBRARY=1)

# Pipeline standalone executable
add_executable(pipeline pipeline.cpp)
target_include_directories(pipeline PRIVATE ${OPENFHE_INCLUDES})
if(OpenFHE_FOUND)
    target_link_libraries(pipeline openfhe_utils ${OPENFHE_LIBS})
    # Only link GPU library if it actually exists
    if(USE_OPENFHE_GPU)
        find_library(OPENFHE_GPU_LIB OPENFHEgpu PATHS /usr/local/lib)
        if(OPENFHE_GPU_LIB)
            target_link_libraries(pipeline ${OPENFHE_GPU_LIB})
            message(STATUS "Linking with OpenFHE GPU library: ${OPENFHE_GPU_LIB}")
        else()
            message(WARNING "GPU support requested but OPENFHEgpu library not found - building without GPU")
        endif()
    endif()
else()
    target_link_libraries(pipeline openfhe_utils)
endif()
target_compile_definitions(pipeline PRIVATE BUILD_AS_LIBRARY=0)

# Parity test
add_executable(test_parity test_parity.cpp)
target_include_directories(test_parity PRIVATE ${OPENFHE_INCLUDES})
if(OpenFHE_FOUND)
    target_link_libraries(test_parity openfhe_utils ${OPENFHE_LIBS} Eigen3::Eigen)
else()
    target_link_libraries(test_parity openfhe_utils Eigen3::Eigen)
endif()

# Syntax test (can use mock)
add_executable(syntax_test syntax_test.cpp)
target_link_libraries(syntax_test openfhe_utils)

# Benchmark modes
add_executable(benchmark_modes benchmark_modes.cpp $<TARGET_OBJECTS:fhe_pipeline>)
target_include_directories(benchmark_modes PRIVATE ${OPENFHE_INCLUDES})
if(OpenFHE_FOUND)
    target_link_libraries(benchmark_modes openfhe_utils ${OPENFHE_LIBS})
    # Only link GPU library if it actually exists
    if(USE_OPENFHE_GPU)
        find_library(OPENFHE_GPU_LIB OPENFHEgpu PATHS /usr/local/lib)
        if(OPENFHE_GPU_LIB)
            target_link_libraries(benchmark_modes ${OPENFHE_GPU_LIB})
            message(STATUS "Linking benchmark_modes with OpenFHE GPU library: ${OPENFHE_GPU_LIB}")
        else()
            message(WARNING "GPU support requested but OPENFHEgpu library not found - building benchmark without GPU")
        endif()
    endif()
else()
    target_link_libraries(benchmark_modes openfhe_utils)
endif() 