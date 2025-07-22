cmake_minimum_required(VERSION 3.12)
project(openfhe_prototype_syntax_check)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wno-unused-parameter -DMOCK_OPENFHE")

# Add include directory for mock
include_directories(.)

# Find Eigen3 for pseudoinverse computation
find_package(Eigen3 REQUIRED)

# Create a simple syntax test executable
add_executable(syntax_test 
    utils.cpp 
    syntax_test.cpp
)
target_link_libraries(syntax_test Eigen3::Eigen)

# Add shared library for utils and core classes (mock mode)
add_library(openfhe_utils SHARED 
    utils.cpp 
    encoding.cpp 
    projection.cpp 
    monitor.cpp
)
target_link_libraries(openfhe_utils Eigen3::Eigen)

# Add workload_100_cycles executable (mock mode)
add_executable(workload_100_cycles workload_100_cycles.cpp)
target_link_libraries(workload_100_cycles openfhe_utils)

# Add pipeline executable (mock mode)  
add_executable(pipeline pipeline.cpp)
target_link_libraries(pipeline openfhe_utils)
target_compile_definitions(pipeline PRIVATE BUILD_AS_LIBRARY=0)

# Add simple workload test (mock mode)
add_executable(simple_workload_test simple_workload_test.cpp)
# No additional libraries needed for this simplified test 