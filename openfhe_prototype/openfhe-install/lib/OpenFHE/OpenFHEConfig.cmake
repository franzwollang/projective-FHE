# - Config file for the OpenFHE package
# It defines the following variables
#  OpenFHE_INCLUDE_DIRS - include directories for OpenFHE
#  OpenFHE_LIBRARIES    - libraries to link against
get_filename_component(OpenFHE_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Our library dependencies (contains definitions for IMPORTED targets)
if(NOT OpenFHE_BINARY_DIR)
    include("${OpenFHE_CMAKE_DIR}/OpenFHETargets.cmake")
endif()

# These are IMPORTED targets created by OpenFHETargets.cmake
# set(OpenFHE_INCLUDE "${OpenFHE_CMAKE_DIR}/../../include/openfhe")
# set(OpenFHE_LIBDIR "${OpenFHE_CMAKE_DIR}/../../lib")
set(OpenFHE_INCLUDE "/Users/franzwollang/Desktop/Code/@distributedsystem/FHE/code/openfhe_prototype/openfhe-install/include/openfhe")
set(OpenFHE_LIBDIR "/Users/franzwollang/Desktop/Code/@distributedsystem/FHE/code/openfhe_prototype/openfhe-install/lib")
set(OpenFHE_LIBRARIES OPENFHEcore;OPENFHEpke;OPENFHEbinfhe  )
set(OpenFHE_STATIC_LIBRARIES   )
set(OpenFHE_SHARED_LIBRARIES OPENFHEcore;OPENFHEpke;OPENFHEbinfhe  )
set(BASE_OPENFHE_VERSION 1.3.1)

set(OPENMP_INCLUDES "")
set(OPENMP_LIBRARIES "")

set(OpenFHE_CXX_FLAGS " -Wall -Werror -O3  -DOPENFHE_VERSION=1.3.1  -DMATHBACKEND=4 -Wno-unknown-pragmas")
set(OpenFHE_C_FLAGS " -Wall -Werror -O3  -DOPENFHE_VERSION=1.3.1 -DMATHBACKEND=4 -Wno-unknown-pragmas")

set(OpenFHE_EXE_LINKER_FLAGS "")

# CXX info
set(OpenFHE_CXX_STANDARD "17")
set(OpenFHE_CXX_COMPILER_ID "AppleClang")
set(OpenFHE_CXX_COMPILER_VERSION "15.0.0.15000309")

# Build Options
set(OpenFHE_STATIC "OFF")
set(OpenFHE_SHARED "ON")
set(OpenFHE_TCM "OFF")
set(OpenFHE_NTL "OFF")
set(OpenFHE_OPENMP "OFF")
set(OpenFHE_NATIVE_SIZE "64")
set(OpenFHE_CKKS_M_FACTOR "1")
set(OpenFHE_NATIVEOPT "OFF")
set(OpenFHE_NOISEDEBUG "OFF")
set(OpenFHE_REDUCEDNOISE "OFF")

# Math Backend
set(OpenFHE_BACKEND "4")

# Build Details
set(OpenFHE_EMSCRIPTEN "")
set(OpenFHE_ARCHITECTURE "arm64")
set(OpenFHE_BACKEND_FLAGS_BASE "-DMATHBACKEND=4")

# Compile Definitions
if("ON")
    set(OpenFHE_BINFHE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_CORE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_PKE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_COMPILE_DEFINITIONS
        ${OpenFHE_BINFHE_COMPILE_DEFINITIONS}
        ${OpenFHE_CORE_COMPILE_DEFINITIONS}
        ${OpenFHE_PKE_COMPILE_DEFINITIONS})
endif()

if("OFF")
    set(OpenFHE_BINFHE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_CORE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_PKE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_COMPILE_DEFINITIONS_STATIC
        ${OpenFHE_BINFHE_COMPILE_DEFINITIONS_STATIC}
        ${OpenFHE_CORE_COMPILE_DEFINITIONS_STATIC}
        ${OpenFHE_PKE_COMPILE_DEFINITIONS_STATIC})
endif()
