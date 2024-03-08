cmake_minimum_required(VERSION 3.17)
project(hello_metal)

set(CMAKE_CXX_STANDARD 17)

# Path to your metal-cpp directory
set(METAL_CPP_DIR ${CMAKE_SOURCE_DIR}/metal-cpp)
set(METAL_CPP_EXT_DIR ${METAL_CPP_DIR}/metal-cpp-extensions)
# Add the include directories for metal-cpp and its dependencies
# (Metal-CPP, QuartzCore-CPP, and Foundation-CPP headers)
include_directories(${METAL_CPP_DIR})
include_directories(${METAL_CPP_DIR}/Foundation)
include_directories(${METAL_CPP_DIR}/Metal)
include_directories(${METAL_CPP_DIR}/MetalFX)
include_directories(${METAL_CPP_DIR}/QuartzCore)
include_directories(${METAL_CPP_EXT_DIR})
include_directories(${METAL_CPP_EXT_DIR}/AppKit)
include_directories(${METAL_CPP_EXT_DIR}/MetalKit)



# OPTIONAL: Find the system Metal and Foundation frameworks
# find_library(METAL_FRAMEWORK Metal)
# find_library(FOUNDATION_FRAMEWORK Foundation)
# find_library(QUARTZCORE_FRAMEWORK QuartzCore)

add_executable(hello_metal main.cpp
        metalChecks.cpp
        metalChecks.h
)

set(USER_FLAGS c++ c++abi)
# Link against the Metal and Foundation frameworks
target_link_libraries(hello_metal
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}  # If device can't be found; likely missing this linkage
    ${APPKIT_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${USER_FLAGS}
)