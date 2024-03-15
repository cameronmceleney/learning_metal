################################################################

# Start of CMakeLists.txt

################################################################

cmake_minimum_required(VERSION 3.17)
project(hello_metal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(OBJCXX)

################################################################

# Path to your metal-cpp directory
set(METAL_CPP_DIR ${CMAKE_SOURCE_DIR}/lib/metal-cpp)
set(METAL_CPP_EXT_DIR ${METAL_CPP_DIR}/metal-cpp-extensions)

################################################################

# Add the include directories for metal-cpp and its dependencies
# (Metal-CPP, QuartzCore-CPP, and Foundation-CPP headers)
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${METAL_CPP_DIR})
include_directories(${METAL_CPP_DIR}/Foundation)
include_directories(${METAL_CPP_DIR}/Metal)
include_directories(${METAL_CPP_DIR}/MetalFX)
include_directories(${METAL_CPP_DIR}/QuartzCore)
include_directories(${METAL_CPP_DIR})
include_directories(${METAL_CPP_DIR}/AppKit)
include_directories(${METAL_CPP_DIR}/MetalKit)

# OPTIONAL: Find the system Metal and Foundation frameworks
# find_library(METAL_FRAMEWORK Metal)
# find_library(FOUNDATION_FRAMEWORK Foundation)
# find_library(QUARTZCORE_FRAMEWORK QuartzCore)

################################################################
# Create lists of all sources
set(PROJECTS_DIR ${CMAKE_SOURCE_DIR}/src/projects)

set(USER_GENERAL_LIBRARIES
        lib/config.h
)

set(METAL_CHECKS_EXAMPLES
        ${PROJECTS_DIR}/checks_examples/check_for_metal_device.cpp
        ${PROJECTS_DIR}/checks_examples/check_for_metal_device.h
)

set(GRAPHICAL_EXAMPLES
        ${PROJECTS_DIR}/graphical_implementation_example/graphical_example_m.cpp
        ${PROJECTS_DIR}/graphical_implementation_example/graphical_example_m.h
        ${PROJECTS_DIR}/graphical_implementation_example/control/app_delegate.cpp
        ${PROJECTS_DIR}/graphical_implementation_example/control/app_delegate.h
        ${PROJECTS_DIR}/graphical_implementation_example/control/view_delegate.cpp
        ${PROJECTS_DIR}/graphical_implementation_example/control/view_delegate.h
        ${PROJECTS_DIR}/graphical_implementation_example/view/renderer.cpp
        ${PROJECTS_DIR}/graphical_implementation_example/view/renderer.h
)

set(COMPUTE_FUNCTION_EXAMPLES
        ${PROJECTS_DIR}/compute_function_examples/compute_function_examples.cpp
        ${PROJECTS_DIR}/compute_function_examples/compute_function_examples.h
        src/projects/Small_test_compute/ArrayAdder.mm
        src/projects/Small_test_compute/ArrayAdder.h
        #src/projects/compute_function_examples/00-window.cpp
)

################################################################

# Links all user sources to the executable
add_executable(${PROJECT_NAME} main.cpp
        ${USER_GENERAL_LIBRARIES}
        ${METAL_CHECKS_EXAMPLES}
        ${GRAPHICAL_EXAMPLES}
        ${COMPUTE_FUNCTION_EXAMPLES}
)

################################################################
# NEW. for create lists of all sources
# Print the CMAKE_CURRENT_BINARY_DIR variable
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

# Path to Metal shader source file
set(METAL_SHADER_SRC ${PROJECTS_DIR}/Small_test_compute/addition.metal)

# Output path for the compiled Metal shader (AIR file)
set(METAL_SHADER_AIR ${CMAKE_CURRENT_BINARY_DIR}/addition.air)

# Output path for the Metal library (.metallib file)
set(METAL_SHADER_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/addition.metallib)
add_definitions(-DMETAL_SHADER_METALLIB_PATH="${METAL_SHADER_METALLIB}")

################################################################
# NEW. Custom commands

# Compile .metal shader to .air
add_custom_command(
    OUTPUT ${METAL_SHADER_AIR}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_SHADER_AIR}
    DEPENDS ${METAL_SHADER_SRC}
    COMMENT "Compiling ${METAL_SHADER_SRC} to AIR"
)

# Compile .air to .metallib
add_custom_command(
    OUTPUT ${METAL_SHADER_METALLIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_SHADER_AIR} -o ${METAL_SHADER_METALLIB}
    DEPENDS ${METAL_SHADER_AIR}
    COMMENT "Compiling ${METAL_SHADER_AIR} to Metal Library"
)

################################################################
# NEW. Custom target to ensure the Metal shader is compiled
add_custom_target(
    CompileMetalShader ALL
    DEPENDS ${METAL_SHADER_METALLIB}
)

################################################################
# NEW. Specify that the executable depends on the CompileMetalShader target
add_dependencies(${PROJECT_NAME} CompileMetalShader)

################################################################
# Include and link directories for third-party libraries

set(USER_FLAGS c++ c++abi)

# Link against the Metal and Foundation frameworks
target_link_libraries(${PROJECT_NAME}
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALFX_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}  # If device can't be found; likely missing this linkage
    ${APPKIT_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${USER_FLAGS}
)

#set_target_properties(hello_metal PROPERTIES
#    LINK_FLAGS "-framework Metal -framework Foundation"
#)

################################################################

# Optional: Uncomment for optimisation (-O2 and -O3 being most common for me)
# target_compile_options(SpinChains PRIVATE -O2)

################################################################

# End of CMakeLists.txt

################################################################