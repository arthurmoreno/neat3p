cmake_minimum_required(VERSION 3.15)

project(neat3p LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Basic Setup
# ------------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# ------------------------------------------------------------------------------
# Python and nanobind Setup
# ------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(NANOBIND_DIR "${CMAKE_SOURCE_DIR}/libs/nanobind")
include_directories(${NANOBIND_DIR}/include)
add_subdirectory(${NANOBIND_DIR})

# ------------------------------------------------------------------------------
# Project Setup
# ------------------------------------------------------------------------------

project(Neat3pProject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# ------------------------------------------------------------------------------
# Build Options & Flags
# ------------------------------------------------------------------------------
# Example optimization flags
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared libraries" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -ffast-math")

# Colored diagnostics in GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fdiagnostics-color=always)
endif()

# Example definition (remove if not needed)
add_compile_definitions(ENTT_ENTITY_TYPE=int)

# Hide internal symbols by default
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -fvisibility=hidden")

# ------------------------------------------------------------------------------
# Required Packages
# ------------------------------------------------------------------------------
set(nanobind_DIR "${CMAKE_SOURCE_DIR}/libs/nanobind/cmake")
find_package(nanobind CONFIG REQUIRED)

find_package(msgpack REQUIRED)

find_package(spdlog REQUIRED)
message(STATUS "→ spdlog version:     ${spdlog_VERSION}")

# find_package(nanobind CONFIG REQUIRED)

find_package(
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# ------------------------------------------------------------------------------
# FlatBuffers Setup
# ------------------------------------------------------------------------------
set(FLATBUFFERS_SRC_DIR "${CMAKE_SOURCE_DIR}/libs/flatbuffers")
add_subdirectory(
    ${FLATBUFFERS_SRC_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
    EXCLUDE_FROM_ALL
)

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${NANOBIND_DIR}/include
    ${MSGPACK_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libs
)

# ------------------------------------------------------------------------------
# FlatBuffers Schema Compilation
# ------------------------------------------------------------------------------
find_program(FLATC_COMPILER flatc REQUIRED)
if(NOT FLATC_COMPILER)
    message(FATAL_ERROR "flatc compiler not found. Please ensure it is installed and in your PATH.")
endif()

# Example schema list. Adjust or remove if not needed.
set(FLATBUFFERS_SCHEMAS
    ${CMAKE_SOURCE_DIR}/schemas/Gene.fbs
)

set(FLATBUFFERS_GENERATED_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${FLATBUFFERS_GENERATED_CPP_DIR})

set(FLATBUFFERS_GENERATED_CPP_HEADERS)

foreach(schema ${FLATBUFFERS_SCHEMAS})
    get_filename_component(schema_name ${schema} NAME_WE)
    add_custom_command(
        OUTPUT ${FLATBUFFERS_GENERATED_CPP_DIR}/${schema_name}.h
        COMMAND ${FLATC_COMPILER} --cpp --gen-mutable -o ${FLATBUFFERS_GENERATED_CPP_DIR} ${schema}
        DEPENDS ${schema}
        COMMENT "Compiling FlatBuffers schema ${schema} to C++"
    )
    list(APPEND FLATBUFFERS_GENERATED_CPP_HEADERS ${FLATBUFFERS_GENERATED_CPP_DIR}/${schema_name}.h)
endforeach()

add_custom_target(GenerateFlatBuffers ALL DEPENDS ${FLATBUFFERS_GENERATED_CPP_HEADERS})
include_directories(${FLATBUFFERS_GENERATED_CPP_DIR})

# ------------------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------------------
# Collect all .cpp files in neat3p/, excluding the ones you don’t want.
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/neat3p.cpp")
# list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/neat3p/my_module.cpp")

# ------------------------------------------------------------------------------
# Build the Python Module (nanobind)
# ------------------------------------------------------------------------------
nanobind_add_module(
    _neat3p
    MODULE
    ${SOURCES}
    src/neat3p.cpp
)

target_link_libraries(_neat3p PRIVATE
    ${MSGPACK_LIBRARIES}
    ${Python3_LIBRARIES}
    flatbuffers
    spdlog::spdlog
)

# Ensure schema generation runs before building the module
add_dependencies(_neat3p GenerateFlatBuffers)

# ------------------------------------------------------------------------------
# Install directives
# ------------------------------------------------------------------------------
# Install the compiled Python extension module into the neat3p package directory
install(TARGETS _neat3p LIBRARY DESTINATION neat3p)

# Install the umbrella header for C++ users.
# Adjust the path to neat3p.hpp if it's located somewhere else.
install(FILES ${CMAKE_SOURCE_DIR}/src/neat3p.hpp DESTINATION include/neat3p)

# ------------------------------------------------------------------------------
# Final Message
# ------------------------------------------------------------------------------
message(STATUS "Build Configuration for '_neat3p' Complete")