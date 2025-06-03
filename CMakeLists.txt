cmake_minimum_required(VERSION 3.10)
project(rcu-lab VERSION 1.0.0 LANGUAGES CXX)

# Project description
set(PROJECT_DESCRIPTION "A NUMA-aware microbenchmark suite for real-world RCU analysis")
set(PROJECT_HOMEPAGE_URL "https://github.com/oat-im/rcu-lab")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Find liburcu using pkg-config
pkg_check_modules(URCU REQUIRED liburcu)

# Find libnuma (optional on non-Linux)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(NUMA_INCLUDE_DIR numa.h
        HINTS /usr/include /usr/local/include
    )
    
    find_library(NUMA_LIBRARY
        NAMES numa
        HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    )
    
    if(NOT NUMA_INCLUDE_DIR OR NOT NUMA_LIBRARY)
        message(FATAL_ERROR "libnuma not found. Please install:\n"
                            "  Ubuntu/Debian: apt-get install libnuma-dev\n"
                            "  Fedora/RHEL: dnf install numactl-devel")
    endif()
    set(NUMA_FOUND TRUE)
else()
    message(WARNING "NUMA support disabled on non-Linux systems")
    set(NUMA_FOUND FALSE)
endif()

# Main executable
add_executable(rcu-lab src/rcu_lab.cpp)

# Include directories
target_include_directories(rcu-lab PRIVATE 
    $<$<BOOL:${NUMA_FOUND}>:${NUMA_INCLUDE_DIR}>
)

# Link libraries
target_link_libraries(rcu-lab PRIVATE
    ${URCU_LIBRARIES}
    $<$<BOOL:${NUMA_FOUND}>:${NUMA_LIBRARY}>
    Threads::Threads
)

# Compile flags from pkg-config
target_compile_options(rcu-lab PRIVATE ${URCU_CFLAGS_OTHER})

# Compile definitions
target_compile_definitions(rcu-lab PRIVATE
    $<$<BOOL:${NUMA_FOUND}>:HAS_NUMA>
)

# Additional tools
option(BUILD_TOOLS "Build analysis tools" ON)
if(BUILD_TOOLS)
    # Latency plotter (Python script)
    configure_file(tools/plot_latencies.py 
                   ${CMAKE_BINARY_DIR}/plot_latencies.py 
                   COPYONLY)
    
    # Stats analyzer
    configure_file(tools/analyze_stats.py 
                   ${CMAKE_BINARY_DIR}/analyze_stats.py 
                   COPYONLY)
endif()

# Installation
install(TARGETS rcu-lab
        RUNTIME DESTINATION bin)

install(FILES README.md LICENSE
        DESTINATION share/doc/rcu-lab)

if(BUILD_TOOLS)
    install(PROGRAMS 
            ${CMAKE_BINARY_DIR}/plot_latencies.py
            ${CMAKE_BINARY_DIR}/analyze_stats.py
            DESTINATION bin)
endif()

# CPack configuration
set(CPACK_PACKAGE_NAME "rcu-lab")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_CONTACT "eric@oat.im")
set(CPACK_GENERATOR "TGZ;DEB;RPM")
include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "RCU-Lab Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message(STATUS "  liburcu: ${URCU_LIBRARIES}")
message(STATUS "  NUMA support: ${NUMA_FOUND}")
if(NUMA_FOUND)
    message(STATUS "  libnuma: ${NUMA_LIBRARY}")
endif()
message(STATUS "")