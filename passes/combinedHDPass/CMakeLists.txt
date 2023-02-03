cmake_minimum_required(VERSION 3.1)

project( CombinedHostDevicePass )

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(LLVM_ENABLE_PLUGINS ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inc/pathToConditionAnalyzer.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/pathToConditionAnalyzer.h)

add_custom_command( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h
        COMMAND clang++ -S -emit-llvm -O3
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/hostRuntime.cpp -o hostRuntime.ll
        -I${CMAKE_CURRENT_SOURCE_DIR}/inc
        COMMAND xxd -i hostRuntime.ll > ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/hostRuntime.cpp )
add_custom_target( hostRuntime DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h)

set( MEKONG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../mekong-utils" )
add_subdirectory( "${MEKONG_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/build/mekong-utils" )

add_library(${PROJECT_NAME} MODULE
    # List your source files here.
    lib/hostTransformer.cpp
    lib/deviceAnalyzer.cpp
    lib/registerPasses.cpp
    lib/recurrence.cpp
    lib/traceback.cpp
    lib/kernelinfo.cpp
    lib/node.cpp
    lib/kernellaunch.cpp
)

target_include_directories( ${PROJECT_NAME} PRIVATE inc "${MEKONG_PATH}/inc")
add_dependencies( ${PROJECT_NAME} hostRuntime )

target_link_libraries( ${PROJECT_NAME} PRIVATE mekong-utils )


# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(${PROJECT_NAME} PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
 POSITION_INDEPENDENT_CODE ON)
