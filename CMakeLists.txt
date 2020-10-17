cmake_minimum_required(VERSION 3.1)
project(kuu)

set(KUU_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/kuu/include)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
find_package(TBB REQUIRED)
message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")

add_subdirectory(kuu/src)
add_subdirectory(examples)
add_subdirectory(test)