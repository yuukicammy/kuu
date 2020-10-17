find_package(xtensor REQUIRED)

add_executable(main main.cpp)

target_include_directories(main PUBLIC ${KUU_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/third_party ${TBB_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
#set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/clang++)

if(MSVC)
    target_compile_options(main PRIVATE /EHsc /MP /bigobj "/std:c++17" )
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(main PRIVATE -march=native)
endif()

target_link_libraries(main xtensor kuu ${TBB_LIBRARIES})


add_executable(mnist mnist.cpp)

target_include_directories(mnist PUBLIC ${KUU_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/third_party ${TBB_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(mnist PRIVATE /EHsc /MP /bigobj "/std:c++17" )
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(mnist PRIVATE -march=native)
endif()

target_link_libraries(mnist xtensor kuu ${TBB_LIBRARIES})