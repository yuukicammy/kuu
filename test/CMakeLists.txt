find_package(xtensor REQUIRED)
find_package(Boost REQUIRED)
enable_testing()
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
   message(FATAL_ERROR "No GTest")
else()
   message(STATUS "Found GTest")
endif()

set(SOURCE test_tensor.cpp test_function.cpp test_optimizer.cpp)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
#set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/clang++)

include_directories(${GTEST_INCLUDE_DIRS} ${KUU_INCLUDE_DIR} ${TBB_INCLUDE_DIRS})

add_executable(kuu_test main.cpp ${SOURCE})

if(MSVC)
    target_compile_options(kuu_test PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(kuu_test PRIVATE -march=native)
endif()

target_link_libraries(kuu_test kuu xtensor ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES})

add_test(AllTests kuu_test)
