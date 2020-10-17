#find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
#find_package(blas REQUIRED)
#find_package(lapack REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(Boost REQUIRED)

# if xtensor was built with xsimd support:
find_package(xsimd REQUIRED)

set(INCLUDES ${KUU_INCLUDE_DIR})
set(SOURCE graph.cpp)

add_library(kuu STATIC ${SOURCE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
#set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/clang++)

if(MSVC)
    target_compile_options(kuu PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(kuu PRIVATE -march=native)
endif()

target_link_libraries(kuu xtensor xtensor-blas ${TBB_LIBRARIES})
target_include_directories(kuu PRIVATE ${TBB_INCLUDE_DIRS} ${KUU_INCLUDE_DIR})
