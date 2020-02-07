project(KNN_JNI)

cmake_minimum_required(VERSION 3.6.0)

set(KNN_JNI_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -march=x86-64 -Wall -pedantic -mavx -msse4 -fvisibility=hidden")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

include_directories(includes)

set(SOURCE_FILES src_cpp/knn_api.cpp)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $ENV{CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $ENV{CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_library(KNN_OBJ OBJECT ${SOURCE_FILES})
add_library(KNN_JNI SHARED ${SOURCE_FILES})

target_link_libraries(KNN_JNI)
