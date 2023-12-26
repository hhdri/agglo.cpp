cmake_minimum_required(VERSION 3.27)
project(untitled2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(untitled2 main2.cpp)

include_directories(/usr/local/include)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(untitled2 PUBLIC OpenMP::OpenMP_CXX)
endif()

# Find and link BLAS
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    target_link_libraries(untitled2 PUBLIC ${BLAS_LIBRARIES})
endif()

# Find Faiss library
find_library(FAISS_LIBRARY REQUIRED)
if(FAISS_LIBRARY)
    target_link_libraries(untitled2 PUBLIC ${FAISS_LIBRARY})
endif()