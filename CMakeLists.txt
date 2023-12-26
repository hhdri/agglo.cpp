cmake_minimum_required(VERSION 3.27)
project(untitled2)

set(CMAKE_CXX_STANDARD 23)

add_executable(untitled2 main2.cpp)

include_directories(/usr/local/include)

#set(CMAKE_CXX_FLAGS "-O3 -march=native")

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