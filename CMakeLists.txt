cmake_minimum_required(VERSION 3.27)
project(agglo.cpp)

set(CMAKE_CXX_STANDARD 23)

add_executable(agglo.cpp main.cpp
        AgglomerativeClustering.h
)

#set(CMAKE_CXX_FLAGS "-O3 -march=native")

# Find Faiss library
find_library(FAISS_LIBRARY faiss REQUIRED)
if(FAISS_LIBRARY)
    target_link_libraries(agglo.cpp PUBLIC ${FAISS_LIBRARY})
endif()

# Find and link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(agglo.cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Find and link BLAS
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    target_link_libraries(agglo.cpp PUBLIC ${BLAS_LIBRARIES})
endif()