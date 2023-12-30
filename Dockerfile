# Use the latest Ubuntu image as the base
FROM ubuntu:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget make g++ intel-mkl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CMAKE_VERSION=3.28.1
ENV CMAKE_BIN=/app/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar zxvf cmake.tar.gz && \
    rm cmake.tar.gz

ADD faiss faiss
RUN ${CMAKE_BIN} -B faiss/build faiss -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 && \
    make -C faiss/build -j faiss && \
    make -C faiss/build install

ADD agglo.cpp agglo.cpp
RUN mkdir agglo.cpp/build && \
    ${CMAKE_BIN} -B agglo.cpp/build agglo.cpp && \
    make -C agglo.cpp/build

# Add glove text file
ADD glove/glove.6B.100d.txt glove.txt

ENTRYPOINT ["/app/agglo.cpp/build/agglo.cpp"]
