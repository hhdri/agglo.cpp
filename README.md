# agglo.cpp
Fast Agglomerative Clustering in C++ using FAISS

## Build
### Local
You need to have FAISS installed. The main.cpp runs agglomerative clustering on the top 40K words of glove-100.
Run the following commands:
```bash
git clone git@github.com:hhdri/agglo.cpp.git
mkdir build
cd build
cmake ..
make
```
### Docker
As FAISS doesn't have official binaries beside anaconda repo, faiss will also be built here. The main.cpp runs agglomerative clustering on the top 40K words of glove-100.
Run the following commands:
```bash
git clone git@github.com:hhdri/agglo.cpp.git
git clone git@github.com:facebookresearch/faiss.git
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
docker build --network=host -t agglo-cpp -f agglo.cpp/Dockerfile .
docker run agglo-cpp
```
