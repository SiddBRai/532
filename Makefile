CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -c -arch=sm_50 \
						 -gencode=arch=compute_50,code=sm_50 \
						 -gencode=arch=compute_52,code=sm_52 \
						 -gencode=arch=compute_60,code=sm_60 \
						 -gencode=arch=compute_61,code=sm_61 \
						 -gencode=arch=compute_70,code=sm_70 \
						 -gencode=arch=compute_75,code=sm_75 \
						 -gencode=arch=compute_75,code=compute_75 \


LIBS=-lopenblas -lpthread -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda-10.1/lib64
INCDIRS=-I/usr/local/cuda-10.1/include

all: apriori

apriori: apriori.cu
	$(NVCC) $(CUDAFLAGS) $^ -o $@

clean:
	@rm -f apriori
