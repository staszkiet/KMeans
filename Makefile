CC=nvcc
C_FLAGS_NODE3=--std=c++20
TARGET=KMeans

all:
	${CC} ${C_FLAGS_NODE3} -o KMeans main.cu

clean:
	rm -f KMeans

.PHONY: clean