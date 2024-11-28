CC=nvcc
# this g++-9 and -arch=sm_35 is only for it to work on university machine (GPUNODE2)
# the last flag disables warning error (it seems like sm_35 is deprecated)
C_FLAGS_NODE2=-ccbin /usr/bin/g++-9 -arch=sm_35 -Wno-deprecated-gpu-targets
C_FLAGS_NODE3=--std=c++20
SRC=src/main.cu
TARGET=k_means_clustering
TARGET_NODE2=${TARGET}-NODE2
TARGET_NODE3=${TARGET}-NODE3

node2: ${TARGET_NODE2}
node3: ${TARGET_NODE3}

${TARGET_NODE2}:
	${CC} ${C_FLAGS_NODE2} -o ${TARGET} ${SRC}

${TARGET_NODE3}:
	${CC} ${C_FLAGS_NODE3} -o ${TARGET} ${SRC}

clean:
	rm -f ${TARGET}

.PHONY: clean