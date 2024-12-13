# Implemantation of k-means algorithm in CUDA
## usage: KMeans input_file_type algorithm input_file output_file
input_file_type - type of input file (possible values - bin/txt)
algorithm - choice of algorithm (possible values - cpu/gpu1/gpu2)
cpu - cpu implementation of algorithm
gpu1 - CUDA implementation using kernels
gpu2 - CUDA implementation using kernel + thrust library

# Input file format:
first line - 3 integers representing:
N - count of points
d - count of dimensions
k - count of clusters

N more lines with d float numbers in each representing points

# Output file format:
k lines with d float numbers in each representing clusters
N more lines with one ineteger number each representing number of a cluster
to which the point was assigned
