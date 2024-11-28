#include "stdio.h"
#include "string.h"
#include <cuda_runtime.h>
#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void init_data_from_txt(FILE* f, int N, int dim, int k, float ** points, 
float ** centroids, int * membership, float ** new_centroids_values, int * new_centroids_count)
{
    memset(new_centroids_count, 0, k*sizeof(int));
    memset(membership, -1, N*sizeof(int));
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            fscanf(f, "%f", &(points[j][i]));
        }
    }
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            centroids[j][i] = points[j][i];
            new_centroids_values[j][i] = 0;
        }
    }
}

void init_data_from_bin(FILE* f, int N, int dim, int k, float ** points, 
float ** centroids, int * membership, float ** new_centroids_values, int * new_centroids_count)
{
    memset(new_centroids_count, 0, k*sizeof(int));
    memset(membership, -1, N*sizeof(int));
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            fread(&(points[j][i]), 1, 4, f);
        }
    }
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            centroids[j][i] = points[j][i];
            new_centroids_values[j][i] = 0;
        }
    }
}


__global__ void countNewMemberships(float* d_points, float* d_centroids, int* d_memberships, int *N_d, int *dim_d, int *k_d)
{
    int idx = blockIdx.x*1024 + threadIdx.x;
    double minDist = 0;
    int kmin = 0;
    double dist = 0;
    int dim = *dim_d;
    int k = *k_d;
    int N = *N_d;
    if(idx < N)
    {
        for(int i = 0; i < dim; i++)
        {
            minDist += (d_centroids[i*k] - d_points[i*N + idx])*(d_centroids[i*k] - d_points[i*N + idx]);
        }
        for(int i = 1; i < k; i++) 
        {
            dist = 0;
            for(int j = 0; j < dim; j++)
            {
                dist += (d_centroids[j*k + i] - d_points[j*N + idx])*(d_centroids[j*k + i] - d_points[j*N + idx]);
            }
            if(dist < minDist)
            {
                minDist = dist;
                kmin = i;
            }
        }
        d_memberships[idx] = kmin;
    }

}

void kmeans_gpu1(char* type)
{
    cudaSetDevice(0);
    FILE* f;
    FILE* out;
    float ** points;
    float ** centroids;
    int * membership;
    int * new_membership;
    float ** new_centroids_values;
    int * new_centroids_count;
    int N;
    int dim;
    int k;
    if(strcmp(type, "dat") == 0)
    {
        printf("noted\n");
        f = fopen("points_5mln_4d_5c.dat", "r");
        fread(&N, 1, 4, f);
        fread(&dim, 1, 4, f);
        fread(&k, 1, 4, f);
    }
    else
    {
        f = fopen("points_5mln_4d_5c.txt", "r");
        fscanf(f, "%d %d %d", &N, &dim, &k);
    }
    printf("najak\n");
    out = fopen("cpu_results.txt", "w");
    printf("%d, %d, %d\n", N, dim, k);
    new_centroids_count = (int*)malloc(k*sizeof(int));
    new_centroids_values = (float**)malloc(dim*sizeof(float*));
    membership = (int*)malloc(N*sizeof(int));
    new_membership = (int*)malloc(N*sizeof(int));
    points = (float**)malloc(dim*sizeof(float*));
    centroids = (float**)malloc(dim*sizeof(float*));
    for(int i = 0; i < dim; i++)
    {
        points[i] = (float*)malloc(N*sizeof(float));
        centroids[i] = (float*)malloc(k*sizeof(float));
        new_centroids_values[i] = (float*)malloc(k*sizeof(float));
    }
    if(strcmp(type, "dat") == 0)
    {
        init_data_from_bin(f, N, dim, k, points, centroids, membership, new_centroids_values, new_centroids_count);
    }
    else
    {
        init_data_from_txt(f, N, dim, k, points, centroids, membership, new_centroids_values, new_centroids_count);
    }
    printf("wczytane\n");
    //alokacja CUDA

    float * d_points;
    float * d_centroids;
    int * d_memberships;
    int * d_N, * d_dim, * d_k;
    
    gpuErrchk( cudaMalloc(&d_points, sizeof(float)*N*dim));
    gpuErrchk( cudaMalloc(&d_centroids, sizeof(float)*k*dim));
    gpuErrchk( cudaMalloc(&d_memberships, sizeof(int)*N));
    gpuErrchk( cudaMalloc(&d_N, sizeof(int)));
    gpuErrchk( cudaMalloc(&d_k, sizeof(int)));
    gpuErrchk( cudaMalloc(&d_dim, sizeof(int)));
    
    //CUDA init
    gpuErrchk(cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice));
    for(int i = 0; i < dim; i++)
    {
        gpuErrchk(cudaMemcpy((d_points + N*i), points[i], N*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk( cudaMemcpy((d_centroids + k*i), centroids[i], k*sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("GPU zainicjalizowane\n");

    //główna pętla
    printf("zaczynamy liczyć..\n");
    int counter = 0;
    int BLOCK_COUNT = N/1024 + 1;
    while(counter < 100)
    {
        //zmiana centroidów
        countNewMemberships<<<BLOCK_COUNT, 1024>>>(d_points, d_centroids, d_memberships, d_N, d_dim, d_k);

        gpuErrchk( cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(new_membership, d_memberships, sizeof(int)*N, cudaMemcpyDeviceToHost));
        int how_many_changed = 0;
        for(int i = 0; i < N; i++)
        {
            int clusterid = new_membership[i];
            if(membership[i] != clusterid)
            {
                membership[i] = clusterid;
                how_many_changed++;
            }
            new_centroids_count[clusterid]++;
            for(int j = 0; j < dim; j++)
            {
                new_centroids_values[j][clusterid] += points[j][i];
            }
        }
        if(how_many_changed == 0)
        {
            break;
        }
        //wyliczenie nowych centroidów
        for(int i = 0; i < k; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                centroids[j][i] = new_centroids_values[j][i]/(double)new_centroids_count[i];
                new_centroids_values[j][i] = 0;
            }
            new_centroids_count[i] = 0;
        }
        for(int i = 0; i < dim; i++)
        {
            gpuErrchk( cudaMemcpy((d_centroids + k*i), centroids[i], k*sizeof(float), cudaMemcpyHostToDevice));
        }
        counter++;
        printf("%d\n", counter);
    }

        //zapisanie wyników
    printf("zapisywanie...\n");
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            fprintf(out, "%4f  ", centroids[j][i]);
        }
        fprintf(out, "\n");
    }
    for(int i = 0; i < N; i++)
    {
        fprintf(out, "%d\n", membership[i]);
    }

    printf("zwalnianie pamięci...\n");
    for(int i = 0; i < dim; i++)
    {
        free(points[i]);
        free(centroids[i]);
        free(new_centroids_values[i]);
    }
    free(new_centroids_count);
    free(new_centroids_values);
    free(points);
    free(centroids);
    free(membership);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_memberships);

}


void kmeans_cpu(char* type)
{
    FILE* f;
    FILE* out;
    float ** points;
    float ** centroids;
    int * membership;
    float ** new_centroids_values;
    int * new_centroids_count;
    int N;
    int dim;
    int k;
    if(strcmp(type, "dat") == 0)
    {
        printf("noted\n");
        f = fopen("points_5mln_4d_5c.dat", "r");
        fread(&N, 1, 4, f);
        fread(&dim, 1, 4, f);
        fread(&k, 1, 4, f);
    }
    else
    {
        f = fopen("points_5mln_4d_5c.txt", "r");
        fscanf(f, "%d %d %d", &N, &dim, &k);
    }
    printf("najak\n");
    out = fopen("cpu_results.txt", "w");
    printf("%d, %d, %d\n", N, dim, k);
    new_centroids_count = (int*)malloc(k*sizeof(int));
    new_centroids_values = (float**)malloc(dim*sizeof(float*));
    membership = (int*)malloc(N*sizeof(int));
    points = (float**)malloc(dim*sizeof(float*));
    centroids = (float**)malloc(dim*sizeof(float*));
    for(int i = 0; i < dim; i++)
    {
        points[i] = (float*)malloc(N*sizeof(float));
        centroids[i] = (float*)malloc(k*sizeof(float));
        new_centroids_values[i] = (float*)malloc(k*sizeof(float));
    }
    if(strcmp(type, "dat") == 0)
    {
        init_data_from_bin(f, N, dim, k, points, centroids, membership, new_centroids_values, new_centroids_count);
    }
    else
    {
        init_data_from_txt(f, N, dim, k, points, centroids, membership, new_centroids_values, new_centroids_count);
    }
    printf("wczytane\n");
    for(int i = 0; i < dim; i++)
    {
        printf("%f\n", points[i][100]);
    }
    printf("zaczynamy liczyć..\n");
    int counter = 0;
    while(counter < 100)
    {
        //zmiana centroidów
        int how_many_changed = 0;
        for(int i = 0; i < N; i++)
        {
            double mindist = -1;
            int clusterid = -1;
            for(int j = 0; j < k; j++)
            {
                double dist = 0;
                for(int l = 0; l < dim; l++)
                {
                    dist += (points[l][i] - centroids[l][j])*(points[l][i] - centroids[l][j]);
                }
                if(mindist == -1 || mindist > dist)
                {
                    clusterid = j;
                    mindist = dist;
                }
            }
            if(i == 100)
            {
                printf("%d, %f\n", i, mindist);
            }
            if(membership[i] != clusterid)
            {
                membership[i] = clusterid;
                how_many_changed++;
            }
            new_centroids_count[clusterid]++;
            for(int j = 0; j < dim; j++)
            {
                new_centroids_values[j][clusterid] += points[j][i];
            }
        }
        if(how_many_changed == 0)
        {
            break;
        }
        //wyliczenie nowych centroidów
        for(int i = 0; i < k; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                centroids[j][i] = new_centroids_values[j][i]/(double)new_centroids_count[i];
                new_centroids_values[j][i] = 0;
            }
            new_centroids_count[i] = 0;
        }
        counter++;
        printf("%d\n", counter);
    }

    //zapisanie wyników
    printf("zapisywanie...\n");
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            fprintf(out, "%4f  ", centroids[j][i]);
        }
        fprintf(out, "\n");
    }
    for(int i = 0; i < N; i++)
    {
        fprintf(out, "%d\n", membership[i]);
    }
    printf("zwalnianie pamięci...\n");
    for(int i = 0; i < dim; i++)
    {
        free(points[i]);
        free(centroids[i]);
        free(new_centroids_values[i]);
    }
    free(new_centroids_count);
    free(new_centroids_values);
    free(points);
    free(centroids);
    free(membership);
}

int main(int argc, char*argv[])
{
    if(argc == 3)
    {
        if(strcmp(argv[2], "cpu") == 0)
        {
            kmeans_cpu(argv[1]);
        }
        else if(strcmp(argv[2], "gpu1") == 0)
        {
            kmeans_gpu1(argv[1]);
        }
    }
    return 0;
}