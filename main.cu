#include "stdio.h"
#include "string.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/iterator/constant_iterator.h>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <int dim>
float** initTxt(FILE * f, int N, int k)
{
    float ** points;
    if((points = (float**)malloc(dim*sizeof(float*))) == NULL)
    {
        ERR("malloc points array\n");
        return NULL;
    }
    for(int i = 0; i < dim; i++)
    {
        if((points[i] = (float*)malloc(N*sizeof(float))) == NULL)
        {
            ERR("malloc point array\n");
            return NULL;
        }
    }
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            if(fscanf(f, "%f", &(points[j][i])) != 1)
            {
                if(ferror(f))
                {
                    ERR("fscanf falied");
                    return NULL;
                }
            }
        }
    }
    return points;

}

template <int dim>
float** initBin(FILE * f, int N, int k)
{
    float ** points;
    if((points = (float**)malloc(dim*sizeof(float*))) == NULL)
    {
        ERR("malloc points array");
        return NULL;
    }
    for(int i = 0; i < dim; i++)
    {
        if((points[i] = (float*)malloc(N*sizeof(float))) == NULL)
        {
            ERR("malloc point array");
            return NULL;
        }
    }
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            if(fread(&(points[j][i]), 1, 4, f) != 4)
            {
                if(ferror(f))
                {
                    ERR("fread failed");
                    return NULL;
                }
            }
        }
    }
    return points;

}

template <int dim>
int saveToFile(char* name, float * centroids, int * membership, int k, int N)
{
    FILE* out;
    if((out = fopen(name, "w")) == NULL)
    {
        ERR("fopen");
        return 1;
    }
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            fprintf(out, "%.4f  ", centroids[j*k + i]);
        }
        fprintf(out, "\n");
    }
    for(int i = 0; i < N; i++)
    {
        fprintf(out, "%d\n", membership[i]);
    }
    fclose(out);
    return 0;
}

template <int dim>
__global__ void countNewCentroids(float * d_points, float* d_centroids, int * d_memberships, float* d_new_centroids_values, int* d_new_centroids_count, int * d_k, int * d_N)
{
    int k = *d_k;
    int N = *d_N;
    extern __shared__ float array[];
    float* values = (float*)array;
    int* count = (int*)&array[k*dim];
    int idx = blockIdx.x*1024 + threadIdx.x;

    for (int i = threadIdx.x; i < k * dim; i += blockDim.x) {
        values[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        count[i] = 0;
    }
    __syncthreads();
    if(idx < N)
    {
        int clusterid = d_memberships[idx];
        for(int i = 0; i < dim; i++)
        {
            atomicAdd(&values[i*k + clusterid], d_points[i*N + idx]);
        }
        atomicAdd(&count[clusterid], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < k * dim; i += blockDim.x) {
        atomicAdd(&d_new_centroids_values[i], values[i]);
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        atomicAdd(&d_new_centroids_count[i], count[i]);
    }
}

template<int dim>
__global__ void countNewMemberships(float* d_points, float* d_centroids, int* d_memberships, int* d_changed, int *N_d,  int *k_d)
{
    int idx = blockIdx.x*1024 + threadIdx.x;
    double minDist = 0;
    int kmin = 0;
    double dist = 0;
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
        if(d_memberships[idx] != kmin)
        {
            d_changed[idx] = 1;
        }
        d_memberships[idx] = kmin;
    }

}

template<int dim>
void kmeans_gpu2(char* type, char* output, int N, int k, FILE * f)
{
    cudaSetDevice(0);
    printf("algorytm k-means na GPU - thrust\n");

    //eventy i tablice + zmienne z CPU
    cudaError_t status;
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ** points;
    float * centroids;
    int * membership;
    int counter = 0;
    int BLOCK_COUNT = N / 1024 + 1;

    //tablice z GPU
    float * d_points;
    float * d_centroids;
    int * d_memberships;
    int * d_N, * d_k;
    thrust::device_vector<int> d_changed;
    
    printf("wczytywanie danych:\n");

    //wybór metody wczytania danych
    cudaEventRecord(start, 0);
    if(strcmp(type, "bin") == 0)
    {
        points = initBin<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error2;
        }
    }
    else if(strcmp(type, "txt") == 0)
    {
        points = initTxt<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error2;
        }
    }
    else
    {
        printf("niepoprawny format pliku\n");
        printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
        exit(0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("dane wczytane czas %.4f\n", time);
    printf("ilość punktów: %d\n", N);
    printf("ilość klastrów: %d\n", k);
    printf("ilość wymiarów: %d\n", dim);
    printf("format danych: %s\n", type);
    printf("alokacja zasobów:\n");
    cudaEventRecord(start, 0);

    //alokacja elementów z CPU
    centroids = (float*)malloc(dim*sizeof(float)*k);
    if(centroids == NULL)
    {
        ERR("centroids");
        goto Error2;
    }
    membership = (int*)malloc(N*sizeof(int));
    if(membership == NULL)
    {
        ERR("membership");
        goto Error2;
    }

    //wybór początowych centroidów
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < k; j++)
        {
            centroids[i*k + j] = points[i][j];
        }
    }



    // alokacja tablic na GPU
    status = cudaMalloc(&d_points, sizeof(float)*N*dim);
    if (status != cudaSuccess) { printf("Error allocating d_points\n"); goto Error2; }

    status = cudaMalloc(&d_centroids, sizeof(float)*k*dim);
    if (status != cudaSuccess) { printf("Error allocating d_centroids\n"); goto Error2; }

    status = cudaMalloc(&d_memberships, sizeof(int)*N);
    if (status != cudaSuccess) { printf("Error allocating d_memberships\n"); goto Error2; }

    status = cudaMalloc(&d_N, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_N\n"); goto Error2; }

    status = cudaMalloc(&d_k, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_k\n"); goto Error2; }

    //inicjalizacja tablic z GPU odpowiednimi wartościami

    status = cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_N\n"); goto Error2; }

    status = cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_k\n"); goto Error2; }

    status = cudaMemcpy(d_centroids, centroids, k*sizeof(float)*dim, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying to d_centroids\n"); goto Error2; }

    status = cudaMemset(d_memberships, -1, N*sizeof(int));
    if (status != cudaSuccess) { printf("Error setting to d_memberships\n"); goto Error2; }

    for (int i = 0; i < dim; i++) {
        status = cudaMemcpy((d_points + N * i), points[i], N * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { printf("Error copying to d_points\n"); goto Error2; }
    }

    d_changed.resize(N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("dane zaalokowane, czas: %.4f\n", time);
    printf("zaczynam liczyć...\n");
    cudaEventRecord(start, 0);

    //główna częśc programu
    while (counter < 100) {
        printf("numer iteracji: %d\n", counter);

        //obliczenie przenależności punktów do centroidów
        countNewMemberships<dim><<<BLOCK_COUNT, 1024>>>(d_points, d_centroids, d_memberships, thrust::raw_pointer_cast(d_changed.data()), d_N, d_k);

        status = cudaGetLastError();
        if (status != cudaSuccess) { printf("Kernel Failed\n"); goto Error2; }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) { printf("Error synchronizing device\n"); goto Error2; }

        int how_many_changed = thrust::reduce(d_changed.begin(), d_changed.end());
        printf("liczba punktów które zmieniły centroid: %d\n", how_many_changed);

        if (how_many_changed == 0) {
            break;
        }

        // stały wektor z przynależnościami
        thrust::device_vector<int> d_new_memberships(d_memberships, d_memberships + N);

        // kopia wektora z przynależnościami (służąca do operacji w pętli)
        thrust::device_vector<int> d_new_memberships_cpy(N);
        thrust::copy(d_new_memberships.begin(), d_new_memberships.end(), d_new_memberships_cpy.begin());

        // wektor zawierający liczności każdego z klastrów
        thrust::device_vector<int> d_counts(k);

        // wektor potrzebny do wykonania operacji reduce_by_key
        thrust::device_vector<int> d_reduced_keys(dim);

        //obliczenia licznośći w klastrach
        thrust::sort(d_new_memberships_cpy.begin(), d_new_memberships_cpy.end());
        thrust::reduce_by_key(d_new_memberships_cpy.begin(), d_new_memberships_cpy.end(),
                            thrust::constant_iterator<int>(1), d_reduced_keys.begin(), d_counts.begin());

        // Wektor trzymający wartości danej współrzędnej + k zer (aby po redukcji obecne były wszystkie klastry)
        thrust::device_vector<float> d_coordinate_values(N + k);

        // Wektor trzymający sumy danego koordynatu dla danego klastra
        thrust::device_vector<float> d_coordinate_sums(k);

        // Wektor zawierający nowe wartości współrzędnej dla centroidów
        thrust::device_vector<float> d_new_centroids(k);
        d_new_memberships_cpy.resize(N + k);

        //wektor zawierający zawartości od 0 do k-1 doczepiany do wektora z przynależnościami
        // (aby po redukcji w kluczach były wszystkie centroidy)
        thrust::device_vector<int> d_all_clusters(k);
        thrust::sequence(d_all_clusters.begin(), d_all_clusters.end());

        for (int i = 1; i <= dim; i++) {

            //przekopiowanie wartości danej współrzędnej do wektora i dodanie do niego zer (dla wartości kontrolujących
            // żeby po redukcji w kluczach były wszystkie centroidy)
            thrust::copy(d_points + (i - 1) * N, d_points + i * N, d_coordinate_values.begin());
            thrust::fill(d_coordinate_values.begin() + N, d_coordinate_values.end(), 0);

            //przekopiowanie wektora przynależności do wektora i doczepienie do niego wektora z wartościami
            //od 0 do k-1 (aby po redukcji w kluczach były wszystkie centroidy)
            thrust::copy(d_new_memberships.begin(), d_new_memberships.end(), d_new_memberships_cpy.begin());
            thrust::copy(d_all_clusters.begin(), d_all_clusters.end(), d_new_memberships_cpy.begin() + N);

            //obliczenie nowych wartości wpółrzędnej dla centroidów i przepisanie ich do tablicy
            thrust::sort_by_key(d_new_memberships_cpy.begin(), d_new_memberships_cpy.end(), d_coordinate_values.begin());
            thrust::reduce_by_key(d_new_memberships_cpy.begin(), d_new_memberships_cpy.end(), d_coordinate_values.begin(),
                                d_reduced_keys.begin(), d_coordinate_sums.begin());

            thrust::transform(d_coordinate_sums.begin(), d_coordinate_sums.end(), d_counts.begin(),
                            d_new_centroids.begin(), thrust::divides<float>());
            thrust::copy(d_new_centroids.begin(), d_new_centroids.end(), d_centroids + (i - 1) * k);
        }

        thrust::fill(d_changed.begin(), d_changed.end(), 0);
        counter++;
        printf("%d\n", counter);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("obliczenia zakończone, czas: %.4f\n", time);

    //kopiowanie z gpu na cpu

    status = cudaMemcpy(membership, d_memberships, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying membership data\n"); goto Error2; }

    status = cudaMemcpy(centroids, d_centroids, k * sizeof(float) * dim, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying centroids data\n"); goto Error2; }

    // zpais do pliku
    printf("zapisywanie...\n");
    cudaEventRecord(start, 0);
    if(saveToFile<dim>(output, centroids, membership, k, N) != 0)
    {
        goto Error2;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zapisano, czas: %.4f\n", time);

    //dealokacja zasobów
Error2:
    if(points != NULL)
    {
        for(int i = 0; i < dim; i++)
        {
            if(points[i] != NULL)
            {
                free(points[i]);
            }
        }
        free(points);
    }
    if(centroids != NULL)
    {
        free(centroids);
    }
    if(membership != NULL)
    {
        free(membership);
    }
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_memberships);
    cudaFree(d_N);
    cudaFree(d_k);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

template<int dim>
void kmeans_gpu1(char* type, char* output, int N, int k, FILE * f)
{
    printf("algorytm k-means na GPU - kernele\n");
    cudaSetDevice(0);

    //inicjalizacja timerów
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //elementy potrzebne na CPU
    float ** points;
    float * centroids;
    float * new_centroids_values;
    int * new_centroids_count;
    int * membership;
    int counter = 0;
    int BLOCK_COUNT = N / 1024 + 1;

    //elementy potrzebne na GPU
    float * d_points;
    float * d_centroids;
    float * d_new_centroids_values;
    int * d_new_centroids_count;
    int * d_memberships;
    int * d_N, * d_k;
    thrust::device_vector<int> d_changed;

    //wczytywanie danych
    printf("wczytywanie danych:\n");
    cudaEventRecord(start, 0);
    if (strcmp(type, "bin") == 0) 
    {
        points = initBin<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error;
        }
    } 
    else if (strcmp(type, "txt") == 0) 
    {
        points = initTxt<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error;
        }
    } 
    else 
    {
        printf("niepoprawny format pliku\n");
        printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
        exit(0);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("dane wczytane czas %.4f\n", time);
    printf("ilość punktów: %d\n", N);
    printf("ilość klastrów: %d\n", k);
    printf("ilość wymiarów: %d\n", dim);
    printf("format danych: %s\n", type);
    
    printf("alokacja zasobów:\n");
    cudaEventRecord(start, 0);


    //alokacja tablic na CPU
    new_centroids_count = (int*)malloc(k * sizeof(int));
    if(new_centroids_count == NULL)
    {
        ERR("new centroids count");
        goto Error;
    }
    new_centroids_values = (float*)malloc(dim * sizeof(float) * k);
    if(new_centroids_values == NULL)
    {
        ERR("new centroids values");
        goto Error;
    }
    centroids = (float*)malloc(dim * sizeof(float) * k);
    if(centroids == NULL)
    {
        ERR("centroids");
        goto Error;
    }
    membership = (int*)malloc(N * sizeof(int));
    if(membership == NULL)
    {
        ERR("membership");
        goto Error;
    }
    
    //inicjalizacja tablic na CPU odpowiednimi wartościami
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < k; j++) {
            centroids[i * k + j] = points[i][j];
            new_centroids_values[i * k + j] = 0;
        }
    }

    // Alokacja CUDA
    cudaError_t status;
    
    status = cudaMalloc(&d_points, sizeof(float) * N * dim);
    if (status != cudaSuccess) { printf("Error allocating d_points\n"); goto Error; }

    status = cudaMalloc(&d_centroids, sizeof(float) * k * dim);
    if (status != cudaSuccess) { printf("Error allocating d_centroids\n"); goto Error; }

    status = cudaMalloc(&d_memberships, sizeof(int) * N);
    if (status != cudaSuccess) { printf("Error allocating d_memberships\n"); goto Error; }

    status = cudaMalloc(&d_N, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_N\n"); goto Error; }

    status = cudaMalloc(&d_k, sizeof(int));
    if (status != cudaSuccess) { printf("Error allocating d_k\n"); goto Error; }

    status = cudaMalloc(&d_new_centroids_values, sizeof(float) * k * dim);
    if (status != cudaSuccess) { printf("Error allocating d_new_centroids_values\n"); goto Error; }

    status = cudaMalloc(&d_new_centroids_count, sizeof(int) * k);
    if (status != cudaSuccess) { printf("Error allocating d_new_centroids_count\n"); goto Error; }

    // inicjalizacja tablic na GPU odpowiednimi wartościami
    status = cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying d_N\n"); goto Error; }

    status = cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying d_k\n"); goto Error; }

    status = cudaMemcpy(d_centroids, centroids, k * sizeof(float) * dim, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { printf("Error copying d_centroids\n"); goto Error; }

    status = cudaMemset(d_memberships, -1, N * sizeof(int));
    if (status != cudaSuccess) { printf("Error setting d_memberships\n"); goto Error; }

    status = cudaMemset(d_new_centroids_count, 0, k * sizeof(int));
    if (status != cudaSuccess) { printf("Error setting d_new_centroids_count\n"); goto Error; }

    status = cudaMemset(d_new_centroids_values, 0, k * sizeof(float) * dim);
    if (status != cudaSuccess) { printf("Error setting d_new_centroids_values\n"); goto Error; }

    for (int i = 0; i < dim; i++) {
        status = cudaMemcpy((d_points + N * i), points[i], N * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { printf("Error copying d_points\n"); goto Error; }
    }

    d_changed.resize(N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zaalokowane, czas:%.4f\n", time);

    printf("zaczynamy liczyć..\n");
    cudaEventRecord(start, 0);
    //główna część porgramu
    while (counter < 100) {
        printf("numer iteracji: %d\n", counter);

        // Obliczenie przynależności punktów do centroidów
        countNewMemberships<dim><<<BLOCK_COUNT, 1024>>>(d_points, d_centroids, d_memberships, thrust::raw_pointer_cast(d_changed.data()), d_N, d_k);
        
        status = cudaGetLastError();
        if (status != cudaSuccess) { printf("Kernel Failed\n"); goto Error; }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) { printf("Error synchronizing device\n"); goto Error; }

        int how_many_changed = thrust::reduce(thrust::device, d_changed.begin(), d_changed.end());
        printf("ilość punktów które zmieniły klaster: %d\n", how_many_changed);

        if (how_many_changed == 0) {
            break;
        }

        int shmemBytes = k * dim * sizeof(float) + k * sizeof(int);

        //obliczenie sum wartośći kolejnych współrzędnych dla elementów należących do odpowiednich centroidów oraz liczności
        //kolejnych centroidów
        countNewCentroids<dim><<<BLOCK_COUNT, 1024, shmemBytes>>>(d_points, d_centroids, d_memberships, d_new_centroids_values, d_new_centroids_count, d_k, d_N);

        status = cudaGetLastError();
        if (status != cudaSuccess) { printf("Kernel Failed\n"); goto Error; }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) { printf("Error synchronizing device after centroids update\n"); goto Error; }


        //kopiowanie wyników na CPU
        status = cudaMemcpy(new_centroids_values, d_new_centroids_values, sizeof(float) * k * dim, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) { printf("Error copying new_centroids_values\n"); goto Error; }

        status = cudaMemcpy(new_centroids_count, d_new_centroids_count, sizeof(int) * k, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) { printf("Error copying new_centroids_count\n"); goto Error; }

        //obliczenie nowych współrzędnych centroidów (średnia arytmetyczna)
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                centroids[j * k + i] = new_centroids_values[j * k + i] / (double)new_centroids_count[i];
                new_centroids_values[j * k + i] = 0;
            }
            new_centroids_count[i] = 0;
        }

        //ustawienie wartości dla odpowiednich tablic
        status = cudaMemcpy(d_centroids, centroids, k * sizeof(float) * dim, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { printf("Error copying d_centroids\n"); goto Error; }

        status = cudaMemset(d_new_centroids_count, 0, k * sizeof(int));
        if (status != cudaSuccess) { printf("Error resetting d_new_centroids_count\n"); goto Error; }

        status = cudaMemset(d_new_centroids_values, 0, k * sizeof(float) * dim);
        if (status != cudaSuccess) { printf("Error resetting d_new_centroids_values\n"); goto Error; }

        thrust::fill(thrust::device, d_changed.begin(), d_changed.end(), 0);
        counter++;
        printf("%d\n", counter);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("obliczenia zakończone, czas:%.4f\n", time);

    status = cudaMemcpy(membership, d_memberships, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { printf("Error copying membership\n"); goto Error; }

    //zapisanie wyników do pliku
    printf("zapisywanie...\n");
    cudaEventRecord(start, 0);
    if(saveToFile<dim>(output, centroids, membership, k, N) != 1)
    {
        goto Error;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zapisano, czas: %.4f\n", time);


    //zwolnienie zaalokowanie pamięci
Error:
    printf("zwalnianie pamięci...\n");
    cudaEventRecord(start, 0);
    
    if (points != NULL) {
        for (int i = 0; i < dim; i++) {
            if (points[i] != NULL) {
                free(points[i]);
            }
        }
        free(points);
    }
    if (centroids != NULL) {
        free(centroids);
    }
    if (new_centroids_count != NULL) {
        free(new_centroids_count);
    }
    if (new_centroids_values != NULL) {
        free(new_centroids_values);
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_memberships);
    cudaFree(d_new_centroids_count);
    cudaFree(d_new_centroids_values);
    cudaFree(d_N);
    cudaFree(d_k);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zwolniono, czas: %.4f\n", time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


template<int dim>
void kmeans_cpu(char* type, char* output, int N, int k, FILE * f)
{
    printf("algorytm k-means na CPU\n");
    // elementy potrzebne na CPU
    float ** points;
    float * centroids;
    int * membership;
    float * new_centroids_values;
    int * new_centroids_count;
    cudaEvent_t start, stop;
    float time;
    int counter = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //wczytywanie danych
    printf("wczytywanie danych:\n");
    cudaEventRecord(start, 0);
    if(strcmp(type, "bin") == 0)
    {
        points = initBin<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error3;
        }
    }
    else if(strcmp(type, "txt") == 0)
    {
        points = initTxt<dim>(f, N, k);
        if(points == NULL)
        {
            goto Error3;
        }
    }
    else
    {
        printf("niepoprawny format pliku\n");
        printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
        exit(0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("dane wczytane czas %.4f\n", time);
    printf("ilość punktów: %d\n", N);
    printf("ilość klastrów: %d\n", k);
    printf("ilość wymiarów: %d\n", dim);
    printf("format danych: %s\n", type);
    printf("alokacja zasobów:\n");

    //alokacja tablic na CPU
    cudaEventRecord(start, 0);
    new_centroids_count = (int*)malloc(k*sizeof(int));
    if(new_centroids_count == NULL)
    {
        ERR("new centroids count");
        goto Error3;
    }
    new_centroids_values = (float*)malloc(dim*sizeof(float)*k);
    if(new_centroids_values == NULL)
    {
        ERR("new centroids values");
        goto Error3;
    }
    membership = (int*)malloc(N*sizeof(int));
    if(membership == NULL)
    {
        ERR("membership");
        goto Error3;
    }
    centroids = (float*)malloc(dim*sizeof(float*)*k);
    if(centroids == NULL)
    {
        ERR("centroids");
        goto Error3;
    }

    //inicjalizacja tablic odpowiednimi wartościami
    memset(new_centroids_count, 0, k*sizeof(int));
    memset(membership, -1, N*sizeof(int));
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < k; j++)
        {
            centroids[i*k + j] = points[i][j];
        }
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zaalokowane, czas: %.4f\n", time);
    printf("rozpoczynamy obliczenia\n");
    cudaEventRecord(start, 0);

    //główna pętla
    while(counter < 100)
    {
        //obliczenie nowych przynależnosći dla punktów
        printf("numer iteracji: %d\n", counter);
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
                    dist += (points[l][i] - centroids[l*k + j])*(points[l][i] - centroids[l*k + j]);
                }
                if(mindist == -1 || mindist > dist)
                {
                    clusterid = j;
                    mindist = dist;
                }
            }
            if(membership[i] != clusterid)
            {
                membership[i] = clusterid;
                how_many_changed++;
            }
            new_centroids_count[clusterid]++;
            for(int j = 0; j < dim; j++)
            {
                new_centroids_values[j*k + clusterid] += points[j][i];
            }
        }
        printf("ilość punktów które zmieniły przynależność: %d\n", how_many_changed);
        if(how_many_changed == 0)
        {
            break;
        }
        //wyliczenie nowych centroidów
        for(int i = 0; i < k; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                centroids[j*k + i] = new_centroids_values[j*k + i]/(double)new_centroids_count[i];
                new_centroids_values[j*k + i] = 0;
            }
            new_centroids_count[i] = 0;
        }
        counter++;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("obliczenia zakończone, czas : %.4f\n", time);

    //zapisanie wyników
    printf("zapisywanie do pliku\n");
    cudaEventRecord(start, 0);
    if(saveToFile<dim>(output, centroids, membership, k, N) != 0)
    {
        goto Error3;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zapisano, czas: %.4f\n", time);
Error3:
    printf("zwalnianie pamięci...\n");
    cudaEventRecord(start, 0);
    //dealokacja
    if(points != NULL)
    {
        for(int i = 0; i < dim; i++)
        {
            if(points[i] != NULL)
            {
                free(points[i]);
            }
        }
        free(points);
    }
    if(new_centroids_count != NULL)
    {
        free(new_centroids_count);
    }
    if(new_centroids_values != NULL)
    {
        free(new_centroids_values);
    }
    if(centroids != NULL)
    {
        free(centroids);
    }
    if(membership != NULL)
    {
        free(membership);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("zwolnione, czas: %.4f\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<int dim>
void chooseAlgorithm(FILE * f, int N, int k, char* argv[])
{
        if(strcmp(argv[2], "cpu") == 0)
        {
            kmeans_cpu<dim>(argv[1], argv[4], N, k, f);
        }
        else if(strcmp(argv[2], "gpu1") == 0)
        {
            kmeans_gpu1<dim>(argv[1], argv[4], N, k, f);
        }
        else if(strcmp(argv[2], "gpu2") == 0)
        {
            kmeans_gpu2<dim>(argv[1], argv[4], N, k, f);
        }
        else
        {
            printf("niepoprawmy wybór_algorytmu\n");
            printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
        }
}

int main(int argc, char*argv[])
{
    FILE* f;
    int N;
    int dim;
    int k;
    if(argc == 5)
    {
        if((f = fopen(argv[3], "r")) == NULL)
        {
            ERR("fopen failed\n");
        }
        if(strcmp(argv[1], "bin") == 0)
        {
            fread(&N, 1, 4, f);
            fread(&dim, 1, 4, f);
            fread(&k, 1, 4, f);
        }
        else if(strcmp(argv[1], "txt") == 0)
        {
            fscanf(f, "%d %d %d", &N, &dim, &k);
        }
        else
        {
            printf("niepoprawny format pliku\n");
            printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
            exit(0);
        }
        switch (dim) {
            case 1: {
                chooseAlgorithm<1>(f, N, k, argv);
                break;
            }
            case 2: {
                chooseAlgorithm<2>(f, N, k, argv);
                break;
            }
            case 3: {
                chooseAlgorithm<3>(f, N, k, argv);
                break;
            }
            case 4: {
                chooseAlgorithm<4>(f, N, k, argv);
                break;
            }
            case 5: {
                chooseAlgorithm<5>(f, N, k, argv);
                break;
            }
            case 6: {
                chooseAlgorithm<6>(f, N, k, argv);
                break;
            }
            case 7: {
                chooseAlgorithm<7>(f, N, k, argv);
                break;
            }
            case 8: {
                chooseAlgorithm<8>(f, N, k, argv);
                break;
            }
            case 9: {
                chooseAlgorithm<9>(f, N, k, argv);
                break;
            }
            case 10: {
                chooseAlgorithm<10>(f, N, k, argv);
                break;
            }
            case 11: {
                chooseAlgorithm<11>(f, N, k, argv);
                break;
            }
            case 12: {
                chooseAlgorithm<12>(f, N, k, argv);
                break;
            }
            case 13: {
                chooseAlgorithm<13>(f, N, k, argv);
                break;
            }
            case 14: {
                chooseAlgorithm<14>(f, N, k, argv);
                break;
            }
            case 15: {
                chooseAlgorithm<15>(f, N, k, argv);
                break;
            }
            case 16: {
                chooseAlgorithm<16>(f, N, k, argv);
                break;
            }
            case 17: {
                chooseAlgorithm<17>(f, N, k, argv);
                break;
            }
            case 18: {
                chooseAlgorithm<18>(f, N, k, argv);
                break;
            }
            case 19: {
                chooseAlgorithm<19>(f, N, k, argv);
                break;
            }
            case 20: {
                chooseAlgorithm<20>(f, N, k, argv);
                break;
            }
            default: {
                fprintf(stderr, "Niepoprawny wymiar: %d\n", dim);
                break;
            }
        }
        fclose(f);
    }
    else
    {
        printf("zła ilość parametrów\n");
        printf("schemat wywołania: ./k_means_clustering format_pliku(bin, txt) wybór_algorytmu(cpu, gpu1, gpu2) plik_wejściowy plik_wyjściowy\n");
    }
    return 0;
}
