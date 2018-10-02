/* objective
 * 	C = A*B  // A[m][k], B[k][n], C[m][n]
 * compile: nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 -O3 matmul_double_t.cu -o matmul_double_t
 */

#include <iostream>
#include <cstdlib>

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define TILE_WIDTH 16

#define EC(ans) { chkerr((ans), __FILE__, __LINE__); }
inline void chkerr(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) << " File: " << file << " Line: " << line << '\n';
        exit(-1);
    }
}

void init (double *A, double *B, int M , int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            A[i * K + j] = i * K + j;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = i * N + j + 1;
        }
    }

}


void matmul_transpose_double_host(double* A, double* B, double* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double tmp = 0;

            for (int k = 0; k < K; ++k)
            {
                //tmp += A[i * K + k] * B[k * N + j];
                tmp +=A[i* K +k ] * B[j*N + k];
            }

            C[i * N + j] = tmp;
        }
    }
}

__global__ void matmul_double(double* A, double* B , double* B_t, double* C, int M, int N, int K)
{
    /// complete code
    // B matrix is N*K so we should find transpose of B


    //__syncthreads() ;

    transposeNoBankConflicts(B_t , B , K , N) ;

    //__syncthreads() ;


    int bx = blockIdx.x ;
    int by = blockIdx.y ;

    int tx = threadIdx.x ;
    int ty = threadIdx.y ;

    int row = by * TILE_WIDTH + ty ;
    int col = bx * TILE_WIDTH + tx ;

    __shared__ double SA[TILE_WIDTH][TILE_WIDTH] ;
    __shared__ double SB[TILE_WIDTH][TILE_WIDTH] ;

    double Csub = 0;

    int rowB = N;
    int colB = K;

    for (int i = 0; i < (K-1)/TILE_WIDTH +1 ; ++i)
    {
        /* code */
        //SA[ty][tx] = A[row*n + i * TILE_WIDTH + tx] ;
        //SB[ty][tx] = B[(i * TILE_WIDTH + ty )*n + col   ] ;

        if ( (row < M) && (i * TILE_WIDTH + tx < K ) ){
            SA[ty][tx] = A[row*N + i * TILE_WIDTH + tx] ;
        }
        else{
            SA[ty][tx] = 0;
        }

        if ( (col < colB ) && ( i * TILE_WIDTH + ty < rowB) ){
            SB[ty][tx] = B[(i*TILE_WIDTH + ty)*rowB + col] ;
            //SB[ty][tx] = B_t[(i*TILE_WIDTH + ty)*K + col] ;

        }
        else{
            SB[ty][tx] = 0;

        }



        __syncthreads() ;

        for (int k = 0; k < TILE_WIDTH; ++k){   
            Csub += SA[ty][k]*SB[k][tx] ;
        }

        __syncthreads() ;
        

    }

    //C[row*n + col] = Csub ;

    if ( (row < M ) && ( col < N )){
        C[ row * N + col] = Csub ;
    }



}

__device__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH+1];

  int xIndex = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int yIndex = blockIdx.y * TILE_WIDTH + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_WIDTH + threadIdx.x;
  yIndex = blockIdx.x * TILE_WIDTH + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) 1{
    for (int i=0; i<TILE_WIDTH; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_WIDTH; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

void validate (double *host, double *gpu, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(std::abs(host[i * N + j] - gpu[i * N + j]) > 1e-3)
            {
                std::cerr << "possible error at position " << i << ',' << j << " host: " << host[i * N + j] << " device " << gpu[i * N + j] << '\n';
            }

        }
    }
}


int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        std::cerr << "Usage: ./matmul_double M N K\n";
        exit(-1);
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    /* Host alloc */
    double *hA = (double*) malloc (M * K * sizeof(double));
    double *hB = (double*) malloc (K * N * sizeof(double));
    double *hC = (double*) malloc (M * N * sizeof(double));
    double *dtohC = (double*) malloc (M * N * sizeof(double));

    /* Device alloc */
    /// complete code

    double *dA;
    double *dB;
    double *dC;
    double *dB_t;

    cudaMalloc((void**) &dA, M*K * sizeof(double)) ;
    cudaMalloc((void**) &dB, K*N * sizeof(double)) ;
    cudaMalloc((void**) &dC, M*N * sizeof(double)) ;
    cudaMalloc((void**) &dB_t, K*N * sizeof(double)) ;


    /* Initialize host memory*/
    init(hA, hB, M, N, K);

    /* host compute */
    matmul_double_host(hA, hB, hC, M, N, K);


    /* Copy from host to device */
    /// complete code
    cudaMemcpy(dA,hA ,M*K * sizeof(double) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy(dB,hB ,K*N * sizeof(double) , cudaMemcpyHostToDevice ) ;

    
    /* call gpu kernel */
    /// complete code

    //Initialize the grid and block dimensions here
    dim3 dimGrid( (N - 1) / TILE_WIDTH + 1 , (M - 1)/ TILE_WIDTH + 1 , 1) ;
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;


    matmul_double<<<dimGrid, dimBlock>>>(dA, dB , dB_t, dC , M , N , K) ;


    /* Copy from device to host (dC -> dtohC) */
    /// complete code

    cudaMemcpy(hC, dC , sizeof(double)*M*N , cudaMemcpyDeviceToHost) ;

    /* host vs device validation */
    validate(hC, dtohC, M, N);


    /* be clean */
    free(hA);
    free(hB);
    free(hC);
    free(dtohC);

    /// add code to free gpu memory

    cudaFree(dA) ;
    cudaFree(dB) ;
    cudaFree(dC) ;


    return 0;
}




