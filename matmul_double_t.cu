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
            //A[i * K + j] = 1 ;
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            
            B[i * K + j] = i * K + j + 1;
            //B[i * K + j] = 2;
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
                tmp +=  A[ i* K +k ] * B[j*K + k];
            }

            C[i * N + j] = tmp;
        }
    }
}




__global__ void matmul_double(double* A, double* B ,  double* C, int M, int N, int K)
{



    int bx = blockIdx.x ;
    int by = blockIdx.y ;

    int tx = threadIdx.x ;
    int ty = threadIdx.y ;

    int row = by * TILE_WIDTH + ty ;
    int col = bx * TILE_WIDTH + tx ;

    __shared__ double SA[TILE_WIDTH][TILE_WIDTH+1] ;
    __shared__ double SB[TILE_WIDTH][TILE_WIDTH+1] ;

    double Csub = 0;

    

    for (int i = 0; i < (K-1)/TILE_WIDTH +1 ; ++i)
    {
        /* code */
        

        if ( (row < M) && (i * TILE_WIDTH + tx < K ) ){
            SA[ty][tx] = A[row*K + i * TILE_WIDTH + tx] ;
        }
        else{
            SA[ty][tx] = 0;
        }

       

        
        if ( (col < N ) &&  ( i * TILE_WIDTH + ty < K)){
            SB[tx][ty] = B[ col * K + i*TILE_WIDTH + ty] ;
        } 
        else{
            SB[tx][ty] = 0;
        }
        

        __syncthreads() ;

        for (int k = 0; k < TILE_WIDTH; ++k){   
            Csub += SA[ty][k]*SB[tx][k] ;
        }

        __syncthreads() ;
        

    }

    //C[row*n + col] = Csub ;

    if ( (row < M ) && ( col < N )){
        C[ row * N + col] = Csub ;
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
    //double *dB_t;

    cudaMalloc((void**) &dA, M*K * sizeof(double)) ;
    cudaMalloc((void**) &dB, K*N * sizeof(double)) ;
    cudaMalloc((void**) &dC, M*N * sizeof(double)) ;
    //
    //cudaMalloc((void**) &dB_t, K*N * sizeof(double)) ;


    /* Initialize host memory*/
    init(hA, hB, M, N, K);

    /* host compute */
    matmul_transpose_double_host(hA, hB, hC, M, N, K);


    /* Copy from host to device */
    /// complete code
    cudaMemcpy(dA,hA ,M*K * sizeof(double) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy(dB,hB ,K*N * sizeof(double) , cudaMemcpyHostToDevice ) ;

    
    /* call gpu kernel */
    /// complete code

    //Initialize the grid and block dimensions here
    dim3 dimGrid( (N - 1) / TILE_WIDTH + 1 , (M - 1)/ TILE_WIDTH + 1 , 1) ;
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , 1) ;


    matmul_double<<<dimGrid, dimBlock>>>(dA, dB , dC , M , N , K) ;


    /* Copy from device to host (dC -> dtohC) */
    /// complete code

    cudaMemcpy(dtohC, dC , sizeof(double)*M*N , cudaMemcpyDeviceToHost) ;

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




