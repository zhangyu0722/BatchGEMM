#include "kernel_128.h"
// #include "kernel_256_new.h"
// #include "kernel_256_new.h"
// #include "kernel_512_new.h"
// #include "kernel_64_new.h"

template<int kThreads>
__global__ void gemm(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[]);


template<>
__global__ void gemm<128>(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];
	int t = T_strategy[i];
	switch(t){
		case 0:
		
        if (blockIdx.x * 16< M[i] && blockIdx.y * 16< N[i])	//&&threadIdx.x<128
			  gemm_128_16x16(M[i], N[i], K[i], A[i], B[i], C[i], sh);
			
		break;
		case 1:
			if (blockIdx.x * 32 < M[i] && blockIdx.y * 32 < N[i])	
				gemm_128_32x32(M[i], N[i], K[i], A[i], B[i], C[i], sh);
			break;
		case 2:
			if (blockIdx.x * 64 < M[i] && blockIdx.y * 64 < N[i])	
				gemm_128_64x64(M[i], N[i], K[i], A[i], B[i], C[i], sh);
			break;
			
	}
	return;
}
