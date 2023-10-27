#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <hipblas.h>
#include "../include/util.h"

#define N_RUNS 10

int  main (int argc, char** argv) {

	hipSetDevice(3);

	if(argc<1){
		printf("Usage: input the batch size\n");
		exit(EXIT_FAILURE);
	}

	int BATCH = atoi(argv[1]);
	
	int *M;
	int *N;
	int *K;

	M = (int*) malloc(BATCH * sizeof(int));
	N = (int*) malloc(BATCH * sizeof(int));
	K = (int*) malloc(BATCH * sizeof(int));

	std::fstream fs;
	fs.open("../data/data_MN_K_512_128");

	if (!fs.is_open()){
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}
	
	//read matrix config	
	for (int i=0; i<BATCH; ++i){
		fs>>M[i]>>N[i]>>K[i];
	}

    float **A;
	float **B;
	float **C;
	float alpha = 1.f;
	float beta = 1.f;

	A = (float**) malloc(BATCH * sizeof(float*));
	B = (float**) malloc(BATCH * sizeof(float*));
	C = (float**) malloc(BATCH * sizeof(float*));

	for (int i=0; i<BATCH; ++i){
		hipMalloc((void**)&A[i], M[i]*K[i]*sizeof(float));
		hipMalloc((void**)&B[i], K[i]*N[i]*sizeof(float));
		hipMalloc((void**)&C[i], M[i]*N[i]*sizeof(float));
	}

	float elapsedTime = 0.f;
    double time=0.f;
	float gflops_per_sec = 0.f;
	double gflops = 0.f;
	for (int i=0; i<BATCH; ++i)
		gflops += ((2 * int64_t(M[i]) * int64_t(N[i]) * int64_t(K[i])) + (2 * int64_t(M[i]) * int64_t(N[i])) ) / 1.0e9;

	hipEvent_t start, stop;
    hipblasHandle_t handle;
    hipblasCreate(&handle);

	
	// warm-up
	for (int i=0; i<BATCH; ++i){
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M[i], N[i], K[i], &alpha, A[i], M[i], B[i], K[i], &beta, C[i], M[i]);
	}    
	hipDeviceSynchronize();


 
	hipEventCreate(&start);
	hipEventRecord(start,0);
	for (int run=0; run<N_RUNS; ++run){

		for (int i=0; i<BATCH; ++i){
			hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M[i], N[i], K[i], &alpha, A[i], M[i], B[i], K[i], &beta, C[i], M[i]);		
		}
	     hipDeviceSynchronize(); 
	}

	hipEventCreate(&stop);
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&elapsedTime, start,stop);	
    time = elapsedTime/N_RUNS;
	time /= 1.0e3; //convert time unit from millisecond to second
	
	int epoch=N_RUNS;
	
    printf("time======%f========N_RUNS====%d=======\n",time,epoch);
    gflops_per_sec = gflops / time;
     printf("gflops==%f         %f\n", gflops,gflops_per_sec);



	
	for (int i=0; i<BATCH; ++i){
		hipFree(A[i]);
		hipFree(B[i]);
		hipFree(C[i]);
	}

	free(M);
	free(N);
	free(K);
	free(A);
	free(B);
	free(C);

	return 0;
}
