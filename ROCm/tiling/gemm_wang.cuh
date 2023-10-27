#include "hip/hip_runtime.h"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <hipblas.h>
#include "../include/util.h"
#include "kernel.h"

#define N_RUNS 10

int main(int argc, char **argv) {

    ErrChk(hipSetDevice(3));

    if (argc < 2) {
        printf("Usage: input the batch size\n");
        exit(EXIT_FAILURE);
    }

    int BATCH = atoi(argv[1]);
    int TLP_thres = atoi(argv[2]);
    int TLP_strategy = atoi(argv[3]);

    int *M;
    int *N;
    int *K;

    M = (int *) malloc(BATCH * sizeof(int));
    N = (int *) malloc(BATCH * sizeof(int));
    K = (int *) malloc(BATCH * sizeof(int));

    std::fstream fs;
    fs.open("../data/data_MN_K_512_128_sort");
    
    if (!fs.is_open()) {
        printf("Error opening input\n");
        exit(EXIT_FAILURE);
    }

    //read matrix config
    for (int i = 0; i < BATCH; ++i) {
        fs >> M[i] >> N[i] >> K[i];
    }

    float **A;
    float **B;
    float **C;

    A = (float **) malloc(BATCH * sizeof(float *));
    B = (float **) malloc(BATCH * sizeof(float *));
    C = (float **) malloc(BATCH * sizeof(float *));
  int qq=0;
    for (int i = 0; i < BATCH; ++i) {
        ErrChk(hipMalloc((void **) &A[i], M[i] * K[i] * sizeof(float)));
        ErrChk(hipMalloc((void **) &B[i], K[i] * N[i] * sizeof(float)));
        ErrChk(hipMalloc((void **) &C[i], M[i] * N[i] * sizeof(float)));
	for(int r=0;r<M[i];r++){
	    for(int c=0;c<K[i];c++){
	        A[i][r*K[i]+c]=i+1;
	    }
	}
	for(int r=0;r<K[i];r++){
	    for(int c=0;c<N[i];c++){
	        B[i][r*N[i]+c]=i+2;
	    }
	}
	for(int r=0;r<M[i];r++){
	    for(int c=0;c<N[i];c++){
	        C[i][r*N[i]+c]=i+3;
	    }
	}
       

    }
    

    float **dev_A;
    float **dev_B;
    float **dev_C;

    ErrChk(hipMalloc((void **) &dev_A, BATCH * sizeof(float *)));
    ErrChk(hipMalloc((void **) &dev_B, BATCH * sizeof(float *)));
    ErrChk(hipMalloc((void **) &dev_C, BATCH * sizeof(float *)));

 

    int *dev_M, *dev_N, *dev_K;
    ErrChk(hipMalloc((void **) &dev_M, BATCH * sizeof(int)));
    ErrChk(hipMalloc((void **) &dev_N, BATCH * sizeof(int)));
    ErrChk(hipMalloc((void **) &dev_K, BATCH * sizeof(int)));



    float elapsedTime = 0.f;
    double time = 0.f;
    float gflops_per_sec = 0.f;
    double gflops = 0.f;
    for (int i = 0; i < BATCH; ++i)
        gflops += ((2 * int64_t(M[i]) * int64_t(N[i]) * int64_t(K[i])) + (2 * int64_t(M[i]) * int64_t(N[i]))) / 1.0e9;
    hipEvent_t start, stop;

    //compute grid size and block size

    int TLP = 0;

    const int tile_size[3][2] = {
            16, 16,
            32, 32,
            64, 64,
    };

    int *t_strategy;
    t_strategy = (int *) malloc(BATCH * sizeof(int));

    int t;
    for (t = 0; t < 3; ++t) {
        TLP = 0;
        for (int j = 0; j < BATCH; ++j)
            TLP += (M[j] / tile_size[t][0]) * (N[j] / tile_size[t][1]) * 128;

        if (TLP < TLP_thres)
            break;
    }

    for (int j = 0; j < BATCH; ++j) {

        t_strategy[j] = 0;
        t = (t >= 3 ? 2 : t);

        if (tile_size[t][0] <= M[j] && tile_size[t][1] <= N[j]&&(M[j]%tile_size[t][0]==0)&&(N[j]%tile_size[t][1]==0))
            t_strategy[j] = t;
        else {
            for (int k = 0; k < t; ++k) {
                if (tile_size[k][0] <= M[j] && tile_size[k][1] <= N[j]&&(M[j]%tile_size[k][0]==0)&&(N[j]%tile_size[k][1]==0)) {
                    t_strategy[j] = k;
                }
            }
        }
    }



	//print the obtained tiling strategy
	

    if(1){
    //calculate total workgroups
    int total_wg = 0;
    for (int j = 0; j < BATCH; ++j) {
        int tmp_wg = 0;
        switch (t_strategy[j]) {
            case 0:
                tmp_wg = M[j] * N[j] / (16 * 16);
                break;
            case 1:
                tmp_wg = M[j] * N[j] / (32 * 32);
                break;
            case 2:
                tmp_wg = M[j] * N[j] / (64 * 64);
                break;
        }
        total_wg += tmp_wg;
    }

    for (int j = BATCH - 1; j >= 0&& total_wg <TLP_strategy; j--) {
        if (t_strategy[j] == 0) continue;
        if (t_strategy[j] == 1) {
            total_wg += (M[j] * N[j] / (32 * 32)) * 3;
            t_strategy[j] = 0;
            continue;	
	}
        if (t_strategy[j] == 2) {
            total_wg += (M[j] * N[j] / (64 * 64)) * 3;
            t_strategy[j] = 1;
            continue;
        }
    }

//    for(int j = 0;j<BATCH;j++) printf("%d ",t_strategy[j]);
//    printf("\n");
//    printf("%d	", total_wg);
    }




    // for (int j = 0; j < BATCH; ++j)
    // {
    //     t_strategy[j] = 0;
    // }
	// for (int j=0; j<BATCH; ++j)
	// 	printf("%d ", t_strategy[j]);
	// printf("\n");

    int *dev_T;
    ErrChk(hipMalloc((void **) &dev_T, BATCH * sizeof(int)));
  
    dim3 block_size;
    block_size.x = 128;
    block_size.y = 1;
    block_size.z = 1;

    dim3 grid_size;

    grid_size.x = M[0] / tile_size[t_strategy[0]][0];
    grid_size.y = N[0] / tile_size[t_strategy[0]][1];
    grid_size.z = BATCH;
    for (int j = 1; j < BATCH; ++j) {
        grid_size.x = (grid_size.x > M[j] / tile_size[t_strategy[j]][0]) ? (grid_size.x) : (M[j] /
                                                                                            tile_size[t_strategy[j]][0]);
        grid_size.y = (grid_size.y > N[j] / tile_size[t_strategy[j]][1]) ? (grid_size.y) : (N[j] /
                                                                                            tile_size[t_strategy[j]][1]);
    }
// printf("++++++++%d %d %d\n", grid_size.x, grid_size.y, grid_size.z);

    ErrChk(hipMemcpy(dev_T, t_strategy, BATCH * sizeof(int), hipMemcpyHostToDevice));

    ErrChk(hipMemcpy(dev_A, A, BATCH * sizeof(float *), hipMemcpyHostToDevice));
    ErrChk(hipMemcpy(dev_B, B, BATCH * sizeof(float *), hipMemcpyHostToDevice));
    ErrChk(hipMemcpy(dev_C, C, BATCH * sizeof(float *), hipMemcpyHostToDevice));

    ErrChk(hipMemcpy(dev_M, M, BATCH * sizeof(int), hipMemcpyHostToDevice));
    ErrChk(hipMemcpy(dev_N, N, BATCH * sizeof(int), hipMemcpyHostToDevice));
    ErrChk(hipMemcpy(dev_K, K, BATCH * sizeof(int), hipMemcpyHostToDevice));


    // printf("************\n");
    // for(int i=0;i<BATCH;i++){
    //     printf("%d %d %d\n",M[i],N[i],K[i]);
    // }
    //warm-up
   hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm<128>), dim3(grid_size), dim3(block_size), 16*128*16, 0,
                      dev_M, dev_N, dev_K, dev_A, dev_B, dev_C, dev_T);
   
   KernelErrChk();

   
  
   
    ErrChk(hipEventCreate(&start));
    ErrChk(hipEventRecord(start, 0));



    for (int run = 0; run < N_RUNS; ++run) {

        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm<128>), dim3(grid_size), dim3(block_size), 16*128*16,
                           0, dev_M, dev_N, dev_K, dev_A, dev_B, dev_C, dev_T);
        KernelErrChk();
    }



    ErrChk(hipEventCreate(&stop));
    ErrChk(hipEventRecord(stop, 0));
    ErrChk(hipEventSynchronize(stop));
    ErrChk(hipEventElapsedTime(&elapsedTime, start, stop));

    time = elapsedTime / N_RUNS;
    time /= 1.0e3; //convert time unit from millisecond to second
    // gflops_per_sec = gflops / time;
    // printf("%f\n", gflops_per_sec);
    int epoch=N_RUNS;
    int  block_size_x=block_size.x;
   
    printf("time======%f========N_RUNS====%d===block_size.x%d====\n",time,epoch,block_size_x);
    gflops_per_sec = gflops / time;
   printf("gflops==%f         %f\n", gflops,gflops_per_sec);
   ErrChk(hipMemcpy(dev_A, A, BATCH * sizeof(float *), hipMemcpyDeviceToHost));
    ErrChk(hipMemcpy(dev_B, B, BATCH * sizeof(float *), hipMemcpyDeviceToHost));
    ErrChk(hipMemcpy(dev_C, C, BATCH * sizeof(float *), hipMemcpyDeviceToHost));

    // ErrChk(hipMemcpy(dev_A, A, BATCH * sizeof(float *), hipMemcpyDeviceToHost));
    // ErrChk(hipMemcpy(dev_B, B, BATCH * sizeof(float *), hipMemcpyDeviceToHost));
    // ErrChk(hipMemcpy(dev_C, C, BATCH * sizeof(float *), hipMemcpyDeviceToHost));

    // for(int i=0;i<BATCH;i++) printf("A=%f,B=%f,C=%f\n",A[i][0],B[i][0],C[i][0]);

        // for(int i=0;i<BATCH;i++) printf("A=%f,B=%f,C=%f\n",A[i][0],B[i][0],C[i][0]);
        // for(int i=0;i<BATCH;i++) printf("A=%f,B=%f,C=%f\n",A[i][K[i]-1],B[i][N[i]-1],C[i][N[i]-1]);
//  printf("\n===============C-end========================\n");
//     for (int i = 0; i < BATCH; ++i)
//     {
//         for (int r = 0; r < M[i]; r++)
//         {
//             for (int c = 0; c < N[i]; c++)
//             {
                   
//                  printf("%f  ",C[i][r * N[i] + c]);
//             }
//              printf("\n");
//         }
//         printf("\n=======================================\n");
//     }

    for (int i = 0; i < BATCH; ++i) {
        ErrChk(hipFree(A[i]));
        ErrChk(hipFree(B[i]));
        ErrChk(hipFree(C[i]));
    }

    free(M);
    free(N);
    free(K);
    free(A);
    free(B);
    free(C);
    free(t_strategy);

    ErrChk(hipFree(dev_M));
    ErrChk(hipFree(dev_N));
    ErrChk(hipFree(dev_K));
    ErrChk(hipFree(dev_T));

    ErrChk(hipFree(dev_A));
    ErrChk(hipFree(dev_B));
    ErrChk(hipFree(dev_C));

    return 0;
}
