#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdio>
#include <cstdlib>
#include "hipblas.h"
#include "rocblas.h"
//include "cudnn.h"


static inline const char* cublasGetErrorString(rocblas_status error)
{
    switch (error)
    {
        case rocblas_status_success:
            return "rocblas_status_success";

        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";

        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";

        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";

        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";

        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";

        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";

        case rocblas_status_perf_degraded:
            return "rocblas_status_perf_degraded";

        case rocblas_status_size_query_mismatch:
            return "rocblas_status_size_query_mismatch";

        case rocblas_status_size_increased:
            return "rocblas_status_size_increased";
	
	case rocblas_status_size_unchanged:
	    return "rocblas_status_size_unchanged";

	case rocblas_status_invalid_value:
	    return "rocblas_status_invalid_value";

	case rocblas_status_continue:
	    return "rocblas_status_continue";
    }
    return "<unknown>";
}


#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
static inline void Assert(hipError_t  code, const char *file, int line){
	if(code!=hipSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line, hipGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}
//static inline void Assert(cudnnStatus_t code, const char *file, int line){
//    if (code!=CUDNN_STATUS_SUCCESS){
//		printf("cuDNN API Error: %s:%d:'%s'\n", file, line, cudnnGetErrorString(code));
//        exit(EXIT_FAILURE);
//    }
//}
static inline void Assert(rocblas_status code, const char *file, int line){
    if (code!=rocblas_status_success){
		printf("cuBLAS API Error: %s:%d:'%s'\n", file, line, cublasGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}


#define KernelErrChk(){\
		hipError_t errSync  = hipGetLastError();\
		hipError_t errAsync = hipDeviceSynchronize();\
		if (errSync != hipSuccess) {\
			  printf("Sync kernel error: %s\n", hipGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != hipSuccess){\
			printf("Async kernel error: %s\n", hipGetErrorString(errAsync));\
			exit(EXIT_FAILURE);\
		}\
}
#endif
