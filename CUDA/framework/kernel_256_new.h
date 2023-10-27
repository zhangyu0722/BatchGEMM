
__device__ void gemm_256_32x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

	float4 reg_C[2];
	float2 reg_A;
	float  reg_B[4];

	//Load C from global memory to register file
	float4 *C_start = (float4*)(C + blockIdx.y*64 + blockIdx.x*32*N + (threadIdx.x % 16) * 4 + (threadIdx.x / 16) *2*N);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start+(N/4));

// 	//load B from global memory to shared memory
	float *A_start = A + blockIdx.x *32*K  + (threadIdx.x/32) + (threadIdx.x%32)*K;
	*(sh_A + threadIdx.x) = *(A_start);

// 	//load A from global memory to shared memory
	float2 *B_start = (float2*)(B + blockIdx.y*64 + (threadIdx.x%32)*2+ (threadIdx.x/32)*N); 
	*(float2*)(sh_B + 2*threadIdx.x) = *(B_start);

	int A_double_buffer = 0;
	int B_double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = A_double_buffer + (threadIdx.x/16)*2;
		int B_offset = B_double_buffer + (threadIdx.x%16)*4;
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			reg_A = *((float2*) (sh_A + A_offset)); 

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+1];  
			reg_B[2] = sh_B[B_offset+2]; 
			reg_B[3] = sh_B[B_offset+3]; 

			reg_C[0].x = fma(reg_A.x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A.x, reg_B[1], reg_C[0].y);
			reg_C[0].z = fma(reg_A.x, reg_B[2], reg_C[0].z);
			reg_C[0].w = fma(reg_A.x, reg_B[3], reg_C[0].w);

			reg_C[1].x = fma(reg_A.y, reg_B[0], reg_C[1].x);
			reg_C[1].y = fma(reg_A.y, reg_B[1], reg_C[1].y);
			reg_C[1].z = fma(reg_A.y, reg_B[2], reg_C[1].z);
			reg_C[1].w = fma(reg_A.y, reg_B[3], reg_C[1].w);

			A_offset += 32;
			B_offset += 64;
		}

		A_double_buffer ^= 256;
		B_double_buffer ^= 512;


		if (k+8 < K){
			A_start += 8; 			
			*(sh_A + A_double_buffer + threadIdx.x) = *(A_start);

			B_start += 4*N; 
			*(float2*)(sh_B + B_double_buffer + 2*threadIdx.x) = *(B_start);
		}
				
 	}
	*(C_start) = reg_C[0];
    *(C_start+(N/4))= reg_C[1];
}
__device__ void gemm_256_64x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[4];
	float4 reg_A;
	float reg_B[4];

	//Load C from global memory to register file
	float4 *C_start = (float4*)(C + blockIdx.y*64 + blockIdx.x*64*N + (threadIdx.x % 16) * 4 + (threadIdx.x / 16) *4*N);
    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + (N/4));
	reg_C[2] = *(C_start + (N/4)*2);
	reg_C[3] = *(C_start + (N/4)*3);
	
	//load A from global memory to shared memory
	float *A_start = A + blockIdx.x *64*K  + (threadIdx.x/32) + (threadIdx.x%32)*2*K;
	*(sh_A + threadIdx.x*2) = *(A_start);
	*(sh_A + threadIdx.x*2+1) = *(A_start+K);
	
	// //load B from global memory to shared memory
	float2 *B_start = (float2*)(B + blockIdx.y*64 + (threadIdx.x%32)*2+ (threadIdx.x/32)*N); 
	*(float2*)(sh_B + 2*threadIdx.x) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x/16)*4;
		int B_offset = double_buffer + (threadIdx.x%16)*4;
			
#pragma unroll
		for (int i=0; i<8; ++i)	{


			reg_A = *((float4*) (sh_A + A_offset)); 


			reg_B[0]=sh_B[B_offset];
			reg_B[1]=sh_B[B_offset+1];
			reg_B[2]=sh_B[B_offset+2];
			reg_B[3]=sh_B[B_offset+3];

			reg_C[0].x = fma(reg_A.x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A.x, reg_B[1], reg_C[0].y);
			reg_C[0].z = fma(reg_A.x, reg_B[2], reg_C[0].z);
			reg_C[0].w = fma(reg_A.x, reg_B[3], reg_C[0].w);

			reg_C[1].x = fma(reg_A.y, reg_B[0], reg_C[1].x);
			reg_C[1].y = fma(reg_A.y, reg_B[1], reg_C[1].y);
			reg_C[1].z = fma(reg_A.y, reg_B[2], reg_C[1].z);
			reg_C[1].w = fma(reg_A.y, reg_B[3], reg_C[1].w);

			reg_C[2].x = fma(reg_A.z, reg_B[0], reg_C[2].x);
			reg_C[2].y = fma(reg_A.z, reg_B[1], reg_C[2].y);
			reg_C[2].z = fma(reg_A.z, reg_B[2], reg_C[2].z);
			reg_C[2].w = fma(reg_A.z, reg_B[3], reg_C[2].w);

			reg_C[3].x = fma(reg_A.w, reg_B[0], reg_C[3].x);
			reg_C[3].y = fma(reg_A.w, reg_B[1], reg_C[3].y);
			reg_C[3].z = fma(reg_A.w, reg_B[2], reg_C[3].z);
			reg_C[3].w = fma(reg_A.w, reg_B[3], reg_C[3].w);




			A_offset += 64;
			B_offset += 64;
			
		}

		double_buffer ^= 512;

		if (k+8 < K){	
			A_start += 4; 
				*(sh_A +double_buffer+ threadIdx.x*2) = *(A_start);
	            *(sh_A + double_buffer+threadIdx.x*2+1) = *(A_start+K);
			B_start += 4*N; 
			*((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
		}
				
	}
	*(C_start) = reg_C[0];
	*(C_start + (N/4)) = reg_C[1];
	*(C_start + (N/4)*2) = reg_C[2];
	*(C_start + (N/4)*3) = reg_C[3];
}



