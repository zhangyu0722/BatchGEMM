__device__ void gemm_128_16x16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*16*8;

	float2 reg_C;
	float2 reg_A;
	float reg_B;

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*16;
	int block_base_y = blockIdx.x*16;

	//Load C from global memory to register file
	float2 *C_start = (float2*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M);

	reg_C = *C_start;

	//load A from global memory to shared memory
	float *A_start = A + block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M;
	*(sh_A + threadIdx.x) = *(A_start);

	//load A from global memory to shared memory
	float *B_start = B + K*block_base_x + (threadIdx.x/16) + (threadIdx.x%16)*K;
	*(sh_B + threadIdx.x) = *(B_start);


	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*2;
		int B_offset = double_buffer + (threadIdx.x/8);
			
#pragma unroll
		for (int i=0; i<8; i++)	{
			
			reg_A.x = sh_A[A_offset];
			reg_A.y = sh_A[A_offset+1];

			reg_B = sh_B[B_offset];

			reg_C.x = fma(reg_A.x, reg_B, reg_C.x);
			reg_C.y = fma(reg_A.y, reg_B, reg_C.y);

			A_offset += 16;
			B_offset += 16;

		}


		double_buffer ^= 128;

		if (k+8 < K){
			A_start += 8*M; 
			*(sh_A + double_buffer + threadIdx.x) = *(A_start);
			B_start += 8; 
			*(sh_B + double_buffer + threadIdx.x) = *(B_start);
		}
	}
	
    *C_start = reg_C;
}

__device__ void gemm_128_32x32(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

	float4 reg_C[2];
	float4 reg_A;
	float  reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*32;
	int block_base_y = blockIdx.x*32;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M);

	reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 4*M);

	//load A from global memory to shared memory
	float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%16)*2 + (threadIdx.x/16)*M);
	*((float2*)(sh_A + 2*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/32)*2 + (threadIdx.x%32)*K);
	*((float2*)(sh_B + 2*threadIdx.x)) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + (threadIdx.x/8)*2;
			
#pragma unroll
		for (int i=0; i<8; i++)	{
			
			reg_A.x = sh_A[A_offset];
			reg_A.y = sh_A[A_offset+1];
			reg_A.z = sh_A[A_offset+2];
			reg_A.w = sh_A[A_offset+3];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+32];

			reg_C[0].x = fma(reg_A.x, reg_B[0], reg_C[0].x);
			reg_C[0].y = fma(reg_A.y, reg_B[0], reg_C[0].y);
			reg_C[0].z = fma(reg_A.z, reg_B[0], reg_C[0].z);
			reg_C[0].w = fma(reg_A.w, reg_B[0], reg_C[0].w);
			reg_C[1].x = fma(reg_A.x, reg_B[1], reg_C[1].x);
			reg_C[1].y = fma(reg_A.y, reg_B[1], reg_C[1].y);
			reg_C[1].z = fma(reg_A.z, reg_B[1], reg_C[1].z);
			reg_C[1].w = fma(reg_A.w, reg_B[1], reg_C[1].w);

			A_offset += 32;
			B_offset += ((i%2)*62 + 1);
		}

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += 4*M; 
			*((float2*)(sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
			B_start += 4; 
			*((float2*)(sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
		}
	}
	
    *C_start = reg_C[0];
    *(C_start + 4*M) = reg_C[1];
}

__device__ void gemm_128_64x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[8];
	float4 reg_A;
	float  reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 8*M;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/64)*4 + (threadIdx.x%64)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
		
	int double_buffer = 0;

#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%16)*4;
		int B_offset = double_buffer + ((threadIdx.x/16)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A = *((float4*)(sh_A + A_offset));
			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+128];
			reg_B[5] = sh_B[B_offset+132];
			reg_B[6] = sh_B[B_offset+136];
			reg_B[7] = sh_B[B_offset+140];

			reg_C[0].x = fma(reg_A.x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A.x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A.x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A.x, reg_B[3], reg_C[3].x);
			reg_C[4].x = fma(reg_A.x, reg_B[4], reg_C[4].x);
			reg_C[5].x = fma(reg_A.x, reg_B[5], reg_C[5].x);
			reg_C[6].x = fma(reg_A.x, reg_B[6], reg_C[6].x);
			reg_C[7].x = fma(reg_A.x, reg_B[7], reg_C[7].x);

			reg_C[0].y = fma(reg_A.y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A.y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A.y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A.y, reg_B[3], reg_C[3].y);
			reg_C[4].y = fma(reg_A.y, reg_B[4], reg_C[4].y);
			reg_C[5].y = fma(reg_A.y, reg_B[5], reg_C[5].y);
			reg_C[6].y = fma(reg_A.y, reg_B[6], reg_C[6].y);
			reg_C[7].y = fma(reg_A.y, reg_B[7], reg_C[7].y);

			reg_C[0].z = fma(reg_A.z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A.z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A.z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A.z, reg_B[3], reg_C[3].z);
			reg_C[4].z = fma(reg_A.z, reg_B[4], reg_C[4].z);
			reg_C[5].z = fma(reg_A.z, reg_B[5], reg_C[5].z);
			reg_C[6].z = fma(reg_A.z, reg_B[6], reg_C[6].z);
			reg_C[7].z = fma(reg_A.z, reg_B[7], reg_C[7].z);

			reg_C[0].w = fma(reg_A.w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A.w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A.w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A.w, reg_B[3], reg_C[3].w);
			reg_C[4].w = fma(reg_A.w, reg_B[4], reg_C[4].w);
			reg_C[5].w = fma(reg_A.w, reg_B[5], reg_C[5].w);
			reg_C[6].w = fma(reg_A.w, reg_B[6], reg_C[6].w);
			reg_C[7].w = fma(reg_A.w, reg_B[7], reg_C[7].w);

			A_offset += 64;
			B_offset += ((i==3)*252 + 1);
		}

		double_buffer ^= 512;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
				
	}
	C_start -= 8*M;
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];
	C_start += 8*M;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

}
