__device__ void gemm_128_16x16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*16*16;

	float2 reg_C;
	float  reg_A;
	float2 reg_B;

	//Load C from global memory to register file
    float2 *C_start = (float2 *)(C + blockIdx.y*16 + blockIdx.x*16*N + (threadIdx.x % 8) * 2 + (threadIdx.x / 8) *N);
	reg_C = *C_start;

	//load A from global memory to shared memory
	float *A_start = A + blockIdx.x *16*K  + (threadIdx.x/8) + (threadIdx.x%8)*2*K;
	*(sh_A + 2*threadIdx.x) = *(A_start);
	*(sh_A + 2*threadIdx.x+1) = *(A_start+K);

	//load A from global memory to shared memory
	float2 *B_start = (float2*)(B + blockIdx.y*16 + (threadIdx.x%8)*2 + (threadIdx.x/8)*N); 
	*(float2*)(sh_B + threadIdx.x*2) = *(B_start);

	int double_buffer = 0;
	
#pragma unroll
	for(int k=0; k<K; k+=16){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x/16);
		int B_offset = double_buffer + (threadIdx.x%8)*2;
		
			
#pragma unroll
		for (int i=0; i<16; i++)	{
			
			reg_A = sh_A[A_offset];
			

			reg_B.x = sh_B[B_offset];
            reg_B.y = sh_B[B_offset+1];

			reg_C.x = fma(reg_A, reg_B.x, reg_C.x);
			reg_C.y = fma(reg_A, reg_B.y, reg_C.y);

           
			A_offset += 16;
			B_offset += 16;
			
		}

		double_buffer ^= 256;

		if (k+16 < K){
			A_start += 8; 
			*(sh_A + double_buffer+2*threadIdx.x) = *(A_start);
	        *(sh_A + double_buffer+2*threadIdx.x+1) = *(A_start+K);
			B_start += 8*N; 
			*(float2*)(sh_B + double_buffer + 2*threadIdx.x) = *(B_start);
		}
	
}	
    *C_start = reg_C;

}
__device__ void gemm_128_16x32(int M, int N, int K, float *A, float *B, float *C, float *sh){

	float *sh_A = sh;
	float *sh_B = sh + 2*16*8;

	float2  reg_C[2];
	float2  reg_A;
	float2  reg_B;


	//Load C from global memory to register file
	float2 *C_start = (float2 *)(C + blockIdx.y*32 + blockIdx.x*16*N + (threadIdx.x % 16) * 2 + (threadIdx.x / 16) *2*N);
	reg_C[0] =*(C_start);
	reg_C[1]=*(C_start+(N/2));

	//load A from global memory to shared memory
	float *A_start = A + blockIdx.x *16*K  + (threadIdx.x/16) + (threadIdx.x%16)*K;
	*(sh_A + threadIdx.x) = *(A_start);

	//load A from global memory to shared memory
	float2 *B_start = (float2*)(B + blockIdx.y*32 + (threadIdx.x%16)*2 + (threadIdx.x/16)*N); 
	*(float2*)(sh_B + threadIdx.x*2) = *(B_start);


	int A_double_buffer = 0;
	int B_double_buffer=0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = A_double_buffer + (threadIdx.x/8)*2;
		int B_offset = B_double_buffer + (threadIdx.x%16)*2;
			 
#pragma unroll
		for (int i=0; i<8; i++)	{
			
			reg_A.x = sh_A[A_offset];
			reg_A.y = sh_A[A_offset+1];

			reg_B.x = sh_B[B_offset];
			reg_B.y = sh_B[B_offset+1];

			reg_C[0].x = fma(reg_A.x, reg_B.x, reg_C[0].x);
			reg_C[0].y = fma(reg_A.x, reg_B.y, reg_C[0].y);

			reg_C[1].x = fma(reg_A.y, reg_B.x, reg_C[1].x);
			reg_C[1].y = fma(reg_A.y, reg_B.y, reg_C[1].y);

			A_offset += 16;
			B_offset += 32;
		}

		A_double_buffer ^= 128;
		B_double_buffer ^= 256;

		if (k+8 < K){
			A_start += 8; 
			*(sh_A + A_double_buffer + threadIdx.x) = *(A_start);

			B_start += 4*N; 
			*((float2*)(sh_B + B_double_buffer +2*threadIdx.x)) = *(B_start);
		}
	}
	
   *(C_start)=reg_C[0];
	*(C_start+(N/2))=reg_C[1];
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



