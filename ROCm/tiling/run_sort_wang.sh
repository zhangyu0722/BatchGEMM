#!/bin/bash
threshold=120
direction=1
TLP=220
rm -f log-verify-wang
	for ((M=128; M<=128; M=M*2))
	do
		for ((K=128; K<=128; K=K*2))
		do
			
			
			./gemm_wang 8 $threshold $TLP >>  log-verify-wang
			
			./gemm_wang 16 $threshold $TLP >>  log-verify-wang
			
			./gemm_wang 32 $threshold $TLP >>  log-verify-wang
	
			./gemm_wang 64 $threshold $TLP >>  log-verify-wang
		
			./gemm_wang 128 $threshold $TLP >>  log-verify-wang

			./gemm_wang 256 $threshold $TLP >>  log-verify-wang
		
			echo "ok==================================" >> log-verify-wang
		done
	done
