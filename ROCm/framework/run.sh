#!/bin/bash
threshold=512
direction=1
TLP=6000

rm -f log-test-verify3
	for ((M=512; M<=512; M=M*2))
	do
		for ((K=1024; K<=1024; K=K*2))
		do
			 
			
			./gemm 8 $threshold $TLP >>  log-test-verify3
		
			./gemm  16 $threshold $TLP >>  log-test-verify3
			
			./gemm  32 $threshold $TLP >>  log-test-verify3
	
			./gemm  64 $threshold $TLP >>  log-test-verify3
			
			./gemm  128 $threshold $TLP >>  log-test-verify3
			
			./gemm  256 $threshold $TLP >>  log-test-verify3

			# ./zy_gemm3 512 $threshold $TLP >>  log-test-verify3

			# ./zy_gemm3 1024 $threshold $TLP >>  log-test-verify3
			echo "ok==================================" >> log-test-verify3
		done
	done

