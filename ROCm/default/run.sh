#!/bin/bash

rm -f log-default
for ((M=1024; M<=1024; M=M*2))
do
	for ((K=1024; K<=1024; K=K*2))
	do
		./gemm 8 >> log-default
		./gemm 16 >> log-default
		./gemm 32 >> log-default
		./gemm 64 >> log-default
		./gemm 128 >> log-default
		./gemm 256 >> log-default
		echo "ok==================================" >> log-default
	done
done
