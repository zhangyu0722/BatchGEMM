# Introduction
This repository provides a collection of kernel functions that enable high-speed computation of Batch GEMM (general matrix-matrix multiplication).   This repository contains the implementation and benchmark code on two GPU platforms (AMD and NVIDIA).  This repository is the source code of "A load-balanced parallel framework for batch general matrix multiplication on GPU".

If you have any questions about running the code, please contact Yu Zhang.  
Email：yuzhang0722@163.com

# Description
The following is the directory tree of the code, which contains CUDA and ROCm code.       
* In **ROCm** directory:  
  * **data:**  In this sub-directory, we provide a gen_data binary to generate the dataset used in the following evaluation.  
  * **include:**  This sub-directory contains a series header file for ROCm platform.  
  * **default：**  This sub-directory contains source core of rocBLAS for batch GEMM.  
  * **tiling:**  This sub-directory contains source code of Wang ("A high‑performance batched matrix multiplication framework for GPUs under unbalanced input distribution") in this paper.  
  * **framework:**  This sub-directory contains source coe of a load-balanced parallel framework for batch GEMM in ROCm.
* In **CUDA** directory:  
  * **data:**  In this sub-directory, we provide a gen_data binary to generate the dataset used in the following evaluation.  
  * **include:**  This sub-directory contains a series header file for CUDA platform.  
  * **default：**  This sub-directory contains source core of cuBLAS for batch GEMM.  
  * **batching:**  This sub-directory contains source code of Li ("A coordinated tiling and batching framework for efficient gemm on gpus") in this paper.  
  * **framework:**  This sub-directory contains source coe of a load-balanced parallel framework for batch GEMM in CUDA.
 
**magma:** download [https://icl.utk.edu/magma/](https://icl.utk.edu/magma/). Download the relevant version of the code (magma 2.6.0 or newer)and adjust the configuration for the different GPU platforms. The installation tutorial is available on the official website.
# Requirements
* ROCm5.4 and CUDA 11.7 or newer
* Ubuntu 20.04
* C++14 or newer
* CMake >=3.6
* GPU Architecture : AMD = CDNA 2.0 or CDNA 1.0, NVIDA = Ampere

 # Release version
 November 2023 Version Alpha
