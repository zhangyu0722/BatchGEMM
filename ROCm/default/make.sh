#!/bin/bash

FILE="gemm.cuh"
INCLUDE="-I /opt/rocm/hip/include -I /opt/rocm/rocblas/include -I /opt/rocm/hipblas/include"
LIBRARIES="-L /opt/rocm/rocblas/lib -L /opt/rocm/hipblas/lib -L /opt/rocm/hip/lib"
LINK_LIBRARIES="-lrocblas -lhipblas"
BIN="./gemm"
hipcc ${FILE} -O3 ${INCLUDE} ${LIBRARIES} ${LINK_LIBRARIES} -o ${BIN}

