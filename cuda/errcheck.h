#pragma once

#define CUDA_SAFE_CALL(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUDA_KERNEL_ERRCHECK do {CUDA_SAFE_CALL( cudaPeekAtLastError() ); CUDA_SAFE_CALL( cudaDeviceSynchronize() );} while (0)
