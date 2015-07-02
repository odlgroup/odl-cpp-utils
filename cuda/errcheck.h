#pragma once

#define CUDA_SAFE_CALL(ans) do { gpuAssert((ans), __FILE__, __LINE__, #ans); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line, const char* call, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d call: %s\n", cudaGetErrorString(code), file, line,call);
      if (abort) exit(code);
   }
}

#define CUDA_KERNEL_ERRCHECK do {CUDA_SAFE_CALL( cudaPeekAtLastError() ); CUDA_SAFE_CALL( cudaDeviceSynchronize() );} while (0)
