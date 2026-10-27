// Standalone diagnostic for the original VM startup initialization pattern.
// nvcc -lineinfo -arch=sm_100a native_pi0/barrier_init_probe.cu -o /tmp/barriers
// compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/barriers
//   reproduces missing-init errors on CUDA 12.8 / B200.
// The same command with a final `serial` argument must pass.
#include <cuda_runtime.h>
#include <cstdio>
__global__ void probe(int serial) {
 __shared__ unsigned long long barriers[13];
 if ((serial && threadIdx.x==0) || (!serial && threadIdx.x<13)) {
  int begin=serial?0:threadIdx.x, end=serial?13:threadIdx.x+1;
  for(int i=begin;i<end;++i){
   unsigned addr=__cvta_generic_to_shared(&barriers[i]);
   asm volatile("mbarrier.init.shared::cta.b64 [%0], 16;"::"r"(addr):"memory");
   asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 16;"::"r"(addr):"memory");
  }
 }
 __syncthreads();
}
int main(int argc,char**){probe<<<1,256>>>(argc>1); auto e=cudaDeviceSynchronize();printf("%s\n",cudaGetErrorString(e));return e!=cudaSuccess;}
