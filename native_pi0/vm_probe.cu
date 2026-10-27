// Resource-lifecycle probe, not a model operator or performance benchmark.
#include "kittens.cuh"
#include "megakernel.cuh"
#include <cuda_runtime.h>
#include <vector>

using Config = megakernel::default_config;
__global__ void query_smid_slots(int* slots) {
    if (threadIdx.x == 0) asm volatile("mov.u32 %0, %%nsmid;" : "=r"(*slots));
}
struct Globals {
    megakernel::instruction_layout<Config> instructions;
    megakernel::timing_layout<Config> timings;
    int* output;
};

struct Probe {
    static constexpr int opcode = 1;
    using controller = megakernel::NoOp<Config>::controller;
    using loader = megakernel::NoOp<Config>::loader;
    using launcher = megakernel::NoOp<Config>::launcher;
    using storer = megakernel::NoOp<Config>::storer;
    struct consumer {
        static __device__ void run(const Globals& g, megakernel::state<Config>& s) {
            if (kittens::laneid() == 0) {
                int index = (blockIdx.x * g.instructions.rows() + s.instruction_index) * Config::NUM_CONSUMER_WARPS + kittens::warpid();
                // Verify every consumer observes the same CTA tensor allocation.
                g.output[index] = int(s.tensor_alloc.addr) + 1;
            }
        }
    };
};

extern "C" int run_vm_probe(int workers, int instructions, int* host_output) {
    if (workers < 1 || instructions < 1 || !host_output) return cudaErrorInvalidValue;
    int device; cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaDeviceProp props; err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) return err;
    if (props.major != 10 || props.minor != 0 || workers > props.multiProcessorCount)
        return cudaErrorInvalidConfiguration;
    int *program = nullptr, *timing = nullptr, *output = nullptr, *device_slots = nullptr;
    int slots = 0;
    err = cudaMalloc(&device_slots, sizeof(int));
    if (err == cudaSuccess) {
        query_smid_slots<<<1, 32>>>(device_slots);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaMemcpy(&slots, device_slots, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_slots);
    if (err != cudaSuccess) return err;
    size_t count = size_t(workers) * instructions;
    size_t program_count = size_t(slots) * instructions;
    std::vector<int> code(program_count * Config::INSTRUCTION_WIDTH, 0);
    for (size_t i = 0; i < program_count; ++i) code[i * Config::INSTRUCTION_WIDTH] = 1;
    err = cudaMalloc(&program, code.size() * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(&timing, program_count * Config::TIMING_WIDTH * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(&output, count * Config::NUM_CONSUMER_WARPS * sizeof(int));
    if (err == cudaSuccess) err = cudaMemcpy(program, code.data(), code.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(output, 0, count * Config::NUM_CONSUMER_WARPS * sizeof(int));
    if (err == cudaSuccess) {
        Globals g{megakernel::instruction_layout<Config>(program, nullptr, slots, instructions, nullptr),
                  megakernel::timing_layout<Config>(timing, nullptr, slots, instructions, nullptr), output};
        auto kernel = megakernel::mk<Config, Globals, Probe>;
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::DYNAMIC_SHARED_MEMORY);
        if (err == cudaSuccess) {
            kernel<<<workers, Config::NUM_THREADS, Config::DYNAMIC_SHARED_MEMORY>>>(g);
            err = cudaGetLastError();
        }
        if (err == cudaSuccess) err = cudaMemcpy(host_output, output, count * Config::NUM_CONSUMER_WARPS * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(program); cudaFree(timing); cudaFree(output);
    return err;
}
