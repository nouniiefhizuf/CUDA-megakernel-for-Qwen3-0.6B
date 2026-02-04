"""
Direct measurement of grid.sync() and cooperative kernel overhead.

Measures:
1. Cost of grid.sync() in isolation
2. Cost of cooperative kernel launch vs regular launch
3. Cost comparison with equivalent work done via separate kernels
"""

import torch
import time


def run_experiment():
    cuda_src = r'''
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int NUM_BLOCKS = 82;
constexpr int BLOCK_SIZE = 256;

// Cooperative kernel with N syncs
__global__ void coop_with_syncs(int num_syncs, int* counter) {
    cg::grid_group grid = cg::this_grid();

    for (int i = 0; i < num_syncs; i++) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(counter, 1);
        }
        grid.sync();
    }
}

// Regular kernel
__global__ void regular_kernel(int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(counter, 1);
    }
}

// Measure cooperative kernel with syncs
extern "C" float measure_coop_syncs(int num_syncs, int num_iterations) {
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Warmup
    void* args[] = {&num_syncs, &d_counter};
    cudaLaunchCooperativeKernel((void*)coop_with_syncs, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), args);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        void* args[] = {&num_syncs, &d_counter};
        cudaLaunchCooperativeKernel((void*)coop_with_syncs, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), args);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / num_iterations;  // ms per call
}

// Measure regular kernel launches
extern "C" float measure_regular_launches(int num_launches, int num_iterations) {
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Warmup
    for (int i = 0; i < num_launches; i++) {
        regular_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_counter);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        for (int j = 0; j < num_launches; j++) {
            regular_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_counter);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / num_iterations;  // ms per call
}

// Measure CUDA graph replay
extern "C" float measure_cuda_graph(int num_kernels, int num_iterations) {
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Capture graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < num_kernels; i++) {
        regular_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(d_counter);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Warmup
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < num_iterations; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / num_iterations;  // ms per call
}
'''

    cpp_src = r'''
extern "C" float measure_coop_syncs(int num_syncs, int num_iterations);
extern "C" float measure_regular_launches(int num_launches, int num_iterations);
extern "C" float measure_cuda_graph(int num_kernels, int num_iterations);

float py_measure_coop_syncs(int num_syncs, int num_iterations) {
    return measure_coop_syncs(num_syncs, num_iterations);
}

float py_measure_regular_launches(int num_launches, int num_iterations) {
    return measure_regular_launches(num_launches, num_iterations);
}

float py_measure_cuda_graph(int num_kernels, int num_iterations) {
    return measure_cuda_graph(num_kernels, num_iterations);
}
'''

    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="sync_overhead",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["py_measure_coop_syncs", "py_measure_regular_launches", "py_measure_cuda_graph"],
        extra_cuda_cflags=["-O3", "-use_fast_math"],
        verbose=False,
    )

    NUM_SYNCS = 225  # Approx syncs per decode step (8 per layer * 28 layers + 1)
    NUM_ITERATIONS = 100

    print("=" * 60)
    print("GRID.SYNC() AND KERNEL LAUNCH OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"\nConfiguration: {NUM_SYNCS} sync points (matching megakernel)")
    print(f"Grid: 82 blocks x 256 threads")
    print()

    # Test 1: Cooperative kernel with syncs
    ms_coop = module.py_measure_coop_syncs(NUM_SYNCS, NUM_ITERATIONS)
    us_coop = ms_coop * 1000

    print(f"1. Cooperative kernel with {NUM_SYNCS} grid.sync():")
    print(f"   Total: {us_coop:.1f} us")
    print(f"   Per sync: {us_coop / NUM_SYNCS:.2f} us")
    print()

    # Test 2: Cooperative kernel with 1 sync (launch overhead)
    ms_coop_1 = module.py_measure_coop_syncs(1, NUM_ITERATIONS)
    us_coop_1 = ms_coop_1 * 1000

    print(f"2. Cooperative kernel with 1 sync (launch overhead):")
    print(f"   Total: {us_coop_1:.1f} us")
    print()

    # Test 3: Regular kernel launches
    ms_regular = module.py_measure_regular_launches(NUM_SYNCS, NUM_ITERATIONS)
    us_regular = ms_regular * 1000

    print(f"3. {NUM_SYNCS} regular kernel launches:")
    print(f"   Total: {us_regular:.1f} us")
    print(f"   Per launch: {us_regular / NUM_SYNCS:.2f} us")
    print()

    # Test 4: CUDA graph
    ms_graph = module.py_measure_cuda_graph(NUM_SYNCS, NUM_ITERATIONS)
    us_graph = ms_graph * 1000

    print(f"4. CUDA graph with {NUM_SYNCS} kernels:")
    print(f"   Total: {us_graph:.1f} us")
    print(f"   Per kernel (amortized): {us_graph / NUM_SYNCS:.2f} us")
    print()

    # Analysis
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    sync_overhead = us_coop - us_coop_1
    sync_per = sync_overhead / (NUM_SYNCS - 1)  # -1 since we already have 1 in baseline

    print(f"\nPure grid.sync() overhead: {sync_overhead:.1f} us total")
    print(f"Per grid.sync(): {sync_per:.2f} us")
    print()

    print(f"Comparison for {NUM_SYNCS} synchronization points:")
    print(f"  Cooperative + grid.sync(): {us_coop:.1f} us")
    print(f"  Separate kernel launches:  {us_regular:.1f} us")
    print(f"  CUDA graph:                {us_graph:.1f} us")
    print()

    winner = "CUDA graph" if us_graph < us_coop else "Cooperative"
    diff = abs(us_coop - us_graph)
    print(f"Winner (for sync overhead only): {winner} by {diff:.1f} us")
    print()

    print("=" * 60)
    print("IMPLICATIONS FOR MEGAKERNEL")
    print("=" * 60)
    print("""
The above measures ONLY synchronization/launch overhead, NOT:
- Memory bandwidth (megakernel avoids intermediate writes)
- Register reuse (megakernel keeps values in registers across ops)
- Cache efficiency (megakernel has better L1/L2 locality)

Real megakernel performance at 527 tok/s = 1.9 ms/token
Pure sync overhead: %.1f us (%.1f%% of decode time)

If memory bandwidth dominates (which it likely does for this
memory-bound kernel), the fusion benefits far outweigh any
synchronization overhead differences.
""" % (us_coop, us_coop / 1900 * 100))

    return {
        "coop_syncs_us": us_coop,
        "coop_1sync_us": us_coop_1,
        "regular_launches_us": us_regular,
        "cuda_graph_us": us_graph,
    }


if __name__ == "__main__":
    run_experiment()
