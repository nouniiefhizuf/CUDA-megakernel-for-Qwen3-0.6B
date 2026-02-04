/**
 * Direct measurement of grid.sync() and cooperative kernel overhead
 *
 * This kernel measures:
 * 1. Cost of grid.sync() in isolation
 * 2. Cost of cooperative kernel launch vs regular launch
 * 3. Cost comparison with equivalent work done via separate kernels
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Configuration matching the megakernel
constexpr int NUM_BLOCKS = 82;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_SYNCS = 225;  // Approx syncs per decode step

// =============================================================================
// Test 1: Measure grid.sync() cost in cooperative kernel
// =============================================================================
__global__ void coop_with_syncs(
    int num_syncs,
    volatile int* __restrict__ counter  // Prevent optimization
) {
    cg::grid_group grid = cg::this_grid();

    // Do minimal work between syncs (like the megakernel pattern)
    for (int i = 0; i < num_syncs; i++) {
        // Simulate minimal work between syncs
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd((int*)counter, 1);
        }
        grid.sync();
    }
}

// =============================================================================
// Test 2: Cooperative kernel with no syncs (measure launch overhead)
// =============================================================================
__global__ void coop_no_syncs(
    volatile int* __restrict__ counter
) {
    cg::grid_group grid = cg::this_grid();

    // Single sync at end
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd((int*)counter, 1);
    }
    grid.sync();
}

// =============================================================================
// Test 3: Regular kernel (for comparison)
// =============================================================================
__global__ void regular_kernel(
    volatile int* __restrict__ counter
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd((int*)counter, 1);
    }
}

// =============================================================================
// Benchmark driver
// =============================================================================

void benchmark_grid_sync() {
    printf("="*60 + "\n");
    printf("GRID.SYNC() OVERHEAD ANALYSIS\n");
    printf("="*60 + "\n\n");

    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    void* kernel_args_warmup[] = {(void*)&NUM_SYNCS, (void*)&d_counter};
    cudaLaunchCooperativeKernel(
        (void*)coop_with_syncs,
        dim3(NUM_BLOCKS),
        dim3(BLOCK_SIZE),
        kernel_args_warmup
    );
    cudaDeviceSynchronize();

    // ==========================================================================
    // Test 1: Cooperative kernel with 225 syncs
    // ==========================================================================
    int num_iterations = 100;
    int num_syncs = NUM_SYNCS;

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        void* kernel_args[] = {(void*)&num_syncs, (void*)&d_counter};
        cudaLaunchCooperativeKernel(
            (void*)coop_with_syncs,
            dim3(NUM_BLOCKS),
            dim3(BLOCK_SIZE),
            kernel_args
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_with_syncs;
    cudaEventElapsedTime(&ms_with_syncs, start, stop);
    float us_per_call_syncs = (ms_with_syncs / num_iterations) * 1000;
    float us_per_sync = us_per_call_syncs / NUM_SYNCS;

    printf("Cooperative kernel with %d grid.sync():\n", NUM_SYNCS);
    printf("  Total time: %.2f us/call\n", us_per_call_syncs);
    printf("  Per sync:   %.2f us\n", us_per_sync);

    // ==========================================================================
    // Test 2: Cooperative kernel with 1 sync (measure launch overhead)
    // ==========================================================================
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        void* kernel_args[] = {(void*)&d_counter};
        cudaLaunchCooperativeKernel(
            (void*)coop_no_syncs,
            dim3(NUM_BLOCKS),
            dim3(BLOCK_SIZE),
            kernel_args
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_no_syncs;
    cudaEventElapsedTime(&ms_no_syncs, start, stop);
    float us_per_call_no_syncs = (ms_no_syncs / num_iterations) * 1000;

    printf("\nCooperative kernel with 1 sync:\n");
    printf("  Total time: %.2f us/call\n", us_per_call_no_syncs);
    printf("  Launch overhead: %.2f us\n", us_per_call_no_syncs);

    // ==========================================================================
    // Test 3: Regular kernel launch (baseline)
    // ==========================================================================
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        regular_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_counter);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_regular;
    cudaEventElapsedTime(&ms_regular, start, stop);
    float us_per_regular = (ms_regular / num_iterations) * 1000;

    printf("\nRegular kernel launch:\n");
    printf("  Total time: %.2f us/call\n", us_per_regular);

    // ==========================================================================
    // Test 4: 225 regular kernel launches (simulate split approach)
    // ==========================================================================
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        for (int j = 0; j < NUM_SYNCS; j++) {
            regular_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_counter);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_many_regular;
    cudaEventElapsedTime(&ms_many_regular, start, stop);
    float us_per_many_regular = (ms_many_regular / num_iterations) * 1000;

    printf("\n%d regular kernel launches (simulating split):\n", NUM_SYNCS);
    printf("  Total time: %.2f us\n", us_per_many_regular);
    printf("  Per launch: %.2f us\n", us_per_many_regular / NUM_SYNCS);

    // ==========================================================================
    // Analysis
    // ==========================================================================
    printf("\n" + "="*60 + "\n");
    printf("ANALYSIS\n");
    printf("="*60 + "\n");

    printf("\nCoop with syncs vs many regular launches:\n");
    printf("  Cooperative: %.2f us\n", us_per_call_syncs);
    printf("  Split:       %.2f us\n", us_per_many_regular);
    printf("  Difference:  %.2f us (%+.1f%%)\n",
           us_per_call_syncs - us_per_many_regular,
           (us_per_call_syncs - us_per_many_regular) / us_per_many_regular * 100);

    printf("\nBreakdown:\n");
    printf("  Coop launch overhead: %.2f us\n", us_per_call_no_syncs);
    printf("  grid.sync() overhead: %.2f us (%.2f us each)\n",
           us_per_call_syncs - us_per_call_no_syncs,
           (us_per_call_syncs - us_per_call_no_syncs) / NUM_SYNCS);
    printf("  Regular launch:       %.2f us each\n", us_per_regular);

    cudaFree(d_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    benchmark_grid_sync();
    return 0;
}
