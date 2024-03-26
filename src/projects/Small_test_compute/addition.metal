//
// Created by Cameron Aidan McEleney on 15/03/2024.
//

#include <metal_stdlib>
using namespace metal;

#define THREADGROUP_SIZE 640 // Matches the default SIMD Width of my Macbook Pro
#define LOAD_SIZE 4 // Number of floats loaded per operation

typedef struct {
    device const float* inA [[ buffer(0) ]]; // Input buffer, read-only data
    device const float* inB [[ buffer(1) ]]; // Input buffer, read-only data
    device float* outC [[ buffer(2) ]]; // Output buffer, writable
} ArgumentBuffer;

kernel void add_arrays( const device float* inA [[ buffer(0) ]],
                        const device float* inB [[ buffer(1) ]],
                        device float* outC [[ buffer(2) ]],
                        uint gid [[ thread_position_in_grid ]]) {
    outC[gid] = inA[gid] + inB[gid];
}

kernel void complex_operation( const device float* inA [[ buffer(0) ]],
                               const device float* inB [[ buffer(1) ]],
                               device float* outC [[ buffer(2) ]],
                               uint gid [[ thread_position_in_grid ]]) {
    outC[gid] = sin(inA[gid] * inB[gid]) + inA[gid];
}

kernel void improved_complex_operation( const device ArgumentBuffer& args [[ buffer(0) ]],
                                        uint gid [[thread_position_in_grid]],
                                        uint lid [[thread_position_in_threadgroup]],
                                        uint groupSize [[threads_per_threadgroup]])
{
    // Assuming each thread handled a float4 vector
    uint idx = gid * 4;

        // Manually load and compute each component of the vector.
    float4 vecC;
    for (int i = 0; i < 4; ++i) {
        float a = args.inA[idx + i];
        float b = args.inB[idx + i];
        vecC[i] = sin(a + b) + a; // Perform the operation per component.
    }

    // Store the result back to outC.
    for (int i = 0; i < 4; ++i) {
        args.outC[idx + i] = vecC[i];
    }
}

kernel void further_improved_complex_operation(device ArgumentBuffer& args,
                                               uint thread_pos [[thread_position_in_threadgroup]],
                                               uint threadsPerGroup [[threads_per_threadgroup]])
{
    // Define shared memory for input A, input B, and output C
    threadgroup float4 sharedInA[THREADGROUP_SIZE];
    threadgroup float4 sharedInB[THREADGROUP_SIZE];
    threadgroup float4 sharedOutC[THREADGROUP_SIZE];

    // Calculate unique index for each thread within the threadgroup
    uint idx = thread_pos * LOAD_SIZE;

    // Load data into shared memory (ensure alignment for float4 operations)
    sharedInA[thread_pos] = *((device float4*)(args.inA + idx));
    sharedInB[thread_pos] = *((device float4*)(args.inB + idx));

    // Synchronize to ensure all threads have loaded their data into shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Example computation - can be replaced with more complex operations
    // Now performing the operation on shared memory
    sharedOutC[thread_pos] = sin(sharedInA[thread_pos] * sharedInB[thread_pos]) + sharedInA[thread_pos];

    // Synchronize to ensure all computations are done before writing back to global memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the results from shared memory back to global memory
    *((device float4*)(args.outC + idx)) = sharedOutC[thread_pos];
}
