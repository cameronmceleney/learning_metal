//
// Created by Cameron Aidan McEleney on 27/03/2024.
//

#include <metal_stdlib>
using namespace metal;

struct CoreParamsForDevice {
    // Must match the host's definition in the .h file!
    size_t N;
    size_t NUM_ELEMENTS_PER_THREAD;
    ushort NUM_VEC4;
    size_t bytes;
};

kernel void simpleVectorAdd( const device float *inA [[ buffer(0) ]],
                             const device float *inB [[ buffer(1) ]],
                             device float *out [[ buffer(2) ]],
                             const device CoreParamsForDevice &params [[ buffer(3) ]],
                             uint blockIdx [[ threadgroup_position_in_grid ]],
                             uint blockDim [[ threads_per_threadgroup ]],
                             uint threadIdx [[ thread_position_in_threadgroup ]],
                             uint gid [[ thread_position_in_grid ]]
                             ) {
    // Manual calculation of global Thread ID. This should equal gid which MSL provides as a function
    // argument
    //uint tid = (blockIdx * blockDim) + threadIdx;

    // Check boundaries
    if (gid < params.N)
    {
        out[gid] = inA[gid] + inB[gid];
    }
}

kernel void simpleVectorSineAdd( const device float *inA [[ buffer(0) ]],
                             const device float *inB [[ buffer(1) ]],
                             device float *out [[ buffer(2) ]],
                             const device CoreParamsForDevice &params [[ buffer(3) ]],
                             uint blockIdx [[ threadgroup_position_in_grid ]],
                             uint blockDim [[ threads_per_threadgroup ]],
                             uint threadIdx [[ thread_position_in_threadgroup ]],
                             uint gid [[ thread_position_in_grid ]]
                             ) {
    // Manual calculation of global Thread ID. This should equal gid which MSL provides as a function
    // argument
    //uint tid = (blockIdx * blockDim) + threadIdx;

    // Check boundaries
    if (gid < params.N)
    {
        out[gid] = sin(inA[gid] * inB[gid]) + inA[gid];
    }
}

kernel void unrollVectorAdd( const device float4 *inA [[ buffer(0) ]],
                             const device float4 *inB [[ buffer(1) ]],
                             device float4 *out [[ buffer(2) ]],
                             const device CoreParamsForDevice &params [[ buffer(3) ]],
                             uint blockIdx [[ threadgroup_position_in_grid ]],
                             uint blockDim [[ threads_per_threadgroup ]],
                             uint threadIdx [[ thread_position_in_threadgroup ]],
                             uint gid [[ thread_position_in_grid ]]
                             ) {
    // Manual calculation of global Thread ID to show how `gid` is obtained
    // uint tid = (blockIdx * blockDim) + threadIdx;

    // Process four elements per thread (or a multiple of 4 as scaled by NUM_VEC4)
    uint tid = gid * params.NUM_VEC4; // Each thread processes two int4 elements

    for (ushort i = 0; i < params.NUM_VEC4; i++)
    {
        // Increment to access the next int4 element
        tid += i;

        // Check bounds for the current int4 elements
        if (tid * 4 < params.N)
        {
            out[tid] = sin(inA[tid] * inB[tid]) + inA[tid];
        }


    }
}