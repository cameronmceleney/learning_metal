//
// Created by Cameron Aidan McEleney on 27/03/2024.
//

#include <metal_stdlib>
using namespace metal;

struct CoreParamsForDevice {
    // Must match the host's definition in the .h file!
    size_t N;
    size_t bytes;
};

kernel void simpleVectorAdd( const device int *inA [[ buffer(0) ]],
                             const device int *inB [[ buffer(1) ]],
                             device int *out [[ buffer(2) ]],
                             const device CoreParamsForDevice &params [[ buffer(3) ]],
                             uint blockIdx [[ threadgroup_position_in_grid ]],
                             uint blockDim [[ threads_per_threadgroup ]],
                             uint threadIdx [[ thread_position_in_threadgroup ]],
                             uint gid [[ thread_position_in_grid ]]
                             ) {
    // Manual calculation of global Thread ID. This should equal gid which MSL provides as a function
    // argument
    uint tid = (blockIdx * blockDim) + threadIdx;

    if (tid < params.N) {
        out[tid] = inA[tid] + inB[tid];
    }
}