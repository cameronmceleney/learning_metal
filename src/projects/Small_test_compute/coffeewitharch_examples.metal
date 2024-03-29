//
// Created by Cameron Aidan McEleney on 27/03/2024.
//

#include <metal_stdlib>
using namespace metal;

struct CoreParamsForDevice {
    // Must match the host's definition in the .h file!

     // Total number of elements
    size_t N;

    // Total elements in inputs including the padding
    size_t N_WITH_DATA;

    // Total elements in inputs including the padding
    size_t N_WITH_PADDING;

    // Number of elements per thread
    ushort N_PER_THREAD;

     // Number of float4 elements each thread is responsible for
    ushort N_FLOAT4_PER_THREAD;

    // Size of this struct in bytes. Not used by the kernels
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
    if (gid < params.N_WITH_DATA)
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

    // Process four elements per thread (or a multiple of 4 as scaled by N_FLOAT4_PER_THREAD)
    uint tid = gid * params.N_FLOAT4_PER_THREAD; // Each thread processes two int4 elements

    for (ushort i = 0; i < params.N_FLOAT4_PER_THREAD; i++)
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


kernel void unrollVectorAddAsync( const device CoreParamsForDevice &params [[ buffer(0) ]],
                                  const device float4 *privateBuffer [[ buffer(1) ]],
                                  const device float4 *inA [[ buffer(2) ]],
                                  const device float4 *inB [[ buffer(3) ]],
                                  device float4 *out [[ buffer(4) ]],
                                  uint blockIdx [[ threadgroup_position_in_grid ]],
                                  uint blockDim [[ threads_per_threadgroup ]],
                                  uint threadIdx [[ thread_position_in_threadgroup ]],
                                  uint gid [[ thread_position_in_grid ]]
                                ) {
    // Manual calculation of global Thread ID to show how `gid` is obtained
    // uint tid = (blockIdx * blockDim) + threadIdx;

    for (ushort i = 0; i < params.N_FLOAT4_PER_THREAD; i++)
    {
        // Process N_FLOAT4_PER_THREAD float4 elements per thread, and increment to access the next int4 element
        uint float4Idx = gid * params.N_FLOAT4_PER_THREAD + i;


        // Check bounds for the current int4 elements
        if (float4Idx * 4 < params.N)
        {
            out[float4Idx] = sin(inA[float4Idx] * inB[float4Idx]) + inA[float4Idx];
        }
    }
}

kernel void unrollVectorAsyncPad( const device CoreParamsForDevice &params [[ buffer(0) ]],
                                  device float4 *privateBuffer [[ buffer(1) ]],
                                  const device float4 *inA [[ buffer(2) ]],
                                  const device float4 *inB [[ buffer(3) ]],
                                  device float4 *out [[ buffer(4) ]],
                                  uint gid [[ thread_position_in_grid ]]
                                ) {

    // Process N_FLOAT4_PER_THREAD float4 elements per thread
    // x: float4Idx = gid * params.N_FLOAT4_PER_THREAD;
    // y: iFloat4Idx = gid * params.N_FLOAT4_PER_THREAD + i;
    // z: elementBaseIdx = iFloat4Idx * 4;


    uint4 floatIdxs = {gid * params.N_FLOAT4_PER_THREAD, 0, 0, 0};

    for (uint8_t i = 0; i < params.N_FLOAT4_PER_THREAD; i++)
    {
        // Process N_FLOAT4_PER_THREAD float4 elements per thread, and increment to access the next int4 element
        floatIdxs.y = floatIdxs.x + i;

        // Find first element in current float4
        floatIdxs.z = floatIdxs.y * 4;

        if ( floatIdxs.z + 3 < params.N_WITH_DATA )
        {
            /*
             * Last element of current float4 is within boundaries so whole float4 also is within current boundaries.
             * As this is the most common outcome of the Boolean checks, put it first to minimise number of tests
             */
            out[floatIdxs.y] = sin(inA[floatIdxs.y] * inB[floatIdxs.y]) + inA[floatIdxs.y];
        }
        else if (floatIdxs.z < params.N_WITH_DATA)
        {
            /*
             * This means (at least) the first element of the float4 is within boundaries but, due to the main IF
             * statement being FALSE, we know not all elements of this float4 are within boundaries.
             *
             * Process each float within the float4 individually.
             */
            float4 tmp = 0.0f;
            // Can guarantee [float4Idx][0] is within boundaries, otherwise `elementBaseIdx < params.N` would fail
            tmp[0] = sin(inA[floatIdxs.y][0] * inB[floatIdxs.y][0]) + inA[floatIdxs.y][0];

            uint8_t j = 1;
            while ( floatIdxs.z + j < params.N_WITH_DATA && j < 4 )
            {
                // Loop through elements of current float4. Test each index to see if it is valid
                tmp[j] = sin(inA[floatIdxs.y][j] * inB[floatIdxs.y][j]) + inA[floatIdxs.y][j];
                j++;
                /*
                 * As `tmp` is initialised to zero, we can exit this inner FOR loop once the first invalid
                 * elementIdx is found.
                 */

            }

            // Write the (sometimes partially) computed result to the output buffer
            out[floatIdxs.y] = tmp;
        }
        else
        {
            /*
             * All that should trigger this ELSE is `elementBaseIdx >= params.N`. Putting this last, instead of at the
             * start as a Guard Clause, beyond this introduces a Boolean check for every float4, but almost every float4
             * will pass this check; wasting computation time and resources.
             */
            break;
        }
    }
}