//
// Created by Cameron Aidan McEleney on 15/03/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(const device float* inA [[ buffer(0) ]],
                       const device float* inB [[ buffer(1) ]],
                       device float* outC [[ buffer(2) ]],
                       uint id [[ thread_position_in_grid ]]) {
    outC[id] = inA[id] + inB[id];
}