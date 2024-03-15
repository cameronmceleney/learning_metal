//
// Created by Cameron Aidan McEleney on 15/03/2024.
//

#ifndef HELLO_METAL_ARRAYADDER_H
#define HELLO_METAL_ARRAYADDER_H

#include <Metal/Metal.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

class ArrayAdder {
public:
    // Adds elements of two input arrays using GPU and stores the result in the output array.
    // Parameters inA and inB are the input arrays, and outC is the output array where the result is stored.
    static void addArrays(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC);
};

#endif //HELLO_METAL_ARRAYADDER_H
