//
// main.cpp
// hello_metal.cpp
//
// Created by Cameron McEleney on 08 Mar 24
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION

// Local
#include "src/projects/checks_examples/check_for_metal_device.h"
#include "src/projects/Small_test_compute/ArrayAdder.h"
#include "src/projects/graphical_implementation_example/graphical_example_m.h"
#include "src/projects/compute_function_examples/compute_function_examples.h"

// Include here for ease while building program
#include <random>
#include <vector>

std::vector<float> getRandomVector(size_t size) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& el : vec) {
        el = static_cast<float>(dis(gen));
    }

    return vec;
}

int main() {
    // DeviceChecks::checkForDevice();
    // DeviceChecks::printDeviceInfo();
    // GraphicalExamples::generateSquare();

    // Written explicitly so I can check the results by hand. Keep below 1 billion elements without chunking!
    const size_t vectorSize = static_cast<int>(583715353);
    std::vector<float> vec1 = getRandomVector(vectorSize);
    std::vector<float> vec2 = getRandomVector(vectorSize);
    std::cout << "len vec1: " << vec1.size() << " ! len vec2: " << vec2.size() << std::endl;
    std::vector<float> resultGPU(vec1.size());
    std::vector<float> resultCPU(vec1.size());

    ArrayAdder::addArraysComplexCPU(vec1, vec2, resultCPU);
    // ArrayAdder::addArraysGPU(vec1, vec2, resultGPU, true);
    // ArrayAdder::addArraysGpuWithChunking(vec1, vec2, resultGPU, true, false);
    ArrayAdder arrayAdder;
    arrayAdder.lengthVector = vectorSize;
    arrayAdder.addArraysGpuChunkingDynamicBufferAsync(vec1, vec2, resultGPU, true, false);

    //ComputeFunctionExamples computeFunctionExamples;
    //computeFunctionExamples.sumSimpleVectors();
    return 0;
}
