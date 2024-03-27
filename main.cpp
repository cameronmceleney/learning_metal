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
//#include "src/projects/Small_test_compute/ArrayAdder.h"
#include "src/projects/graphical_implementation_example/graphical_example_m.h"
//#include "src/projects/compute_function_examples/compute_function_examples.h"

#include "src/projects/Small_test_compute/CoffeeWithArch_examples.h"

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

void compareGpuToCpuResults(const std::vector<float>& resultGPU, const std::vector<float>& resultCPU) {
    if (resultGPU.size() != resultCPU.size()) {
        std::cerr << "Error: Result vectors are of different sizes." << std::endl;
        return;
    }

    double maxAbsoluteDifference = 0.0;
    double sumAbsoluteDifferences = 0.0;

    for (size_t i = 0; i < resultCPU.size(); ++i) {
        double difference = std::abs(resultGPU[i] - resultCPU[i]);
        maxAbsoluteDifference = std::max(maxAbsoluteDifference, difference);
        sumAbsoluteDifferences += difference;
    }

    double meanAbsoluteDifference = sumAbsoluteDifferences / resultCPU.size();

    std::cout << "Maximum Absolute Difference: " << maxAbsoluteDifference << std::endl;
    std::cout << "Mean Absolute Difference: " << meanAbsoluteDifference << std::endl;
}

void compareSumOfResults(const std::vector<float>& resultGPU, const std::vector<float>& resultCPU) {
    if (resultGPU.size() != resultCPU.size()) {
        std::cerr << "Error: Result vectors are of different sizes." << std::endl;
        return;
    }

    // Compute the sum of each result vector
    float sumGPU = std::accumulate(resultGPU.begin(), resultGPU.end(), 0.0f);
    float sumCPU = std::accumulate(resultCPU.begin(), resultCPU.end(), 0.0f);

    // Compute the absolute difference of the sums
    float difference = std::abs(sumGPU - sumCPU);

    std::cout << "Sum of GPU Results: " << sumGPU << std::endl;
    std::cout << "Sum of CPU Results: " << sumCPU << std::endl;
    std::cout << "Absolute Difference of Sums: " << difference << std::endl;

    // Optionally, compute and display the relative difference if sumCPU is not zero
    if (sumCPU != 0) {
        float relativeDifference = difference / std::abs(sumCPU);
        std::cout << "Relative Difference of Sums: " << relativeDifference << std::endl;
    }
}

int main() {
    // DeviceChecks::checkForDevice();
    // DeviceChecks::printDeviceInfo();
    // GraphicalExamples::generateSquare();

    int N = 1 << 27;
    std::string functionName = "simpleVectorSineAdd";
    bool useDebug = false;

    CoffeeExample coffeeExample{};

    coffeeExample.vectorAddition(N, functionName, useDebug);
    coffeeExample.vectorAdditionPrivateResources(N, functionName, useDebug);
    coffeeExample.vectorAdditionManagedResources(N, functionName, useDebug);
    coffeeExample.vectorAdditionFullyManagedResources(N, functionName, useDebug);
    coffeeExample.vectorAdditionAsyncBuffers(N, "unrollVectorAdd", useDebug);



    // Written explicitly so I can check the results by hand. Keep below 1 billion elements without chunking!
    /*
    const size_t vectorSize = static_cast<int>(1e8);
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
    arrayAdder.addArraysGpuChunkingDynamicBufferAsync(vec1, vec2, resultGPU, true, true);

    compareGpuToCpuResults(resultGPU, resultCPU);
    compareSumOfResults(resultGPU, resultCPU);

    //ComputeFunctionExamples computeFunctionExamples;
    //computeFunctionExamples.sumSimpleVectors();
     */
    return 0;
}
