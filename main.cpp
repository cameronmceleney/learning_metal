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
#include <vector>

int main() {
    DeviceChecks::checkForDevice();
    //GraphicalExamples::generateSquare();

    // Written explicitly so I can check the results by hand
    std::vector<float> vec1{0.0, 5.0, 4.0, 9.0, 11.0};
    std::vector<float> vec2{3.0, 5.0, 14.0, 2.0, 99.0};
    std::vector<float> resultVec(vec1.size(), 0.0);
    ArrayAdder::addArrays(vec1, vec2, resultVec);
    for (auto val: resultVec){
        std::cout << val << ", ";
    }
    //ComputeFunctionExamples computeFunctionExamples;
    //computeFunctionExamples.sumSimpleVectors();
    return 0;
}
