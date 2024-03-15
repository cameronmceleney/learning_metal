
#include "ArrayAdder.h"


void ArrayAdder::addArrays(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC) {
    // Assuming device setup is similar to checkForDevice()

    auto device = MTL::CreateSystemDefaultDevice();
    auto commandQueue = device->newCommandQueue();
    NS::Error* error = nullptr;

    // Load the .metal shader file
    // Modify this line to use METAL_SHADER_METALLIB_PATH if one wants it to be hardcoded
    auto libraryPath = NS::String::string(METAL_SHADER_METALLIB_PATH, NS::UTF8StringEncoding);
    auto library = device->newLibrary(libraryPath, &error);
    if (!library) {
        std::cerr << "Failed to load the library from path: " << METAL_SHADER_METALLIB_PATH << std::endl;
        return;
    }

    auto kernelFunction = library->newFunction(NS::String::string("add_arrays", NS::UTF8StringEncoding));
    auto computePipelineState = device->newComputePipelineState(kernelFunction, &error);

    // Create buffers for input and output
    auto bufferA = device->newBuffer(inA.data(), inA.size() * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferB = device->newBuffer(inB.data(), inB.size() * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferC = device->newBuffer(inA.size() * sizeof(float), MTL::ResourceStorageModeShared);

    // Encoding commands
    auto commandBuffer = commandQueue->commandBuffer();
    auto computeCommandEncoder = commandBuffer->computeCommandEncoder();
    computeCommandEncoder->setComputePipelineState(computePipelineState);
    computeCommandEncoder->setBuffer(bufferA, 0, 0);
    computeCommandEncoder->setBuffer(bufferB, 0, 1);
    computeCommandEncoder->setBuffer(bufferC, 0, 2);

    MTL::Size gridSize = {inA.size(), 1, 1};
    MTL::Size threadgroupSize = {
            static_cast<NS::UInteger>(std::min(
                    static_cast<size_t>(computePipelineState->maxTotalThreadsPerThreadgroup()),
                    inA.size()
            )),
            1,
            1
    };
    computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeCommandEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Retrieve the result
    memcpy(outC.data(), bufferC->contents(), inA.size() * sizeof(float));

    // Clean up
    bufferA->release();
    bufferB->release();
    bufferC->release();
    computeCommandEncoder->release();
    commandBuffer->release();
    computePipelineState->release();
    kernelFunction->release();
    library->release();
    commandQueue->release();
    device->release();
}
