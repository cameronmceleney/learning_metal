
#include "ArrayAdder.h"

void ArrayAdder::addArraysCPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC) {
    Timer cpuTimer;
    cpuTimer.setName("CPU Timer");

    cpuTimer.start(true);
    for (size_t i = 0; i < inA.size(); i++) {
        outC[i] = inA[i] + inB[i];
    }
    cpuTimer.stop();

    cpuTimer.print();
}

void ArrayAdder::addArraysComplexCPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC) {
    Timer cpuTimer;
    cpuTimer.setName("CPU Timer");
    cpuTimer.start(true);

    for (size_t i = 0; i < inA.size(); i++) {
        outC[i] = std::sin(inA[i] * inB[i]) + inA[i];
    }

    cpuTimer.stop();
    cpuTimer.print();
}

void ArrayAdder::addArraysGPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC, bool complexAddition) {
    // Assuming device setup is similar to checkForDevice()
    Timer gpuTimer;
    gpuTimer.setName("GPU Timer (compute)");

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

    auto kernelFunction = library->newFunction(NS::String::string(complexAddition ? "complex_operation" : "add_arrays", NS::UTF8StringEncoding));
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

    bool simpleThreadGroupSize = false;

    if (simpleThreadGroupSize)
    {
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
    }
    else
    {
        // Obtain thread execution width and max total threads per threadgroup
        size_t threadExecutionWidth = computePipelineState->threadExecutionWidth();
        // Query max threads per threadgroup from the device
        size_t maxThreadsPerThreadgroup = computePipelineState->maxTotalThreadsPerThreadgroup();

        // Calculate threads per threadgroup based on the thread execution width
        size_t threadsPerThreadgroupWidth = threadExecutionWidth;
        size_t threadsPerThreadgroupHeight = maxThreadsPerThreadgroup / threadExecutionWidth;
        // For 1D
        MTL::Size threadgroupSize = {threadExecutionWidth, 1, 1};
        // For 2D
        //MTL::Size threadgroupSize = {threadsPerThreadgroupWidth, threadsPerThreadgroupHeight, 1};


        // Calculate the total number of threads in the grid
        // In this case, we are assuming a 1D data structure, so height and depth are set to 1
        MTL::Size gridSize = {inA.size(), 1, 1};

        // Dispatch threads using nonuniform thread groups
        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
    }
    computeCommandEncoder->endEncoding();

    gpuTimer.start(true);
    // Initiate the computation on the GPU
    commandBuffer->commit();
    // Ensure that the CPU checks that the GPU is finished before continuing
    commandBuffer->waitUntilCompleted();
    gpuTimer.stop();
    gpuTimer.print();

    gpuTimer.setName("GPU Timer (copy memory)");
    gpuTimer.start(true);
    // Retrieve the result
    memcpy(outC.data(), bufferC->contents(), inA.size() * sizeof(float));
    gpuTimer.stop();
    gpuTimer.print();

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

void ArrayAdder::addArraysGpuAsyncWithChunking(const std::vector<float>& inA, const std::vector<float>& inB,
                                               std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu) {
    Timer gpuTimer;
    gpuTimer.setName("GPU Timer (compute)");

    const size_t maxChunkSize = 50000000; // Example chunk size, adjust based on GPU capabilities
    size_t vectorSize = inA.size();

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

    auto kernelFunction = library->newFunction(NS::String::string(complexAddition ? "complex_operation" : "add_arrays", NS::UTF8StringEncoding));
    auto computePipelineState = device->newComputePipelineState(kernelFunction, &error);

    if (!kernelFunction || !computePipelineState) {
        std::cerr << "Failed to initialize GPU resources." << std::endl;
        return;
    }

    // Create buffers for input and output. Use A/B/C and D/E/F sets of buffers to enable async processing
    auto bufferA = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferB = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferC = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferD = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferE = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto bufferF = device->newBuffer(maxChunkSize * sizeof(float), MTL::ResourceStorageModeShared);

    // Load the first chunk into bufferA and bufferB
    size_t chunkSize = std::min(vectorSize, maxChunkSize);
    memcpy(bufferA->contents(), inA.data(), chunkSize * sizeof(float));
    memcpy(bufferB->contents(), inB.data(), chunkSize * sizeof(float));

    size_t processedChunks = 0; // Keep track of the number of chunks processed

    gpuTimer.start(true);

    for (size_t start = 0; start < vectorSize; start += maxChunkSize) {
        size_t currentChunkSize = std::min(vectorSize - start, maxChunkSize);

        // Encoding commands
        auto commandBuffer = commandQueue->commandBuffer();
        auto computeCommandEncoder = commandBuffer->computeCommandEncoder();
        computeCommandEncoder->setComputePipelineState(computePipelineState);
        computeCommandEncoder->setBuffer(bufferA, 0, 0);
        computeCommandEncoder->setBuffer(bufferB, 0, 1);
        computeCommandEncoder->setBuffer(bufferC, 0, 2);

        MTL::Size gridSize = {currentChunkSize, 1, 1};
        MTL::Size threadgroupSize = {std::min(static_cast<size_t>(computePipelineState->threadExecutionWidth()), gridSize.width), 1, 1};
        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();

        // Submit current chunk for processing
        commandBuffer->commit();

        // Prepare the next chunk asynchronously while the GPU works
        if (start + maxChunkSize < vectorSize) {
            size_t nextChunkSize = std::min(vectorSize - (start + maxChunkSize), maxChunkSize);
            memcpy(bufferD->contents(), inA.data() + start + maxChunkSize, nextChunkSize * sizeof(float));
            memcpy(bufferE->contents(), inB.data() + start + maxChunkSize, nextChunkSize * sizeof(float));
        }

        if (start != 0 && onlyOutputToCpu) {
            // Copy results back to the CPU from the previous iteration's output buffer while the GPU works
            memcpy(outC.data() + start - maxChunkSize, bufferF->contents(), currentChunkSize * sizeof(float));
        }

        commandBuffer->waitUntilCompleted();

        if (start + maxChunkSize < vectorSize) {
            // Swap buffers for the next iteration if there exists a next iteration
            std::swap(bufferA, bufferD);
            std::swap(bufferB, bufferE);
            std::swap(bufferC, bufferF);
        } else {
            // This is the final iteration so we do clean-up of the data
            if (onlyOutputToCpu) {
                if ( vectorSize % maxChunkSize > 0 ) { // Check if there's a remainder chunk
                    size_t remainderSize = vectorSize % maxChunkSize;
                    memcpy(outC.data() + vectorSize - remainderSize, bufferC->contents(), remainderSize * sizeof(float));
                } else { // Full chunk
                    memcpy(outC.data() + vectorSize - maxChunkSize, bufferC->contents(), maxChunkSize * sizeof(float));
                }
            }
        }

        // Release resources for this iteration
        computeCommandEncoder->release();
        commandBuffer->release();

        processedChunks++;
    }
    gpuTimer.stop();
    gpuTimer.print();

    // Clean up
    bufferA->release();
    bufferB->release();
    bufferC->release();
    bufferD->release();
    bufferE->release();
    bufferF->release();
    computePipelineState->release();
    kernelFunction->release();
    library->release();
    commandQueue->release();
    device->release();
}
