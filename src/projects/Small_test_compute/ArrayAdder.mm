
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

void ArrayAdder::addArraysGpuWithChunking( const std::vector<float>& inA, const std::vector<float>& inB,
                                           std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu) {
    /*
     *
Given the specifications of your Apple M3 Pro system and the sizes of your vectors (10 billion elements for a large example and 12,000 elements for a small example), let's break down how to structure your workgroups for optimal performance. The approach to chunking and workgroup organization is critical in leveraging the GPU efficiently.

Understanding Your System's Constraints

     Max Threads Per Threadgroup (1024):
                                    This is the upper limit of threads you can have in a single threadgroup.

     Thread Execution Width (32):
                                    This indicates the optimal number of threads that can be executed in parallel for a single threadgroup.

     Max Buffer Length (9,663,676,416 bytes):
                                    Maximum size of a buffer that can be allocated,
                                        important for understanding how much data you can work with in a single operation.

     Unified Memory:
                                    Simplifies memory management but still requires efficient use of memory to avoid performance bottlenecks.
Strategy for Large Vectors (10 Billion Elements)

     Chunking Data:

     Given the vector's size far exceeds the Max Buffer Length, you'll need to chunk the data.

     Determine an appropriate chunk size based on the Max Buffer Length and the size of a float (4 bytes).
     However, processing 10 billion elements in a single pass isn't feasible due to memory constraints,
        so decide on a chunk size that balances memory use and performance. For instance, processing 1 million elements per chunk would require 4MB per buffer (since each float is 4 bytes), which is well within your buffer limit.

     Configuring Workgroups:
            With a Thread Execution Width of 32, configure each workgroup to handle a segment of the chunk efficiently.
            If processing 1 million elements per chunk, divide this work among several threadgroups.

            You might not use the maximum of 1024 threads per threadgroup if it doesn't divide evenly into your chunk size.
            Aim for full utilization of the 32-thread execution width. For example, you could have workgroups of 32 threads each.

Example Calculation for Small and Large Vectors:

     Large Vector (10 Billion Elements):
                                    If you're processing in chunks of 1 million elements, that's 10,000 chunks in total.
                                    Each thread could process multiple elements to match the execution width and threads per threadgroup.

     Small Vector (12,000 Elements):
                                    This can be processed in a single chunk.
                                    You could use 375 threads (12,000 elements / 32 = 375), fitting within the 1024 thread per threadgroup limit.
                                    If you opt for an exact match to the execution width, consider structuring your kernel to handle this efficiently,
                                        possibly by making each thread process multiple elements if necessary.
     */
    Timer gpuTimer;
    gpuTimer.setName("GPU Timer (compute)");

    const size_t maxChunkSize = static_cast<int>(1e7); // TODO. Chunk size below 1E8 required for GPU to beat CPU. Why?
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

    // Error handling example for buffer creation
    if (!bufferA || !bufferB || !bufferC || !bufferD || !bufferE || !bufferF) {
        std::cerr << "Failed to create one or more buffers." << std::endl;
        std::exit(1);
    }

    // Load the first chunk into bufferA and bufferB
    size_t chunkSize = std::min(vectorSize, maxChunkSize);
    memcpy(bufferA->contents(), inA.data(), chunkSize * sizeof(float));
    memcpy(bufferB->contents(), inB.data(), chunkSize * sizeof(float));

    size_t processedChunks = 0; // Keep track of the number of chunks processed

    gpuTimer.start(true);

    // Use a semaphore for synchronization between CPU and GPU; allows a single one to be made
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(1);

    for (size_t start = 0; start < vectorSize; start += maxChunkSize) {
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

        size_t currentChunkSize = std::min(vectorSize - start, maxChunkSize);

        // Preparing command buffer and command encoder
        auto commandBuffer = commandQueue->commandBuffer();

        auto computeCommandEncoder = commandBuffer->computeCommandEncoder();

        // Ensure commandBuffer and computeCommandEncoder creation succeeded
        if (!commandBuffer || !computeCommandEncoder) {
            std::cerr << "Failed to create command buffer or encoder." << std::endl;
            std::exit(1);
        }

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Encoding commands
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

    // Ensure all command buffers are completed before exiting the function
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

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

void ArrayAdder::initializeResources(const std::string& kernelFunctionName) {
    deviceAsync = MTL::CreateSystemDefaultDevice();

    semaphoreAsync = dispatch_semaphore_create(6); // Allows up to N buffers to be processed asynchronously.

    /*
     * Selecting chunk size (made easy)
     *      1. Get the vector's length
     *      2. Divide the vector by 32 (this is threadExecutionWith)
     *          a. If this result is not an int, decrement the variable holding the vector-length and continue until an
     *          int result is found
 *          3. Take the result and divide by 1024 (this is maxThreadsPerGroup)
     */
    if (lengthVector <= 0) { std::exit(1); }

    const int threadExecutionWidth = 32;
    const int maxThreadsPerGroup = static_cast<int>(deviceAsync->maxThreadsPerThreadgroup().height);
    // Find the largest possible value X where X / 32 is an integer, starting with X = lengthOfVector.
    size_t X = lengthVector - (lengthVector % threadExecutionWidth);

    // Divide the used X/32 result by 1024, and repeat the process
    // (if X/32/1024 is not an integer then move onto 1023 etc)
    int threadsPerGroup = maxThreadsPerGroup;
    while (X % (threadExecutionWidth * threadsPerGroup) != 0 && threadsPerGroup > 0) {
        threadsPerGroup -= 1;
    }

    // Resulting values to use
    maxChunkSizeAsync = threadExecutionWidth * threadsPerGroup;
    size_t numChunks = X / maxChunkSizeAsync;

    // std::cout << "threadsPerGroup: " << threadsPerGroup << std::endl;
    // std::cout << "Chunk Size: " << maxChunkSizeAsync << std::endl;
    // std::cout << "Number of Chunks: " << numChunks << std::endl;

    if (maxChunkSizeAsync > static_cast<int>(deviceAsync->maxArgumentBufferSamplerCount())) {
        std::cout << "Calculated number of chunks is larger than MaxArgumentBufferSamplerCount" << std::endl;
        exit(1);
    }

    if (static_cast<int>(maxChunkSizeAsync * sizeof(float)) > static_cast<int>(deviceAsync->maxBufferLength())) {
        // Assume we are using floats for now
        std::cout << "Each chunk has a ByteSize that is larger than maxBufferLength" << std::endl;
        exit(1);
    }

    if ( sizeof(float) * threadsPerGroup > static_cast<int>(deviceAsync->maxThreadgroupMemoryLength())) {
        std::cout << "Total shared memory of Thread Groups is larger than maxThreadGroupsMemoryLane" << std::endl;
        exit(1);
    }

    commandQueueAsync = deviceAsync->newCommandQueue();

    auto libraryPath = NS::String::string(METAL_SHADER_METALLIB_PATH, NS::UTF8StringEncoding);
    auto library = deviceAsync->newLibrary(libraryPath, nullptr);
    auto kernelFunction = library->newFunction(NS::String::string(kernelFunctionName.c_str(), NS::UTF8StringEncoding));
    computePipelineStateAsync = deviceAsync->newComputePipelineState(kernelFunction, &errorAsync);

    // Initialize a pool of buffers for asynchronous processing.
    for (int i = 0; i < numSemaphores; ++i) {
        // 10 buffers for a pool, allowing 5 in use and 5 being prepared.
        bufferPoolAsync.push_back(deviceAsync->newBuffer(maxChunkSizeAsync * sizeof(float), MTL::ResourceStorageModeShared));
    }
}

void ArrayAdder::releaseResources() {
    for (auto* buffer : bufferPoolAsync) buffer->release();
    computePipelineStateAsync->release();
    commandQueueAsync->release();
    deviceAsync->release();
}

MTL::Buffer* ArrayAdder::getNextBuffer() {
    auto* buffer = bufferPoolAsync[bufferIndexAsync % bufferPoolAsync.size()];
    bufferIndexAsync++;
    return buffer;
}

void ArrayAdder::processChunks(const std::vector<float>& inA, const std::vector<float>& inB,
                               std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu) {
    size_t vectorSize = inA.size();

    // Assuming initialization has already been done.
    size_t currentChunkSize = 0;
    for (size_t start = 0; start < vectorSize; start += maxChunkSizeAsync) {
        currentChunkSize = std::min(vectorSize - start, maxChunkSizeAsync);
        auto* inputBufferA = getNextBuffer();
        auto* inputBufferB = getNextBuffer();
        auto* outputBuffer = getNextBuffer();

        memcpy(inputBufferA->contents(), inA.data() + start, currentChunkSize * sizeof(float));
        memcpy(inputBufferB->contents(), inB.data() + start, currentChunkSize * sizeof(float));

        auto commandBuffer = commandQueueAsync->commandBuffer();
        auto computeCommandEncoder = commandBuffer->computeCommandEncoder();
        computeCommandEncoder->setComputePipelineState(computePipelineStateAsync);
        computeCommandEncoder->setBuffer(inputBufferA, 0, 0);
        computeCommandEncoder->setBuffer(inputBufferB, 0, 1);
        computeCommandEncoder->setBuffer(outputBuffer, 0, 2);

        MTL::Size gridSize = {currentChunkSize, 1, 1};
        MTL::Size threadgroupSize = {std::min(static_cast<size_t>(computePipelineStateAsync->threadExecutionWidth()), gridSize.width), 1, 1};
        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();

        commandBuffer->addCompletedHandler(^(MTL::CommandBuffer*){
            // Upon completion of the GPU for this iteration
            if (onlyOutputToCpu) {
                // copy data from the output buffer to the CPU.
                memcpy(outC.data() + start, outputBuffer->contents(), currentChunkSize * sizeof(float));
            }
            // Signal that this buffer is now free for reuse.
            dispatch_semaphore_signal(semaphoreAsync);
        });

        commandBuffer->commit();
        // After committing, the CPU moves on to prepare the next chunk without waiting for the GPU,
        // except for semaphore limiting overall parallel command buffers.
    }
}

void ArrayAdder::addArraysGpuChunkingDynamicBufferAsync(const std::vector<float>& inA, const std::vector<float>& inB,
                                                        std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu) {
    Timer gpuTimer;
    gpuTimer.setName("GPU Timer (compute)");

    initializeResources(complexAddition ? "complex_operation" : "add_arrays");

    gpuTimer.start(true);
    processChunks(inA, inB, outC, complexAddition, onlyOutputToCpu);
    gpuTimer.stop();
    gpuTimer.print();

    // Wait for all GPU tasks to complete before releasing resources.
    for (int i = 0; i < numSemaphores; ++i) { // Assuming up to 5 buffers can be processed in parallel.
        dispatch_semaphore_wait(semaphoreAsync, DISPATCH_TIME_FOREVER);
    }

    releaseResources();
}