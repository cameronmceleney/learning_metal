//
// Created by Cameron Aidan McEleney on 27/03/2024.
//

#include "CoffeeWithArch_examples.h"

std::vector<float> CommonTools::getRandomVector(size_t size) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);

    for (auto& el : vec) {
        el = static_cast<float>(dis(gen));
    }

    return vec;
}

void CommonTools::padVectorForCacheLine(std::vector<float>& vec, CoreParamsForHost &coreParams) {
    // ONLY WORKS FOR FLOAT FOR NOW DUE TO FUNCTION PROTOTYPE, AND PUSH_BACK(0.0F)

    size_t bytesCacheLine = 16;  // value in bytes
    // e.g. for a float bytesPerElement=4 so elementsPerCacheLine = 4
    size_t elementsPerCacheLine = bytesCacheLine / coreParams.SIZE_DTYPE;

    // Check number of elements in vector is a multiple of the cache line width
    size_t remainder = coreParams.NUM_ELEMENTS_WITH_DATA % elementsPerCacheLine;
    if (remainder != 0) {
        coreParams.NUM_ELEMENTS_PADDING = elementsPerCacheLine - remainder;
        for (size_t i = 0; i < coreParams.NUM_ELEMENTS_PADDING; ++i) {
            vec.push_back(0.0f); // Pad with zeros for now for a FLOAT
        }
    }
}

void CommonTools::verifyResultsAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                                  const std::vector<float> &resultVec, const bool &printAnswer) {

    for (size_t i = 0; i < inVecA.size(); ++i) {
        if (resultVec[i] != inVecA[i] + inVecB[i]) {
            std::cerr << "Verification failed at index " << i << ": " << resultVec[i] << " != " << inVecA[i] << " + "
            << inVecB[i] << std::endl;
            throw std::runtime_error("Result verification failed.");
        }

        if (printAnswer)
        {
            std::cout << resultVec[i] << " = " << inVecA[i] << " + " << inVecB[i] << std::endl;
        }
    }
}

void CommonTools::verifyResultsSineAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                                  const std::vector<float> &resultVec, const bool &printAnswer) {

    constexpr float epsilon = 0.000001;
    for (size_t i = 0; i < inVecA.size(); ++i) {
        float resultCpu = sin(inVecA[i] * inVecB[i]) + inVecA[i];
        float diffCpuToGpu = resultCpu - resultVec[i];
        if (fabs(diffCpuToGpu) >= epsilon) {
            // Beyond acceptable range
            std::cerr << "Verification failed at index " << i << " with diff " << diffCpuToGpu << " | " << resultVec[i] << " != sin(" << inVecA[i] << " * "
            << inVecB[i] << ") + " << inVecA[i] << std::endl;
            throw std::runtime_error("Result verification failed.");
        }

        if (printAnswer)
        {
            std::cout << resultVec[i] << " = " << inVecA[i] << " + " << inVecB[i] << std::endl;
        }
    }
}

void CommonTools::verifyResultsSineAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                                  const std::vector<float> &resultVec, CoreParamsForDevice &coreParams, const bool &printAnswer) {

    constexpr float epsilon = 0.000001;
    for (size_t i = 0; i < coreParams.NUM_ELEMENTS; ++i) {
        float resultCpu = sin(inVecA[i] * inVecB[i]) + inVecA[i];
        float diffCpuToGpu = resultCpu - resultVec[i];
        if (fabs(diffCpuToGpu) >= epsilon) {
            // Beyond acceptable range
            std::cerr << "Verification failed at index " << i << " with diff " << diffCpuToGpu << " | " << resultVec[i] << " != sin(" << inVecA[i] << " * "
            << inVecB[i] << ") + " << inVecA[i] << std::endl;
            throw std::runtime_error("Result verification failed.");
        }

        if (printAnswer)
        {
            std::cout << resultVec[i] << " = " << inVecA[i] << " + " << inVecB[i] << std::endl;
        }
    }
}

void CoffeeExample::initialiseResources(const std::string& kernelFunctionName) {
    device = MTL::CreateSystemDefaultDevice();

    commandQueuePrimary = device->newCommandQueue();

    libraryPath = NS::String::string(METAL_SHADER_METALLIB_PATH, NS::UTF8StringEncoding);
    library = device->newLibrary(libraryPath, &error);
    if (!library)
    {
        std::cerr << "Failed to load the library from path: " << METAL_SHADER_METALLIB_PATH << std::endl;
        return;
    }

    kernelFunction = library->newFunction(NS::String::string(kernelFunctionName.c_str(), NS::UTF8StringEncoding));

    computePrimaryPipelineState = device->newComputePipelineState(kernelFunction, &error);
    if (!computePrimaryPipelineState)
    {
        std::cerr << "Failed to set compute pipeline state." << std::endl;
        std::exit(1);
    }
}

void CoffeeExample::releaseResources() {
    computeCommandEncoder->release();
    computeCommandBuffer->release();
    computePrimaryPipelineState->release();
    kernelFunction->release();
    library->release();
    libraryPath->release();
    commandQueuePrimary->release();
    device->release();
}

void CoffeeExample::processVectorAddition( const std::vector<float> &inA, const std::vector<float> &inB, std::vector<float> &outC ) {
    std::cout << "Made it to processVectorAddition" << std::endl;
}


void CoffeeExample::vectorAddition( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {
    CoreParamsForHost hostCoreParams(numElements);
    hostCoreParams.calculateDataPerThread(1, 1);
    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Shared Resources)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Shared Resources)");

    // Original code
    //auto vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    //auto vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS);

    // Initialise random numbers in each array
    if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS_WITH_DATA)
        vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS_WITH_DATA);
    if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS_WITH_DATA)
        vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS_WITH_DATA);

    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS_WITH_DATA);

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    auto bufferVec1 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferVec2 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferVec3 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, MTL::ResourceStorageModeShared);

    // Error handling example for buffer creation
    if ( !bufferVec1 || !bufferVec2 || !bufferVec3 || !bufferCoreParameters )
    {
        std::cerr << "Failed to create one or more buffers!" << std::endl;
        std::exit(1);
    }

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();
    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferVec1, 0, 0);
    computeCommandEncoder->setBuffer(bufferVec2, 0, 1);
    computeCommandEncoder->setBuffer(bufferVec3, 0, 2);
    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 3);

    // Copy data from host to device (CPU -> GPU)
    memcpy(bufferVec1->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferVec2->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Threads per block (1<<10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = (hostCoreParams.NUM_ELEMENTS + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    kernelTimer.start(true);
    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);
    // Send instruction that our pipeline state encoding is completed
    computeCommandEncoder->endEncoding();

    // Asynchronously launch the kernel on the device (GPU)
    //commandBuffer->GPUStartTime()
    std::cout << std::endl;
    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferVec3->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, false);

    // Free memory on the device (GPU)
    bufferVec1->release();
    bufferVec2->release();
    bufferVec3->release();

    // Release all other resources
    releaseResources();
}

void CoffeeExample::vectorAdditionPrivateResources( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {

    CoreParamsForHost hostCoreParams(numElements);
    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Private Resources)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Private Resources)");


    // Initialise random numbers in each array
    if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS)
        vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS)
        vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS);

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    auto bufferVec1Shared = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferVec2Shared = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferVec3 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, MTL::ResourceStorageModeShared);

    // Error handling example for buffer creation
    if ( !bufferVec1Shared || !bufferVec2Shared || !bufferVec3 || !bufferCoreParameters )
    {
        std::cerr << "Failed to create one or more shared buffers!" << std::endl;
        std::exit(1);
    }

    // Copy data to shared buffers with data from the host (CPU)
    memcpy(bufferVec1Shared->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferVec2Shared->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Allocate private memory for the device
    auto bufferVec1Private = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModePrivate);
    auto bufferVec2Private = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModePrivate);

    // Error handling example for buffer creation
    if ( !bufferVec1Private || !bufferVec2Private)
    {
        std::cerr << "Failed to create one or more private buffers!" << std::endl;
        std::exit(1);
    }

    // Step 4: Copy Data from Shared to Private Buffers using a Blit Encoder
    MTL::CommandBuffer* copyCommandBuffer = commandQueuePrimary->commandBuffer();
    MTL::BlitCommandEncoder* blitEncoder = copyCommandBuffer->blitCommandEncoder();

    blitEncoder->copyFromBuffer(bufferVec1Shared, 0, bufferVec1Private, 0, hostCoreParams.BYTES_FOR_ELEMENTS);
    blitEncoder->copyFromBuffer(bufferVec2Shared, 0, bufferVec2Private, 0, hostCoreParams.BYTES_FOR_ELEMENTS);

    blitEncoder->endEncoding();
    copyCommandBuffer->commit();
    copyCommandBuffer->waitUntilCompleted();

    // Release the blit encoder and copy command buffer
    blitEncoder->release();
    copyCommandBuffer->release();

    // Since the data is now copied to the private buffers, the shared buffers are no longer needed and can be released
    bufferVec1Shared->release();
    bufferVec2Shared->release();

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();

    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferVec1Private, 0, 0);
    computeCommandEncoder->setBuffer(bufferVec2Private, 0, 1);
    computeCommandEncoder->setBuffer(bufferVec3, 0, 2);
    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 3);

    // Threads per block (1<<10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = (hostCoreParams.NUM_ELEMENTS + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);

    // Finalise encoding
    computeCommandEncoder->endEncoding();

    kernelTimer.start(true);

    // Asynchronously submit the command buffer containing the kernel to the device (GPU)
    //commandBuffer->GPUStartTime()
    std::cout << std::endl;
    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferVec3->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, false);

    // Free memory on the device (GPU)
    bufferVec1Private->release();
    bufferVec2Private->release();
    bufferVec3->release();
    bufferCoreParameters->release();

    // Release all other resources
    releaseResources();
}

void CoffeeExample::vectorAdditionManagedResources( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {

    CoreParamsForHost hostCoreParams(numElements);
    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Managed Resources)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Managed Resources)");


    // Initialise random numbers in each array
    if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS)
        vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS)
        vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS);

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    auto bufferVec1Managed = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeManaged);
    auto bufferVec2Managed = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeManaged);
    auto bufferVec3 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeShared);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, MTL::ResourceStorageModeShared);

    // Error handling example for buffer creation
    if ( !bufferVec1Managed || !bufferVec2Managed || !bufferVec3 || !bufferCoreParameters )
    {
        std::cerr << "Failed to create one or more shared buffers!" << std::endl;
        std::exit(1);
    }

    // Copy data directly into managed buffers
    memcpy(bufferVec1Managed->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferVec2Managed->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Synchronize managed buffers from CPU to GPU
    bufferVec1Managed->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferVec2Managed->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();

    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferVec1Managed, 0, 0);
    computeCommandEncoder->setBuffer(bufferVec2Managed, 0, 1);
    computeCommandEncoder->setBuffer(bufferVec3, 0, 2);
    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 3);


    // Threads per block (1<<10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = (hostCoreParams.NUM_ELEMENTS + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);

    // Finalise encoding
    computeCommandEncoder->endEncoding();

    kernelTimer.start(true);

    // Asynchronously submit the command buffer containing the kernel to the device (GPU)
    //commandBuffer->GPUStartTime()
    std::cout << std::endl;
    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferVec3->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, false);

    // Free memory on the device (GPU)
    bufferVec1Managed->release();
    bufferVec2Managed->release();
    bufferVec3->release();
    bufferCoreParameters->release();

    // Release all other resources
    releaseResources();
}

void CoffeeExample::vectorAdditionFullyManagedResources( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {

    CoreParamsForHost hostCoreParams(numElements);
    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Fully Managed Resources)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Fully Managed Resources)");

    // Initialise random numbers in each array
    if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS)
        vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS)
        vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS);

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    auto bufferVec1 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeManaged);
    auto bufferVec2 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeManaged);
    auto bufferVec3 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, MTL::ResourceStorageModeManaged);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, MTL::ResourceStorageModeManaged);

    // Error handling example for buffer creation
    if ( !bufferVec1 || !bufferVec2 || !bufferVec3 || !bufferCoreParameters )
    {
        std::cerr << "Failed to create one or more shared buffers!" << std::endl;
        std::exit(1);
    }

    // Copy data directly into managed buffers
    memcpy(bufferVec1->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferVec2->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Synchronize managed buffers from CPU to GPU
    bufferVec1->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferVec2->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferCoreParameters->didModifyRange(NS::Range::Make(0, deviceCoreParams.bytes));

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();

    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferVec1, 0, 0);
    computeCommandEncoder->setBuffer(bufferVec2, 0, 1);
    computeCommandEncoder->setBuffer(bufferVec3, 0, 2);
    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 3);

    // Threads per block (1<<10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = (hostCoreParams.NUM_ELEMENTS + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);

    // Finalise encoding
    computeCommandEncoder->endEncoding();

    kernelTimer.start(true);

    // Asynchronously submit the command buffer containing the kernel to the device (GPU)
    //commandBuffer->GPUStartTime()
    std::cout << std::endl;
    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Inform host of changes made to the output buffer by the device
    bufferVec3->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferVec3->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, false);

    // Free memory on the device (GPU)
    bufferVec1->release();
    bufferVec2->release();
    bufferVec3->release();
    bufferCoreParameters->release();

    // Release all other resources
    releaseResources();
}

void CoffeeExampleAsync::initialiseResources(const std::string& kernelFunctionName) {
    device = MTL::CreateSystemDefaultDevice();

    commandQueuePrimary = device->newCommandQueue();

    libraryPath = NS::String::string(METAL_SHADER_METALLIB_PATH, NS::UTF8StringEncoding);
    library = device->newLibrary(libraryPath, &error);
    if (!library)
    {
        std::cerr << "Failed to load the library from path: " << METAL_SHADER_METALLIB_PATH << std::endl;
        return;
    }

    kernelFunction = library->newFunction(NS::String::string(kernelFunctionName.c_str(), NS::UTF8StringEncoding));

    computePrimaryPipelineState = device->newComputePipelineState(kernelFunction, &error);
    if (!computePrimaryPipelineState)
    {
        std::cerr << "Failed to set compute pipeline state." << std::endl;
        std::exit(1);
    }

    initialiseMemoryAllocations();
}

void CoffeeExampleAsync::initialiseMemoryAllocations() {
    // Allocate a private heap for the GPU to use as memory for its ResourcePrivate buffers
    MTL::HeapDescriptor* heapDescriptor = MTL::HeapDescriptor::alloc()->init();
    heapDescriptor->setSize(1024 * 1024); // 1 MB for now
    heapDescriptor->setStorageMode(MTL::StorageModePrivate);
    heapDescriptor->setCpuCacheMode(MTL::CPUCacheModeDefaultCache); // MTL::CPUCacheModeWriteCombined when CPU (GPU) write (read)-only

    privateHeap = device->newHeap(heapDescriptor);

    if (!privateHeap)
    {
        std::cerr << "Failed to create private heap for the GPU." << std::endl;
        std::exit(1);
    }

    // Allocate a private buffer from the private heap
    MTL::ResourceOptions options = MTL::ResourceStorageModePrivate | MTL::ResourceCPUCacheModeDefaultCache;
    size_t bufferSize = 512; // TODO. Make the core-param struct a class attribute
    bufferSize *= sizeof(float);

    // Declare and initialise the allocated private buffer within the private heap
    privateBuffer = privateHeap->newBuffer(bufferSize, options);

    if (!privateBuffer)
    {
        std::cerr << "Failed to create private buffer for the GPU." << std::endl;
        std::exit(1);
    }

    // Release resources that are no longer needed by the wider program
    heapDescriptor->release();
}

void CoffeeExampleAsync::releaseResources() {
    computeCommandEncoder->release();
    computeCommandBuffer->release();
    privateHeap->release();
    computePrimaryPipelineState->release();
    kernelFunction->release();
    library->release();
    libraryPath->release();
    commandQueuePrimary->release();
    device->release();
}

void CoffeeExampleAsync::vectorAdditionHeaps( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {

    hostCoreParams.NUM_ELEMENTS_WITH_DATA = numElements;
    // Number of elements for each thread to compute
    hostCoreParams.calculateDataPerThread(2, 4);
    hostCoreParams.SIZE_DTYPE = sizeof(float);
    hostCoreParams.calculateSize();

    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Async Heaps)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Async Heaps)");

    // Initialise random numbers in each array
    if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS)
        vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS)
        vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS);
    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS);

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    CoffeeExampleAsync::initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    MTL::ResourceOptions optionsManagedBuffer = MTL::ResourceStorageModeManaged  | MTL::ResourceCPUCacheModeDefaultCache;
    auto bufferVec1 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsManagedBuffer);
    auto bufferVec2 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsManagedBuffer);
    auto bufferVec3 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsManagedBuffer);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, optionsManagedBuffer);

    // Error handling example for buffer creation
    if ( !bufferVec1 || !bufferVec2 || !bufferVec3 || !bufferCoreParameters )
    {
        std::cerr << "Failed to create one or more shared buffers!" << std::endl;
        std::exit(1);
    }

    // Copy data directly into managed buffers
    memcpy(bufferVec1->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferVec2->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Synchronize managed buffers from CPU to GPU
    bufferVec1->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferVec2->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferCoreParameters->didModifyRange(NS::Range::Make(0, deviceCoreParams.bytes));

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();

    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 0);
    computeCommandEncoder->setBuffer(privateBuffer, 0, 1);
    computeCommandEncoder->setBuffer(bufferVec1, 0, 2);
    computeCommandEncoder->setBuffer(bufferVec2, 0, 3);
    computeCommandEncoder->setBuffer(bufferVec3, 0, 4);

    // Threads per block (1<<10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};
    MTL::Size NUM_THREADS_PER_GRID = {hostCoreParams.NUM_ELEMENTS / hostCoreParams.DATAPOINTS_PER_THREAD, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = ((hostCoreParams.NUM_ELEMENTS / hostCoreParams.DATAPOINTS_PER_THREAD) + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    kernelTimer.start(true);
    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);
    //computeCommandEncoder->dispatchThreads(NUM_THREADS_PER_GRID, NUM_THREADS_PER_THREADGROUP);

    // Finalise encoding
    computeCommandEncoder->endEncoding();
    
    // Asynchronously submit the command buffer containing the kernel to the device (GPU)
    //commandBuffer->GPUStartTime()
    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Inform host of changes made to the output buffer by the device
    bufferVec3->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferVec3->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, false);

    // Free memory on the device (GPU)
    bufferVec1->release();
    bufferVec2->release();
    bufferVec3->release();
    privateBuffer->release();
    bufferCoreParameters->release();

    // Release all other resources
    CoffeeExampleAsync::releaseResources();
}

void CoffeeExampleAsync::vectorAdditionOptimisedCaches( const int &numElements, const std::string &kernelFunctionName,
                                    const bool &fullDebug) {
    /*
     * Change log:
     *              - Separate the types of buffer into input (CPU write / GPU read), output (CPU & GPU, read & write),
     *              and private (GPU only) to take advantage of CPU-caching bypasses.
     *              - Flatten the input vectors to enable one input (CPU write / GPU read) buffer to be used. This
     *              reduces the total number of managed buffers, increases buffer length (minimising time to allocate),
     *              and reduces the number of buffers to be synced.
     *                      - As the length of the combined buffer is simply
     *                      2*N, this also means the striding of the GPU is regular.
*                   - Keep output in its own buffer as this is the only buffer that actually needs to be shared.
     */
    hostCoreParams.NUM_ELEMENTS_WITH_DATA = numElements;

    // Number of elements for each thread to compute
    hostCoreParams.calculateDataPerThread(1, 4);
    hostCoreParams.SIZE_DTYPE = sizeof(float);

    FULL_DEBUG = fullDebug;

    Timer gpuTimer, kernelTimer;
    gpuTimer.setName("Overall Timer (Vector Addition)(Async Optimised Caches)");
    kernelTimer.setName("Kernel Timer (Vector Addition)(Async Optimised Caches)");

    // Initialise random numbers in each array
    //if (vec1.empty() || vec1.size() != hostCoreParams.NUM_ELEMENTS)
    vec1 = getRandomVector(hostCoreParams.NUM_ELEMENTS_WITH_DATA);
    padVectorForCacheLine(vec1, hostCoreParams);
    
    //if  (vec2.empty() || vec2.size() != hostCoreParams.NUM_ELEMENTS)
    vec2 = getRandomVector(hostCoreParams.NUM_ELEMENTS_WITH_DATA);
    padVectorForCacheLine(vec2, hostCoreParams);

    std::vector<float> vec3(hostCoreParams.NUM_ELEMENTS_WITH_DATA);
    padVectorForCacheLine(vec3, hostCoreParams);
    
    hostCoreParams.calculateSize();

    // Exclude costly vector population as this isn't part of the timings tests.
    gpuTimer.start(true);

    CoreParamsForDevice deviceCoreParams(hostCoreParams.NUM_ELEMENTS_WITH_DATA, hostCoreParams.NUM_ELEMENTS_PADDING,
                                         hostCoreParams.DATAPOINTS_PER_THREAD);

    if (FULL_DEBUG)
    {
        std::cout << "(Vector Addition)(Async Optimised Caches) | Elements in vectors. (vec1): " << vec1.size() << " | (vec2): " << vec2.size() << std::endl;
    }

    // Initialise Metal device, command queue, and pipeline
    CoffeeExampleAsync::initialiseResources(kernelFunctionName);

    // Allocate shared memory for the host and device
    MTL::ResourceOptions optionsBufferIn = MTL::ResourceStorageModeManaged  | MTL::ResourceCPUCacheModeWriteCombined;
    MTL::ResourceOptions optionsBufferOut = MTL::ResourceStorageModeManaged  | MTL::ResourceCPUCacheModeDefaultCache;

    auto bufferInVec1 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsBufferIn);
    auto bufferInVec2 = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsBufferIn);
    auto bufferOutVec = device->newBuffer(hostCoreParams.BYTES_FOR_ELEMENTS, optionsBufferOut);
    auto bufferCoreParameters = device->newBuffer(deviceCoreParams.bytes, optionsBufferIn);

    // Error handling example for buffer creation
    if ( !bufferInVec1 || !bufferInVec2 || !bufferOutVec || !bufferCoreParameters )
    {
        std::cerr << "(Vector Addition)(Async Optimised Caches) | Failed to create one or more shared buffers!" << std::endl;
        std::exit(1);
    }

    // Copy data directly into managed buffers
    memcpy(bufferInVec1->contents(), vec1.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferInVec2->contents(), vec2.data(), hostCoreParams.BYTES_FOR_ELEMENTS);
    memcpy(bufferCoreParameters->contents(), &deviceCoreParams, deviceCoreParams.bytes);

    // Synchronize managed buffers from CPU to GPU
    bufferInVec1->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferInVec2->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));
    bufferCoreParameters->didModifyRange(NS::Range::Make(0, deviceCoreParams.bytes));

    // Encode commands
    computeCommandBuffer = commandQueuePrimary->commandBuffer();
    computeCommandEncoder = computeCommandBuffer->computeCommandEncoder();

    computeCommandEncoder->setComputePipelineState(computePrimaryPipelineState);

    computeCommandEncoder->setBuffer(bufferCoreParameters, 0, 0);
    computeCommandEncoder->setBuffer(privateBuffer, 0, 1);
    computeCommandEncoder->setBuffer(bufferInVec1, 0, 2);
    computeCommandEncoder->setBuffer(bufferInVec2, 0, 3);
    computeCommandEncoder->setBuffer(bufferOutVec, 0, 4);

    // Threads per block (1 << 10 == 1024)
    hostCoreParams.NUM_THREADS = 1 << 10;
    if ( FULL_DEBUG && (hostCoreParams.NUM_THREADS > device->maxThreadsPerThreadgroup().height ) )
    {
        std::cout << "(Vector Addition)(Async Optimised Caches) | Attempted to set more threads per thread-group than device permits\n";
        std::exit(1);
    }
    MTL::Size NUM_THREADS_PER_THREADGROUP = {hostCoreParams.NUM_THREADS, 1, 1};
    MTL::Size NUM_THREADS_PER_GRID = {hostCoreParams.NUM_ELEMENTS / hostCoreParams.DATAPOINTS_PER_THREAD, 1, 1};

    // Blocks per Grid (padded as required)
    hostCoreParams.NUM_THREADGROUPS = ((hostCoreParams.NUM_ELEMENTS / hostCoreParams.DATAPOINTS_PER_THREAD) + hostCoreParams.NUM_THREADS - 1) / hostCoreParams.NUM_THREADS;
    MTL::Size NUM_THREADGROUPS_PER_GRID = {hostCoreParams.NUM_THREADGROUPS, 1, 1};

    // Inform the kernel of the layout of the threads (similar to <<<NUM_BLOCKS, NUM_THREADS>>> in CUDA
    kernelTimer.start(true);
    computeCommandEncoder->dispatchThreadgroups(NUM_THREADGROUPS_PER_GRID, NUM_THREADS_PER_THREADGROUP);
    //computeCommandEncoder->dispatchThreads(NUM_THREADS_PER_GRID, NUM_THREADS_PER_THREADGROUP);

    // Finalise encoding
    computeCommandEncoder->endEncoding();

    // Asynchronously submit the command buffer containing the kernel to the device (GPU)
    //commandBuffer->GPUStartTime()

    computeCommandBuffer->commit();

    // Ensure synchronisation is explicitly managed (don't want to rely on memcpy for now)
    computeCommandBuffer->waitUntilCompleted();
    kernelTimer.stop();
    kernelTimer.print();

    // Inform host of changes made to the output buffer by the device
    bufferOutVec->didModifyRange(NS::Range::Make(0, hostCoreParams.BYTES_FOR_ELEMENTS));

    // Copy sum vector from device to host (GPU -> CPU)
    memcpy(vec3.data(), bufferOutVec->contents(), hostCoreParams.BYTES_FOR_ELEMENTS);

    // No need to time checking results; not part of test
    gpuTimer.stop();
    gpuTimer.print();

    // Check result for errors
    verifyResultsSineAddition(vec1, vec2, vec3, deviceCoreParams, false);

    // Free memory on the device (GPU)
    bufferInVec1->release();
    bufferInVec2->release();
    bufferOutVec->release();
    privateBuffer->release();
    bufferCoreParameters->release();

    // Release all other resources
    CoffeeExampleAsync::releaseResources();
}