//
// Created by Cameron Aidan McEleney on 27/03/2024.
//

#ifndef HELLO_METAL_COFFEEWITHARCH_EXAMPLES_H
#define HELLO_METAL_COFFEEWITHARCH_EXAMPLES_H

#include <Metal/Metal.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

class CommonTools {
public:
    struct CoreParamsForDevice {
        size_t NUM_ELEMENTS;
        size_t NUM_ELEMENTS_WITH_DATA;
        size_t NUM_ELEMENTS_PADDING;
        ushort NUM_ELEMENTS_PER_THREAD;
        ushort NUM_VEC4;
        size_t bytes;

        // Method to calculate the size of the struct
        void calculateSize() {
            // Since the struct contains static-sized elements only,
            // we can directly use sizeof to get its total size.
            // This includes all its members, such as 'N' and 'bytes'.
            bytes = sizeof(*this);
        }

        // Constructor to automatically calculate bytes upon creation
        explicit CoreParamsForDevice(size_t N, size_t numDp) : NUM_ELEMENTS_WITH_DATA(N), NUM_ELEMENTS_PER_THREAD(numDp) {
            NUM_VEC4 = std::ceil(NUM_ELEMENTS_PER_THREAD / 4);  // ensures we have at least one float 4 per thread
            NUM_ELEMENTS_PADDING = 0;
            NUM_ELEMENTS = NUM_ELEMENTS_WITH_DATA;
            calculateSize();
        }

        explicit CoreParamsForDevice(size_t N_true, size_t N_pad,size_t numDp) : NUM_ELEMENTS_WITH_DATA(N_true),
        NUM_ELEMENTS_PADDING(N_pad), NUM_ELEMENTS_PER_THREAD(numDp) {
            NUM_VEC4 = std::ceil(NUM_ELEMENTS_PER_THREAD / 4);
            NUM_ELEMENTS = NUM_ELEMENTS_WITH_DATA + NUM_ELEMENTS_PADDING;
            calculateSize();
        }
    };

    struct CoreParamsForHost {
        size_t NUM_ELEMENTS = -1;
        size_t NUM_ELEMENTS_WITH_DATA = -1;
        size_t NUM_ELEMENTS_PADDING = -1;
        ushort NUM_THREADS = -1;
        size_t NUM_THREADGROUPS = -1;
        ushort DATAPOINTS_PER_CONTAINER = -1;
        ushort CONTAINERS_PER_THREAD = -1;
        ushort DATAPOINTS_PER_THREAD = -1;
        ushort SIZE_DTYPE = -1;
        size_t BYTES_FOR_ELEMENTS = -1;

        void calculateDataPerThread(size_t containersPerThread, size_t datapointsPerContainer)
        {
            CONTAINERS_PER_THREAD = containersPerThread;
            DATAPOINTS_PER_CONTAINER = datapointsPerContainer;
            DATAPOINTS_PER_THREAD = datapointsPerContainer * containersPerThread;
        }

        void calculateSize() {

            if (NUM_ELEMENTS_PADDING >= 0)
            {
                NUM_ELEMENTS = NUM_ELEMENTS_WITH_DATA + NUM_ELEMENTS_PADDING;
            }
            else
            {
                std::cout << "Padding less than 0 during host calculateSize." << std::endl;
                std::exit(1);
            }

            BYTES_FOR_ELEMENTS = NUM_ELEMENTS * SIZE_DTYPE;
        }

        // Constructor to automatically calculate bytes upon creation
        explicit CoreParamsForHost(size_t N) : NUM_ELEMENTS_WITH_DATA(N) {
            SIZE_DTYPE = sizeof(float);
            NUM_ELEMENTS_PADDING = 0;
            calculateDataPerThread(1, 1);
            calculateSize();
        }
        CoreParamsForHost() = default;
    };

    struct Timer {
        std::string timerName;
        std::chrono::time_point<std::chrono::high_resolution_clock> startSolver;
        std::chrono::time_point<std::chrono::high_resolution_clock> endSolver;
        long long solverElapsedTime = 0;
        bool useMilliseconds = false;

        void start(bool useMs = false) {
            if (useMs)
                useMilliseconds = true;
            startSolver = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            endSolver = std::chrono::high_resolution_clock::now();
            if (useMilliseconds)
                solverElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endSolver - startSolver).count();
            else
                solverElapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endSolver - startSolver).count();
        }

        void setName( const std::string &name ) {
            timerName = name;
        }

        void cleanUp() {
            // Clear the timerName and reset the start and end times to ensure it's fully reset.
            timerName.clear();
            startSolver = std::chrono::time_point<std::chrono::high_resolution_clock>();
            endSolver = std::chrono::time_point<std::chrono::high_resolution_clock>();
            solverElapsedTime = 0;
            useMilliseconds = false;
        }

        void print() {
            std::cout << "----------------------------------------------------------------\n";
            if (!timerName.empty())
                std::cout << "Timing Information - " << timerName << "\n";
            else
                std::cout << "\nTiming Information. \n";

            // Directly printing the elapsed time as high_resolution_clock does not directly support conversion to a time_t value.
            if (useMilliseconds)
                std::cout << "\tElapsed: " << solverElapsedTime << " [milliseconds]\n";
            else
                std::cout << "\tElapsed: " << solverElapsedTime << " [seconds]\n";

            std::cout << "----------------------------------------------------------------\n";

            cleanUp();
        }
    };

public:
    static std::vector<float> getRandomVector(size_t size);

    static void padVectorForCacheLine(std::vector<float>& vec, CoreParamsForHost &coreParams);

    static void verifyResultsAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                              const std::vector<float> &resultVec, const bool &printAnswer);

    static void verifyResultsSineAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                              const std::vector<float> &resultVec, const bool &printAnswer);

    static void verifyResultsSineAddition(const std::vector<float> &inVecA, const std::vector<float> &inVecB,
                          const std::vector<float> &resultVec, CoreParamsForDevice &coreParams, const bool &printAnswer);
};

class CoffeeExample : public CommonTools {
public:
    void vectorAddition( const int &numElements, const std::string &kernelFunctionName, const bool &fullDebug );
    void vectorAdditionPrivateResources( const int &numElements, const std::string &kernelFunctionName,
                                         const bool &fullDebug);
    void vectorAdditionManagedResources( const int &numElements, const std::string &kernelFunctionName,
                                     const bool &fullDebug);
    void vectorAdditionFullyManagedResources( const int &numElements, const std::string &kernelFunctionName,
                                 const bool &fullDebug);

private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueuePrimary;
    MTL::ComputePipelineState* computePrimaryPipelineState;
    NS::Error* error = nullptr;
    NS::String* libraryPath;
    MTL::Library* library;
    MTL::Function* kernelFunction;
    MTL::CommandBuffer* computeCommandBuffer;
    MTL::ComputeCommandEncoder* computeCommandEncoder;

    std::vector<float> vec1;
    std::vector<float> vec2;

    bool FULL_DEBUG;

    void initialiseResources(const std::string& kernelFunctionName);
    void releaseResources();

    void processVectorAddition(const std::vector<float> &inA, const std::vector<float> &inB, std::vector<float> &outC);
};

class CoffeeExampleAsync : public CommonTools {
public:
    void vectorAdditionHeaps( const int &numElements, const std::string &kernelFunctionName,
                             const bool &fullDebug);

    void vectorAdditionOptimisedCaches( const int &numElements, const std::string &kernelFunctionName,
                         const bool &fullDebug);
private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueuePrimary;
    MTL::ComputePipelineState* computePrimaryPipelineState;
    NS::Error* error = nullptr;
    NS::String* libraryPath;
    MTL::Library* library;
    MTL::Function* kernelFunction;
    MTL::CommandBuffer* computeCommandBuffer;
    MTL::ComputeCommandEncoder* computeCommandEncoder;

    std::vector<float> vec1;
    std::vector<float> vec2;

    bool FULL_DEBUG;

    CoreParamsForHost hostCoreParams;

    MTL::Heap* privateHeap;
    MTL::Buffer* privateBuffer;

    void initialiseResources(const std::string& kernelFunctionName);
    void initialiseMemoryAllocations();
    void releaseResources();
};
#endif //HELLO_METAL_COFFEEWITHARCH_EXAMPLES_H
