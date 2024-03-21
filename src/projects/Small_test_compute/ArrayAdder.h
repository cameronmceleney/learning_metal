//
// Created by Cameron Aidan McEleney on 15/03/2024.
//

#ifndef HELLO_METAL_ARRAYADDER_H
#define HELLO_METAL_ARRAYADDER_H

#include <Metal/Metal.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

class ArrayAdder {
public:
    // Adds elements of two input arrays using GPU and stores the result in the output array.
    // Parameters inA and inB are the input arrays, and outC is the output array where the result is stored.
    static void addArraysGPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC, bool complexAddition);
    static void addArraysCPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC);

    static void addArraysComplexCPU(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC);

    static void addArraysGpuWithChunking( const std::vector<float>& inA, const std::vector<float>& inB,
                                          std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu);
    void addArraysGpuChunkingDynamicBufferAsync(const std::vector<float>& inA, const std::vector<float>& inB,
                                                        std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu);

    int lengthVector = -1;

private:
    dispatch_semaphore_t semaphoreAsync;
    MTL::Device* deviceAsync;
    MTL::CommandQueue* commandQueueAsync;
    MTL::ComputePipelineState* computePipelineStateAsync;
    std::vector<MTL::Buffer*> bufferPoolAsync;
    size_t maxChunkSizeAsync; // Adjustable based on GPU vs CPU performance testing.
    size_t bufferIndexAsync = 0; // Current index for buffer swapping.

    void initializeResources(const std::string& kernelFunctionName);
    void releaseResources();
    void processChunks(const std::vector<float>& inA, const std::vector<float>& inB, std::vector<float>& outC, bool complexAddition, bool onlyOutputToCpu);
    MTL::Buffer* getNextBuffer();
    NS::Error* errorAsync = nullptr;

    const int numSemaphores = 9;

private:
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
};

#endif //HELLO_METAL_ARRAYADDER_H
