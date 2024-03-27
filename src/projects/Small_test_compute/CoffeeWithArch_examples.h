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

class CoffeeExample {
public:
    void vectorAddition( const int &numElements, const std::string &kernelFunctionName, const bool &fullDebug );
    void vectorAdditionPrivateResources( const int &numElements, const std::string &kernelFunctionName,
                                         const bool &fullDebug);
    void vectorAdditionManagedResources( const int &numElements, const std::string &kernelFunctionName,
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

    bool FULL_DEBUG;

    static std::vector<int> getRandomVector(size_t size);
    static void verifyResults(const std::vector<int> &inVecA, const std::vector<int> &inVecB,
                              const std::vector<int> &resultVec, const bool &printAnswer);

    void initialiseResources(const std::string& kernelFunctionName);
    void processVectorAddition(const std::vector<int> &inA, const std::vector<int> &inB, std::vector<int> &outC);
    void releaseResources();

private:
    struct CoreParamsForDevice {
        size_t NUM_ELEMENTS;
        size_t bytes;

        // Method to calculate the size of the struct
        void calculateSize() {
            // Since the struct contains static-sized elements only,
            // we can directly use sizeof to get its total size.
            // This includes all its members, such as 'N' and 'bytes'.
            bytes = sizeof(*this);
        }

        // Constructor to automatically calculate bytes upon creation
        explicit CoreParamsForDevice(size_t N) : NUM_ELEMENTS(N) {
            calculateSize();
        }
    };

    struct CoreParamsForHost {
        size_t NUM_ELEMENTS;
        size_t NUM_THREADS;
        size_t NUM_THREADGROUPS;
        size_t SIZE_DTYPE;
        size_t BYTES_FOR_ELEMENTS;

        void calculateSize() {
            SIZE_DTYPE = sizeof(int);
            BYTES_FOR_ELEMENTS = NUM_ELEMENTS * SIZE_DTYPE;
        }

        // Constructor to automatically calculate bytes upon creation
        explicit CoreParamsForHost(size_t N) : NUM_ELEMENTS(N) {
            calculateSize();
        }
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
};


#endif //HELLO_METAL_COFFEEWITHARCH_EXAMPLES_H
