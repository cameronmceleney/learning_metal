//
// Created by Cameron Aidan McEleney on 14/03/2024.
//

#ifndef HELLO_METAL_COMPUTE_FUNCTION_EXAMPLES_H
#define HELLO_METAL_COMPUTE_FUNCTION_EXAMPLES_H

#include "../../../lib/config.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

class ComputeFunctionExamples {
public:
    void sumSimpleVectors();

private:
    static std::vector<double> getRandomVector( const int &count, std::mt19937 rng,
                                                std::uniform_real_distribution<double> uni_rdm_real );
    void computeSequential( std::vector<double> vector1, std::vector<double> vector2);
    void computeParallel( std::vector<double> vector1, std::vector<double> vector2);

    void addition_main();
private:
    struct Timer {
        std::string timerName;
        std::chrono::time_point<std::chrono::system_clock> startSolver;
        std::chrono::time_point<std::chrono::system_clock> endSolver;
        long long solverElapsedTime = 0;
        bool useMilliseconds = false;

        void start(bool useMs = false) {
            if (useMs)
                useMilliseconds = true;
            startSolver = std::chrono::system_clock::now();
        }

        void stop() {
            endSolver = std::chrono::system_clock::now();
            if (useMilliseconds)
                solverElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endSolver - startSolver).count();
            else
                solverElapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endSolver - startSolver).count();
        }

        void setName( const std::string &name ) {
            timerName = name;
        }

        void cleanUp() {
            solverElapsedTime = 0;
            useMilliseconds = false;
        }

        void print() {
            auto printTimePoint = []( auto &timePoint) {
                                        std::time_t printTime = std::chrono::system_clock::to_time_t(timePoint);
                                        return std::ctime(&printTime);
            };

            std::cout << "----------------------------------------------------------------\n";
            if ( !timerName.empty())
                std::cout << "Timing Information - " << timerName << "\n";
            else
                std::cout << "\nTiming Information. \n";

            std::cout << "\tStart: " << printTimePoint(startSolver);
            std::cout << "\tEnd: " << printTimePoint(endSolver);
            if (useMilliseconds)
                std::cout << "\tElapsed: " << solverElapsedTime << " [milliseconds]" << std::endl;
            else
                std::cout << "\tElapsed: " << solverElapsedTime << " [seconds]" << std::endl;

            std::cout << "----------------------------------------------------------------\n";

            cleanUp();
        }
    };
};


#endif //HELLO_METAL_COMPUTE_FUNCTION_EXAMPLES_H
