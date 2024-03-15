//
// Created by Cameron Aidan McEleney on 14/03/2024.
//

#include "compute_function_examples.h"

void ComputeFunctionExamples::sumSimpleVectors() {
    int count = 10;

    // Create random vectors
    std::random_device rd;  // Initialises seed engine
    std::mt19937 rng(rd());  // Use Mersenne-Twister random-number engine
    double real_min = 0.0, real_max = 1.0;
    std::uniform_real_distribution<double> uni_rdm_real(real_min, real_max);

    std::vector<double> vec1 = ComputeFunctionExamples::getRandomVector(count, rng, uni_rdm_real);
    std::vector<double> vec2 = ComputeFunctionExamples::getRandomVector(count, rng, uni_rdm_real);

    // Call out functions
    std::cout << "Beginning to compute the sum of two vectors\n";
    computeSequential(vec1, vec2);
    computeParallel(vec1, vec2);
}

std::vector<double> ComputeFunctionExamples::getRandomVector( const int &count, std::mt19937 rng,
                                                              std::uniform_real_distribution<double> uni_rdm_real ) {
    std::vector<double> vec;

    for (int i = 0; i < count; i++)
        vec.push_back(uni_rdm_real(rng));

    return vec;
}

void ComputeFunctionExamples::computeSequential( std::vector<double> vector1, std::vector<double> vector2) {
    std::cout << "Computing Sequentially" << std::endl;

    // Begin process
    Timer timer;
    timer.setName("Sequential Timer");

    std::vector<double> result(vector1.size());

    timer.start(true);
    for ( int i = 0; i < vector1.size(); i++)
        result[i] = vector1[i] + vector2[i];
    timer.stop();

    // Print the results
    for (int i = 0; i < result.size(); i++)
        std::cout << "(" << vector1[i] << " + " << vector2[i] << " = " << result[i] << ")\n";

    // Print out the time
    timer.print();

}

void ComputeFunctionExamples::computeParallel( std::vector<double> vector1, std::vector<double> vector2) {
    std::cout << "Computing in Parallel" << std::endl;
}

void addition_seq_compute_function(const float* vec1in, const float* vec2in, float* result, int length) {
    // An ex
    for (int index = 0; index < length ; index++)
        result[index] = vec1in[index] + vec2in[index];
}

void ComputeFunctionExamples::addition_main() {
    NS::AutoreleasePool *autoreleasePool = NS::AutoreleasePool::alloc()->init();

    // Create the device
    MTL::Device *pDevice = MTL::CreateSystemDefaultDevice();

    // Create buffers to hold data

    // Send a command to the GPU to perform the calculation.

    autoreleasePool->release();
}