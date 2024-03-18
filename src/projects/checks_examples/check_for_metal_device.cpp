//
// Created by Cameron Aidan McEleney on 08/03/2024.
//

#include "check_for_metal_device.h"

void DeviceChecks::checkForDevice() {
    MTL::Device *pDevice = MTL::CreateSystemDefaultDevice();

    if ( pDevice ) {
        NS::String *deviceName = pDevice->name();
        if ( deviceName ) {
            // Convert NS::String to a C-style string or std::string if possible.
            std::cout << "Metal device used: " << deviceName->utf8String() << std::endl;
        }
        else
            std::cout << "Unable to obtain the Metal device name." << std::endl;
    }
    else
        std::cout << "Metal device not available on this device as pDevice: " << pDevice->name() << std::endl;

    pDevice->release();
}

void DeviceChecks::printDeviceInfo() {
    auto device = MTL::CreateSystemDefaultDevice();

    std::cout << "Name: " << device->name()->utf8String() << std::endl;
    std::cout << "Max Threads Per Threadgroup: " << device->maxThreadsPerThreadgroup().depth << std::endl;

    // Metal doesn't expose cache sizes or total memory directly.
    // Demonstrating maxBufferLength as an example of accessible property.
    std::cout << "Max Buffer Length: " << device->maxBufferLength() << std::endl;
    std::cout << "Has Unified Memory: " << device->hasUnifiedMemory() << std::endl;
    std::cout << "Max Buffer Arg. Count: " << device->maxArgumentBufferSamplerCount() << std::endl;
    std::cout << "Max Transfer Rate: " << device->maxTransferRate() << std::endl;
    std::cout << "Max ThreadGroups Memory Len.: " << device->maxThreadgroupMemoryLength() << std::endl;
    std::cout << "Thread Execution Width: 32 (hardcoded here from using Macbook M3 Pro)" << std::endl;

    std::cout << "Max Concurrent Compilation Task Count: " << device->maximumConcurrentCompilationTaskCount() << std::endl;

    // Release the device if you're done with it
    device->release();
}