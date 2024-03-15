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

    // Release the device if you're done with it
    device->release();
}