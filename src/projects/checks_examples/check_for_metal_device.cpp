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