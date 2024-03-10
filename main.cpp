//
// main.cpp
// hello_metal.cpp
//
// Created by Cameron McEleney on 08 Mar 24
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION

// Local
#include "metalChecks.h"
#include "boilerplate_example.h"

int main() {
    //metalChecks::checkForDevice();
    BoilerPlateExample::run();
    //metalChecks::checkForDevice();
    return 0;
}
