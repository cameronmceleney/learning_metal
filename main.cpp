//
// main.cpp
// hello_metal.cpp
//
// Created by Cameron McEleney on 08 Mar 24
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation.hpp>
#include <Metal.hpp>
#include <QuartzCore.hpp>
#include <AppKit.hpp>

#include "metalChecks.h"

int main() {
    metalChecks::checkForDevice();
    return 0;
}
