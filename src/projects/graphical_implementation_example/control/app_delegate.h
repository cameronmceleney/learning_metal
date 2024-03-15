//
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#ifndef HELLO_METAL_APP_DELEGATE_H
#define HELLO_METAL_APP_DELEGATE_H

#include "../../../../lib/config.h"
#include <iostream>
#include "view_delegate.h"

class AppDelegate : public NS::ApplicationDelegate {
public:
    // Needs constructor
    ~AppDelegate() override;

    // Managers for the application's delegate
    void applicationWillFinishLaunching(NS::Notification *notification) override;
    void applicationDidFinishLaunching(NS::Notification *notification) override;
    bool applicationShouldTerminateAfterLastWindowClosed(NS::Application *sender) override;

private:
    NS::Window* window;
    MTK::View* mtkView;
    MTL::Device* device;  // Abstraction of the GPU
    ViewDelegate* viewDelegate = nullptr;
};


#endif //HELLO_METAL_APP_DELEGATE_H
