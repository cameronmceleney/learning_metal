//
// app_delegate.cpp
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#include "app_delegate.h"

AppDelegate::~AppDelegate() {
    std::cout << "AppDelegate destroyed.\n";
    mtkView->release();
    window->release();
    device->release();
    delete viewDelegate;
}

void AppDelegate::applicationWillFinishLaunching( NS::Notification *notification ) {
    std::cout << "Application will finish launching is called." << std::endl;
    NS::Application *app = reinterpret_cast<NS::Application*>(notification->object());
    app->setActivationPolicy(NS::ActivationPolicy::ActivationPolicyRegular);
}

void AppDelegate::applicationDidFinishLaunching( NS::Notification *notification ) {
    std::cout << "Application did finish launching is called." << std::endl;

    // Computer graphics rectangle specifying the size and position of the window
    CGRect frame = (CGRect){ {100.0, 100.0}, {640.0, 480.0} };
    window = NS::Window::alloc()->init(
            frame,
            NS::WindowStyleMaskClosable | NS::WindowStyleMaskTitled,
            NS::BackingStoreBuffered,
            false);

    device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        std::cout << "Device creation failed." << std::endl;
    } else {
        std::cout << "Device successfully created." << std::endl;
    }

    // Manually managed object (mtkView) and then freed; AppDelegate destructor is responsible for this
    mtkView = MTK::View::alloc()->init(frame, device);

    // Set the pixel colour and clear colour which impacts rendering
    mtkView->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
    mtkView->setClearColor(MTL::ClearColor::Make(1.0, 1.0, 0.6, 1.0));


    // Set the view delegate to the view delegate class which we have abstracted and encapsulated
    viewDelegate = new ViewDelegate(device);
    mtkView->setDelegate(viewDelegate);

    window->setContentView(mtkView);
    window->setTitle(NS::String::string("Window", NS::StringEncoding::UTF8StringEncoding));
    window->makeKeyAndOrderFront(nullptr);

    // Careful! Must use a pointer (NS::Application*) as the argument's type
    NS::Application *app = reinterpret_cast<NS::Application*>(notification->object());
    app->activateIgnoringOtherApps(true);
}

bool AppDelegate::applicationShouldTerminateAfterLastWindowClosed( NS::Application *sender ) {
    std::cout << "Application should terminate after last window closed is called." << std::endl;
    return true;
}