//
// boilerplate_example.cpp
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#include "graphical_example_m.h"

void GraphicalExamples::generateSquare() {

    bool useExample = false;

    if (useExample){
        NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
        AppDelegate controller;

        NS::Application* app = NS::Application::sharedApplication();
        app->setDelegate(&controller);
        app->run();

        autoreleasePool->release();
    }
    else {
        // Create an autorelease pool to manage the memory of the application.
        NS::AutoreleasePool *autoreleasePool = NS::AutoreleasePool::alloc()->init();

        // The application delegate is the main controller for the application.
        auto *controller = new AppDelegate();

        // Create our application
        NS::Application *app = NS::Application::sharedApplication();
        app->setDelegate(controller);
        app->run();

        // Manual cleanup
        delete controller;
        autoreleasePool->release();
    }
}
