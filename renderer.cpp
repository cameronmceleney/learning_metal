//
// renderer.cpp
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#include "renderer.h"

Renderer::Renderer(MTL::Device* device) :
    device(device->retain()) {
    /*
     * The retain method is called upon the device pointer to create a copy that increments the reference count
     * on the device that guarantees that so long as the Renderer class doesn't free the device, we have a
     * guarantee that the *device pointer will be valid.
     */
    commandQueue = device->newCommandQueue();
}

Renderer::~Renderer() {
    // All that's created must be destroyed
    commandQueue->release();
    device->release();
}

void Renderer::draw(MTK::View* view) {
    std::cout << "Drawing the view" << std::endl;
    // We're likely to have many objects in a draw method, so we create an autorelease pool to manage them
    // and scope them to this pool

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // Create a command buffer to store the commands that will be sent to the GPU
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Describes all of the resources that will be accessed/read/written in the render pass
    MTL::RenderPassDescriptor* renderPass = view->currentRenderPassDescriptor();

    // Responsible for recording drawing commands and then submitting them to the GPU
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPass);

    // Stop encoding immediately to invoke default behaviour (screen clear)
    encoder->endEncoding();

    // Presents any graphics that were drawn by 'recalling' a present operation to the command buffer
    commandBuffer->presentDrawable(view->currentDrawable());

    // 'Remember' all of the work
    commandBuffer->commit();

    pool->release();
}