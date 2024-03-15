//
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#ifndef HELLO_METAL_RENDERER_H
#define HELLO_METAL_RENDERER_H

#include "../../../../lib/config.h"
#include <iostream>

class Renderer {
public:
    /*
     * Note that the renderer is created with a device to render to as opposed to being created with
     *  a signal (like the view delegate)
     */
    explicit Renderer(MTL::Device* device);
    ~Renderer();
public:
    void draw(MTK::View* view);

private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
};


#endif //HELLO_METAL_RENDERER_H
