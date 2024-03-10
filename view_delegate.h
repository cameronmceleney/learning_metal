//
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#ifndef HELLO_METAL_VIEW_DELEGATE_H
#define HELLO_METAL_VIEW_DELEGATE_H

#include "config.h"
#include "renderer.h"

class ViewDelegate : public MTK::ViewDelegate {
public:
    // Read their implementations to see that everything created is destroyed
    explicit ViewDelegate(MTL::Device* device);
    ~ViewDelegate() override;
public:
    // Allows for a message to be passed to a delegate to draw in the view
    void drawInMTKView(MTK::View* view) override;

private:
    // This class manages the rendering of the view; remember that the remember is where we do most of our coding
    Renderer* renderer;
};


#endif //HELLO_METAL_VIEW_DELEGATE_H
