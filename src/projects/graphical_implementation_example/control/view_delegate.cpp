//
// view_delegate.cpp
// Created by Cameron Aidan McEleney on 10/03/2024.
//

#include "view_delegate.h"

ViewDelegate::ViewDelegate(MTL::Device* device) :
    MTK::ViewDelegate(),
    renderer(new Renderer(device)) {}

ViewDelegate::~ViewDelegate() {
    delete renderer;
}

void ViewDelegate::drawInMTKView(MTK::View* view) {
    // Intercepts a draw message and tells the renderer class to draw the view
    renderer->draw(view);
}