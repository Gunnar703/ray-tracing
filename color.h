#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

struct rgb_int {
    int r;
    int g; 
    int b;
};

inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0)
        return std::sqrt(linear_component);
    
    return 0;
}

rgb_int color_to_rgb_int(const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Apply a linear -> gamma transform for gamma=2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0, 1] component values to the byte range [0, 255].
    static const interval intensity(0.000, 0.999);
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    rgb_int out = {rbyte, gbyte, bbyte};
    return out;
}

#endif