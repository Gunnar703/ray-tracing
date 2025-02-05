#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

struct rgb_int {
    int r;
    int g; 
    int b;
};

__device__ inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0)
        return std::sqrt(linear_component);
    
    return 0;
}

#endif