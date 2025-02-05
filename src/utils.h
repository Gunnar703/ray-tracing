#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <iostream>

// CUDA error checking macro
#define CHECK_CUDA(call) check_cuda( (call), #call, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char * const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "'\n";

        cudaDeviceReset();
        exit(99);
    }
}

// Random number functions
__device__ inline float cuda_random(curandState* rs) {
    return curand_uniform(rs);
}

__device__ inline float cuda_random(curandState* rs, float min, float max) {
    return min + (max - min) * curand_uniform(rs);
}

#include "vec3.h"
#include "imageio.h"
#include "ray.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"

#endif