#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

__global__ void create_world(hittable_list& d_world, material** material_list) {
    material_list[0] = new lambertian(color(1, 0, 0));
    material_list[1] = new lambertian(color(0, 1, 0));
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new sphere(vec3(0, 0, -1), 0.5, material_list[0]);
        d_world[1] = new sphere(vec3(0, -100.5, -1), 100, material_list[1]);
    }
}

int main() {

    // Camera Setup
    camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 500;
    cam.samples_per_pixel = 10;
    cam.max_depth         = 50;

    cam.vfov     = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    // Create world
    material** material_list;
    CUDA_CHECK(cudaMalloc((void**)&material_list, 2 * sizeof(material*)));

    hittable_list d_world(2);
    create_world<<<1, 1>>>(d_world, material_list);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Render
    render(d_world, "image.ppm", &cam);
}