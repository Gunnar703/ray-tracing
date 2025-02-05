#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"

// Ray Color Functions
__device__ color ray_color(const ray& r, hittable_list** d_world) {
    hit_record rec;

    bool hit_anything = (*d_world)->hit(r, 0.0f, FLT_MAX, rec);
    if (hit_anything) {
        return 0.5f * ( rec.normal + color(1.0f, 1.0f, 1.0f) );
    }
    
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f-t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

// Render kernel
__global__ void render(
    hittable_list** d_world,
    color* frame_buffer, 
    int image_width, 
    int image_height,
    point3 pixel00_loc,
    vec3 pixel_delta_u,
    vec3 pixel_delta_v,
    point3 center
) {
    // Get the current pixel and make sure it's in bounds
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    int pixel_index = j * image_width + i;

    ray r(center, pixel00_loc + i*pixel_delta_u + j*pixel_delta_v);

    frame_buffer[pixel_index] = ray_color(r, d_world);
}

// Kernel to create world on the device
__global__ void create_world(hittable** d_list, hittable_list** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list + 0) = new sphere(point3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(point3(0, -100.5, -1), 100);

        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void free_world(hittable **d_list, hittable_list** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *(d_list + 0);
        delete *(d_list + 1);
        delete *(d_world);
    }
}

int main() {
    // Create world
    hittable** d_list;
    hittable_list** d_world;
    int n_objects = 2;

    CHECK_CUDA(cudaMalloc((void**)&d_list, n_objects * sizeof(hittable*)));
    CHECK_CUDA(cudaMalloc((void**)&d_world, sizeof(hittable_list*)));
    
    create_world<<<1, 1>>>(d_list, d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Image height and width
    int image_width  = 1200;
    int image_height = 600;

    // Camera
    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * float(image_width) / image_height;
    point3 center(0, 0, 0);

    vec3 viewport_u(viewport_width, 0, 0);
    vec3 viewport_v(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / float(image_width);
    vec3 pixel_delta_v = viewport_v / float(image_height);

    point3 viewport_upper_left = center 
                               - vec3(0, 0, focal_length) 
                               - viewport_u / 2 
                               - viewport_v / 2;
    point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Determine how many threads/blocks to use
    int threads_x = 8;
    int threads_y = 8;

    std::cout << "Rendering a " << image_width << "x" << image_height << " image "
              << "in " << threads_x << "x" << threads_y << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t frame_buffer_size = num_pixels * sizeof(color);

    // Allocate framebuffer
    color* frame_buffer;
    CHECK_CUDA(cudaMallocManaged( (void**)&frame_buffer, frame_buffer_size ));

    // Render the buffer
    dim3 blocks(image_width / threads_x + 1, image_height / threads_y + 1);
    dim3 threads(threads_x, threads_y);

    render<<<blocks, threads>>>(
        d_world,
        frame_buffer,
        image_width,
        image_height,
        pixel00_loc,
        pixel_delta_u,
        pixel_delta_v,
        center
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    output_png(frame_buffer, "image.png", image_width, image_height);

    CHECK_CUDA(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(*d_world));
    CHECK_CUDA(cudaFree(*d_list));
    CHECK_CUDA(cudaFree(frame_buffer));

    cudaDeviceReset();
}