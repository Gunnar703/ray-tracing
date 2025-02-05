#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"

#include <fstream>

#include <curand_kernel.h>

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    // Each thread gets same ssed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

class camera {
    public:
        double aspect_ratio      = 1.0;  // Ratio of image width over height
        int    image_height;             // Rendered image height
        int    image_width       = 100;  // Rendered image width in pixels        
        int    samples_per_pixel = 10;   // Count of random samples for each pixel
        int    max_depth         = 10;   // Maximum number of ray bounces into scene

        point3 center;               // Camera center
        point3 pixel00_loc;          // Location of pixel 0, 0
        vec3   pixel_delta_u;        // Offset to pixel to the right
        vec3   pixel_delta_v;        // Offset to pixel below

        double vfov     = 90;                // Vertical view angle (field of view)
        point3 lookfrom = point3(0, 0, 0);   // Point camera is looking from
        point3 lookat   = point3(0, 0, -1);  // Point camera is looking at
        vec3   vup      = vec3(0, 1, 0);     // Camera-relative "up" direction

        double defocus_angle = 0;   // Variation angle of rays through each pixel
        double focus_dist    = 10;  // Distance from camera lookfrom point to plane of perfect focus

        __device__ ray get_ray(int i, int j, curandState* rand_state) const {
            // Construct a camer ray originating from the defocus disk and directed at randomly sampled
            // point around the pixel location i, j.

            vec3 offset = sample_square(rand_state);
            point3 pixel_sample = pixel00_loc
                                + (i + offset.x()) * pixel_delta_u
                                + (j + offset.y()) * pixel_delta_v;

            point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
            vec3 ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        __device__ color ray_color(const ray& r, int depth, hittable_list* world, curandState* rand_state) {
            ray cur_ray = r;
            color cur_attenuation = color(1.0f, 1.0f, 1.0f);

            for (int i = 0; i < depth; i++) {
                hit_record rec;

                if (world->hit(cur_ray, interval(0.001f, CUDA_INFINITY), rec)) {
                    // If the ray strikes a surface
                    ray scattered;
                    color attenuation;

                    if (rec.mat->scatter(r, rec, attenuation, scattered, rand_state)) {
                        cur_attenuation = attenuation * cur_attenuation;
                        cur_ray = scattered;
                    }

                } else {
                    // If the ray does not strike a surface, render the background and return
                    vec3 unit_direction = unit_vector(cur_ray.direction());
                    float t = 0.5f * (unit_direction.y() + 1.0f);
                    vec3 c = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
                    return cur_attenuation * c;
                }
            }

            // Exceeded recursion - this is basically assuming that if a ray bounces more than 'max_depth' times, it will never return to the atmosphere and is therefore black.
            return vec3(0.0f, 0.0f, 0.0f);
        }

        __host__ void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;

            // Determine viewport dimensions
            double theta           = degrees_to_radians(vfov);
            double h               = std::tan(theta/2);
            double viewport_height = 2.0 * h * focus_dist;
            double viewport_width  = viewport_height * double(image_width) / image_height;

            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges
            vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
            vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

            // Calculate the horizontal and vertical delta vectors from pixel to pixel
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate the location of the upper left pixel
            point3 viewport_upper_left = center
                                       - (focus_dist * w)
                                       - viewport_u / 2
                                       - viewport_v / 2;

            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors.
            double defocus_radius = focus_dist * std::tan( degrees_to_radians( defocus_angle / 2 ) );
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

    private:
        double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
        vec3   u, v, w;              // Camera frame basis vectors
        vec3   defocus_disk_u;       // Defocus disk horizontal radius
        vec3   defocus_disk_v;       // Defocus disk vertical radius

        __device__ vec3 sample_square(curandState* rand_state) const {
            // Returns the vector to a random point in the [-.5, .5] x [-.5, .5] unit square.
            return vec3(curand_uniform(rand_state) - 0.5, curand_uniform(rand_state) - 0.5, 0);
        }

        __device__ point3 defocus_disk_sample (curandState* rand_state) const {
            // Returns a random point in the camera defocus disk.
            vec3 p = random_in_unit_disk(rand_state);

            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
        }
};

__global__ void render_kernel(
    float* frame_buffer, 
    int max_x, 
    int max_y,
    hittable_list* d_world,
    point3 center,
    point3 pixel00_loc,
    vec3 pixel_delta_u,
    vec3 pixel_delta_v,
    curandState* rand_state,
    int samples_per_pixel,
    camera* cam
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int pixel_index = j*max_x + i;  // For setting the pixel in frame-buffer
    curandState local_rand_state = rand_state[pixel_index];

    // Bounds-check
    if ((i >= max_x) || (j >= max_y)) return;

    point3 pixel_center = pixel00_loc
                        + i * pixel_delta_u
                        + j * pixel_delta_v;
    vec3 ray_direction = pixel_center - center;
    ray r(center, ray_direction);

    color pixel_color(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_per_pixel; s++) {
        ray r = cam->get_ray(i, j, &local_rand_state);
        pixel_center += cam->ray_color(r, cam->max_depth, d_world, &local_rand_state);
    }

    frame_buffer[3*pixel_index + 0] = pixel_color[0] / float(samples_per_pixel);
    frame_buffer[3*pixel_index + 1] = pixel_color[1] / float(samples_per_pixel);
    frame_buffer[3*pixel_index + 2] = pixel_color[2] / float(samples_per_pixel);
}

__host__ void render(hittable_list d_world, std::string output_filename, camera* cam) {
    cam->initialize();

    // Determine how many blocks/threads to use
    int tx = 4;
    int ty = 4;

    dim3 blocks(cam->image_width/tx+1, cam->image_height/ty+1);
    dim3 threads(tx, ty);

    // Initialize the per-pixel random state
    curandState *d_rand_state;
    CUDA_CHECK(cudaMalloc((void**)&d_rand_state, cam->image_width * cam->image_height * sizeof(curandState)));

    render_init<<<blocks, threads>>>(cam->image_width, cam->image_height, d_rand_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate the frame buffer
    size_t fb_size = 3 * cam->image_width * cam->image_height * sizeof(float);
    float* frame_buffer;
    CUDA_CHECK(cudaMallocManaged((void **)&frame_buffer, fb_size));

    // Render the image
    std::clog << "Rendering...\n";
    render_kernel<<<blocks, threads>>>(
        frame_buffer,
        cam->image_width,
        cam->image_height,
        &d_world,
        cam->center,
        cam->pixel00_loc,
        cam->pixel_delta_u,
        cam->pixel_delta_v,
        d_rand_state,
        cam->samples_per_pixel,
        cam
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::clog << "Rendered with no errors.\n";

    // Print it to the output file
    std::clog << "Writing to '" << output_filename << "'\n";
    
    std::ofstream out;
    out.open(output_filename);
    out << "P3\n" << cam->image_width << ' ' << cam->image_height << "\n255\n";  // PPM header

    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            size_t pixel_index = j * cam->image_width + i;

            float r = frame_buffer[3*pixel_index + 0];
            float g = frame_buffer[3*pixel_index + 1];
            float b = frame_buffer[3*pixel_index + 2];

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            out << ir << " " << ig << " " << ib << "\n";
        }
    }

    out.close();
    std::clog << "Done writing file.";

    CUDA_CHECK(cudaFree(d_rand_state));
    CUDA_CHECK(cudaFree(frame_buffer));
}

#endif