#include "lodepng.h"
#include <fstream>
#include <iostream>

void output_ppm(
    color* frame_buffer, 
    std::string outfile_name,
    int image_width,
    int image_height
) {
    std::ofstream out;
    out.open(outfile_name);
    
    // Write PPM Header
    out << "P3\n" << image_width << " " << image_height << "\n255\n";

    // Write image
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;

            float r = frame_buffer[pixel_index][0];
            float g = frame_buffer[pixel_index][1];
            float b = frame_buffer[pixel_index][2];

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            out << ir << " " << ig << " " << ib << "\n";
        }
    }

    std::cout << "Wrote image to " << outfile_name << "\n";
}

void output_png(
    color* frame_buffer, 
    std::string outfile_name,
    int image_width,
    int image_height
) {
    std::vector<unsigned char> image;
    image.resize(image_width * image_height * 4);
    
    // Write image
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;

            float r = frame_buffer[pixel_index][0];
            float g = frame_buffer[pixel_index][1];
            float b = frame_buffer[pixel_index][2];

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            image[4*pixel_index + 0] = static_cast<unsigned char>(ir);
            image[4*pixel_index + 1] = static_cast<unsigned char>(ig);
            image[4*pixel_index + 2] = static_cast<unsigned char>(ib);
            image[4*pixel_index + 3] = 255;
        }
    }

    unsigned error = lodepng::encode(outfile_name, image, image_width, image_height);
    if(error) 
        std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    else 
        std::cout << "Wrote image to " << outfile_name << "\n";
}