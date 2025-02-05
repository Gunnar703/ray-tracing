#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <vector>

class hittable_list : public hittable {
    public:
        hittable** objects;
        int len;

        hittable_list(int len) : len(len) {
            CUDA_CHECK(cudaMalloc( (void**)&objects, len * sizeof(hittable*) ));
        }

        ~hittable_list() {
            CUDA_CHECK(cudaFree(objects));
        }

        __device__ hittable*& operator [](int i) {
            return objects[i];
        }

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;

            double closest_so_far = ray_t.max;

            for (int i = 0; i < len; i++) {
                hittable* object = objects[i];
                if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

};

#endif