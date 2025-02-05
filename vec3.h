#ifndef VEC3_H
#define VEC3_H

#include <curand_kernel.h>

class vec3 {
    public:
        float e[3];

        __host__ __device__ vec3() : e{0, 0, 0} {}  // Default constructor - for when vec3 is initialized without arguments
        __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}  // Constructor for when e0 e1 and e2 are specified

        __host__ __device__ float x() const { return e[0]; }  // NOTE: `const` means that x() doesn't modify any member variables
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ float operator[](int i) const { return e[i]; }
        __host__ __device__ float& operator[](int i) { return e[i]; }

        __host__ __device__ vec3& operator +=(const vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator *=(float t) {
            e[0] += t;
            e[1] += t;
            e[2] += t;
            return *this;
        }

        __host__ __device__ vec3& operator /=(float t) {
            return *this *= 1/t;
        }

        __host__ __device__ float length() const {
            return std::sqrt(length_squared());
        }

        __host__ __device__ float length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        __host__ __device__ bool near_zero() const {
            // Return true if the vector is close to zero in all dimensions
            float s = 1e-8;
            return (fabsf(e[0]) < s) 
                   && (fabsf(e[1] < s))
                   && (fabsf(e[2]) < s);
        }

        __device__ static vec3 random(curandState* rs) {
            return vec3(curand_uniform(rs), curand_uniform(rs), curand_uniform(rs));
        }

        __device__ static vec3 random(float min, float max, curandState* rs) {
            return vec3(
                curand_uniform(rs) * (max - min) + min,    
                curand_uniform(rs) * (max - min) + min,    
                curand_uniform(rs) * (max - min) + min   
            );
        }
};   

// Alias point3 to vec3. Helps wiht geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions
__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(
        u.e[0] * v.e[0],
        u.e[1] * v.e[1],
        u.e[2] * v.e[2]
    );
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0] 
         + u.e[1] * v.e[1] 
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState* rs) {
    while (true) {
        vec3 p = vec3(
            curand_uniform(rs) * 2 - 1, 
            curand_uniform(rs) * 2 - 1, 
            0
        );
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState* rs) {
    while (true) {
        vec3 p = vec3::random(-1, 1, rs);
        float lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState *rs) {
    vec3 on_unit_sphere = random_unit_vector(rs);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = std::fmin(dot(-uv,n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt( fabsf( 1.0 - r_out_perp.length_squared() ) ) * n;
    return r_out_perp + r_out_parallel;
}

#endif