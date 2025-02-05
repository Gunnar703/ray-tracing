#ifndef SPHERE_H
#define SPHERE_H

class sphere : public hittable {
    public:
        __device__ sphere(const point3& center, float radius) : center(center), radius(fmaxf(0, radius)) {}

        __device__ bool hit (const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
            vec3 oc = center - r.origin();
            float a = r.direction().length_squared();
            float h = dot(r.direction(), oc);
            float c = oc.length_squared() - radius * radius;
            float d = (h*h - a*c);

            if (d < 0) return false;

            float sqrtd = sqrtf(d);

            // Find the nearest root that lies in the acceptable range
            float root = (h - sqrtd) / a;
            if (root <= ray_tmin || root >= ray_tmax) {
                root = (h + sqrtd) / a;
                if (root <= ray_tmin || root >= ray_tmax) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);

            return true;
        }

    private:
        point3 center;
        float radius;
};

#endif