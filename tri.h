#ifndef TRI_H
#define TRI_H

class tri : public quad {
    public:
        tri(
            const point3& Q, 
            const vec3& u, 
            const vec3& v, 
            shared_ptr<material> mat
        ) : quad(Q, u, v, mat), Q(Q), u(u), v(v), mat(mat) {
            vec3 n = cross(u, v);
            normal = unit_vector(n);
            D = dot(normal, Q);
            w = n / dot(n, n);

            set_bounding_box();
        }

        void set_bounding_box() override {
            // Compute the bounding box of all four vertices
            aabb bbox_diagonal1 = aabb(Q, Q + u + v);
            aabb bbox_diagonal2 = aabb(Q + u, Q + v);
            bbox = aabb(bbox_diagonal1, bbox_diagonal2);
        }

        bool is_interior(double a, double b, hit_record& rec) const override {
            interval unit_interval = interval(0, 1);

            // Given the hit point in plane coordinates, return false if it is outside the
            // primitive, otherwise set the hit record UV coordinates and return true.
            
            bool inside = (a > 0) &&
                          (b > 0) &&
                          (a + b <= 1);
            if (!inside) return false;

            rec.u = a;
            rec.v = b;
            return true;
        }

    private:
        point3 Q;
        vec3 u, v;
        vec3 w;
        shared_ptr<material> mat;
        aabb bbox;
        vec3 normal;
        double D;

};

#endif