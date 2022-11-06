#pragma once

#include "Object.hpp"
#include "Vector.hpp"

class Sphere : public Object
{
public:
    __host__ __device__ Sphere()
        : center(Vector3f(0,0,0))
        , radius(0)
        , radius2(0)
    {
        Scenter = center;
        R = radius;
        R2 = radius2;
    }
    __host__ __device__ Sphere(const Vector3f& c, const float& r)
        : center(c)
        , radius(r)
        , radius2(r * r)
    {
        Scenter = center;
        R = radius;
        R2 = radius2;
    }

    Vector3f center;
    float radius, radius2;
};


