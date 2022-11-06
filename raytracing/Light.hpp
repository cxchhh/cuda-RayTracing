#pragma once

#include "Vector.hpp"

class Light
{
public:
    __host__ __device__ Light(const Vector3f& p, const Vector3f& i,const Vector3f& d)
        : position(p)
        , intensity(i)
        , dir(d)
    {
        normalize(d);
    }
    virtual ~Light() = default;
    Vector3f position;
    Vector3f intensity;
    Vector3f dir;
};
