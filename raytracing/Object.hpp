#pragma once

#include "Vector.hpp"
#include "global.hpp"

class Object
{
public:
    __host__ __device__ Object()
        : materialType(DIFFUSE_AND_GLOSSY)
        , ior(1.3)
        , Kd(1.2)
        , Ks(0.6)
        , diffuseColor(0.2)
        , specularExponent(40)
        , type(0)
        , Scenter(0)
        , R(0)
        , R2(0)
    {}

    virtual ~Object() = default;

    __host__ __device__ virtual bool intersect(const Vector3f&, const Vector3f&, float&, uint32_t&, Vector2f&) {
        printf("oh no\n");
        return false;
    }

    __host__ __device__ virtual void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t&, const Vector2f&, Vector3f&, Vector2f&) {
        printf("oh no\n");
    }

    __host__ __device__ virtual Vector3f evalDiffuseColor(const Vector2f&)
    {
        return diffuseColor;
    }

    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vector3f diffuseColor;
    Vector3f Scenter;
    float specularExponent;
    int type;
    float R, R2;
};
