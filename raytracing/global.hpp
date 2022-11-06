#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

#define M_PI 3.14159265358979323846

#define kInfinity 2147483647.0;

__host__ __device__ inline bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr <= 0 && discr >= 0)
        x0 = x1 = -0.5 * b / a;
    else
    {
        float q = (b > 0) ? -0.5 * (b + sqrtf(discr)) : -0.5 * (b - sqrtf(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) {
        float tmp = x1;
        x1 = x0;
        x0 = tmp;
    }
        
    return true;
}

enum MaterialType
{
    DIFFUSE_AND_GLOSSY,
    REFLECTION_AND_REFRACTION,
    REFLECTION
};

inline float get_random_float()
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.f, 1.f); // distribution in range [1, 6]

    return dist(rng);
}



