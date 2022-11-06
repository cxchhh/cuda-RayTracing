#pragma once
#include "Scene.hpp"
#include "params.hpp"

__host__ __device__ inline float d_clamp(const float& lo, const float& hi, const float& v){
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
    __host__ __device__ hit_payload() :tNear(0), index(0), uv(0), hit_obj(nullptr) {}
    __host__ __device__ hit_payload(float tn, uint32_t in, Vector2f uv, Object* ho) :tNear(tn), index(in), uv(uv), hit_obj(ho) {}
};

class Renderer
{
    public:
        void Render(Scene* scene, Vector3f* framebuffer, Vector3f eye_pos, Vector3f g, Vector3f u);
    private:
};