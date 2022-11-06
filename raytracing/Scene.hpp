#pragma once

#include <vector>
#include <memory>
#include "Vector.hpp"
#include "Object.hpp"
#include "Light.hpp"
#include "params.hpp"

class Scene
{
public:
    // setting up options
    int width = WIDTH;
    int height = HEIGHT;
    float fov = 90;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 3;
    float epsilon = EPSILON;
    // creating the scene (adding objects and lights)
    Object* objects;
    Light* lights;
    int llen = 0;
    int olen = 0;

    Scene(int w, int h) : width(w), height(h){
        HANDLE_ERROR(cudaMallocManaged(&lights, sizeof(Light) * LLEN));
        HANDLE_ERROR(cudaMallocManaged(&objects, sizeof(Object) * (SLEN + VLEN)));
    }
    void Addo(Object object) {
        objects[olen] = object;
        olen++;
        objects[olen] = Object();
    }
    void Addl(Light light) {
        lights[llen] = light;
        llen++;
    }
};