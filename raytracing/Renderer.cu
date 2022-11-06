#include "Vector.hpp"
#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include <opencv2/opencv.hpp>


__host__ __device__ void printVV(Vector3f v) {
    printf("(%.3f,%.3f,%.3f)\n", v.x, v.y, v.z);
}
__host__ __device__ bool sphere_intersect(Sphere* sp, const Vector3f& orig, const Vector3f& dir, float& tnear) {
    Vector3f L = orig - sp->Scenter;
    float a = dotProduct(dir, dir);
    float b = 2 * dotProduct(dir, L);
    float c = dotProduct(L, L) - sp->R2;
    float t0, t1;
    if (!solveQuadratic(a, b, c, t0, t1))
        return false;
    if (t0 < 0)
        t0 = t1;
    if (t0 < 0)
        return false;
    tnear = t0;
    return true;
}

__host__ __device__ inline float deg2rad(const float& deg){
    return deg * M_PI / 180.0;
}
__host__ __device__ float d_max(const float& da, const float& db) {
    if (da > db) return da;
    return db;
}
__host__ __device__ Vector3f reflect(const Vector3f& I, const Vector3f& N){
    return I - 2 * dotProduct(I, N) * N;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and negate the normal N
// [/comment]
__host__ __device__ Vector3f refract(const Vector3f& I, const Vector3f& N, const float& ior)
{
    float cosi = d_clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    Vector3f n = N;
    if (cosi < 0) { cosi = -cosi; }
    else {
        float t = etai;
        etai = etat;
        etat = t;
        n = -N; 
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the material refractive index
// [/comment]
__host__ __device__ float fresnel(const Vector3f& I, const Vector3f& N, const float& ior) {
    float cosi = d_clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    if (cosi > 0) {
        float t = etai;
        etai = etat;
        etat = t;
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(d_max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        return 1;
    }
    else {
        float cost = sqrtf(d_max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

__host__ __device__ Vector3f getBgColor(const Vector3f& orig, const Vector3f& dir) {
    float tx = kInfinity;
    float ty = kInfinity;
    float tz = kInfinity;
    int fx=0, fy=0, fz=0;
    if (!(dir.x >= 0 && dir.x <= 0)) {
        tx= (L_MAX - orig.x) / dir.x;
        if (tx < 0) {
            tx = (-L_MAX - orig.x) / dir.x;
            fx = 1;
        }
    }

    if (!(dir.y >= 0 && dir.y <= 0)) {
        ty = (L_MAX - orig.y) / dir.y;
        if (ty < 0) {
            ty = (-L_MAX - orig.y) / dir.y;
            fy = 1;
        }
    }

    if (!(dir.z >= 0 && dir.z <= 0)) {
        tz = (L_MAX - orig.z) / dir.z;
        if (tz < 0) {
            tz = (-L_MAX - orig.z) / dir.z;
            fz = 1;
        }
    }

    if (tx < ty && tx < tz) {
        if (fx) return Vector3f(0.3, 0.3, 0.8);
        else return Vector3f(0.3, 0.8, 0.3);
    }
    
    if (ty < tx && ty < tz) {
        if (!fy) return Vector3f(0.8, 0.8, 0.3);
        else return Vector3f(0.9, 0.9, 0.9);
    }
    if (tz < ty && tz < tx) {
        if (fz) return Vector3f(0.8, 0.3, 0.3);
        else return Vector3f(0.8, 0.5, 0.3);
    }
}

// [comment]
// Returns true if the ray intersects an object, false otherwise.
//
// \param orig is the ray origin
// \param dir is the ray direction
// \param objects is the list of objects the scene contains
// \param[out] tNear contains the distance to the cloesest intersected object.
// \param[out] index stores the index of the intersect triangle if the interesected object is a mesh.
// \param[out] uv stores the u and v barycentric coordinates of the intersected point
// \param[out] *hitObject stores the pointer to the intersected object (used to retrieve material information, etc.)
// \param isShadowRay is it a shadow ray. We can return from the function sooner as soon as we have found a hit.
// [/comment]
__host__ __device__ hit_payload d_trace(const Vector3f& orig, const Vector3f& dir, Object* objects) {
    float tNear = kInfinity;
    hit_payload payload=hit_payload();
    Object* object;
    for (int j=0;j<SLEN+VLEN;++j)
    {
        object = objects + j;
        float tNearK = kInfinity;
        uint32_t indexK=0;
        if (object->type==0 && sphere_intersect((Sphere*)object,orig, dir, tNearK) && tNearK < tNear)
        {   
            payload.hit_obj = object;
            payload.tNear = tNearK;
            payload.index = indexK;
            tNear = tNearK;
        }
    }
    return payload;
}

// [comment]
// Implementation of the Whitted-style light transport algorithm (E [S*] (D|G) L)
//
// This function is the function that compute the color at the intersection point
// of a ray defined by a position and a direction. Note that thus function is recursive (it calls itself).
//
// If the material of the intersected object is either reflective or reflective and refractive,
// then we compute the reflection/refraction direction and cast two new rays into the scene
// by calling the castRay() function recursively. When the surface is transparent, we mix
// the reflection and refraction color using the result of the fresnel equations (it computes
// the amount of reflection and refraction depending on the surface normal, incident view direction
// and surface refractive index).
//
// If the surface is diffuse/glossy we use the Phong illumation model to compute the color
// at the intersection point.
// [/comment]
__host__ __device__ Vector3f castRay(const Vector3f& orig, const Vector3f& dir, Scene* scene, int depth) {
    if (depth > MAX_DEPTH) return Vector3f(0, 0, 0);

    Vector3f hitColor = NIGHT_MODE ? Vector3f(0, 0, 0) : getBgColor(orig, dir);
    hit_payload payload = d_trace(orig, dir, scene->objects);
    
    if (payload.hit_obj)
    {
        
        Vector3f hitPoint = orig + dir * payload.tNear;
        Vector3f N; // normal
        Vector2f st; // st coordinates
        if (payload.hit_obj->type == 0) N = normalize(hitPoint - payload.hit_obj->Scenter);
        
        switch (payload.hit_obj->materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vector3f reflectionDirection = normalize(reflect(dir, N));
                Vector3f refractionDirection = normalize(refract(dir, N, payload.hit_obj->ior));
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint - N * EPSILON :
                    hitPoint + N * EPSILON;
                Vector3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                    hitPoint - N * EPSILON :
                    hitPoint + N * EPSILON;
                Vector3f reflectionColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1);
                Vector3f refractionColor = castRay(refractionRayOrig, refractionDirection, scene, depth + 1);
                float kr = fresnel(dir, N, payload.hit_obj->ior);
                hitColor = reflectionColor * kr + refractionColor * (1 - kr);
                break;
            }
            case REFLECTION:
            {
                float kr = fresnel(dir, N, payload.hit_obj->ior);
                Vector3f reflectionDirection = reflect(dir, N);
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint + N * EPSILON :
                    hitPoint - N * EPSILON;
                hitColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1) * kr;
                break;
            }
            default:
            {
                // [comment]
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vector3f lightAmt = 0, specularColor = 0;
                Vector3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
                    hitPoint + N * scene->epsilon :
                    hitPoint - N * scene->epsilon;
                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law
                // [/comment]
                for (int i = 0; i < LLEN; i++) {
                    Light* light = scene->lights + i;
                    Vector3f lightDir = light->position - hitPoint;
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir);
                    lightDir = normalize(lightDir);
                    Vector3f LI = light->intensity * (NIGHT_MODE ? powf(d_clamp(0.f, 1.f, dotProduct(light->dir, -lightDir)), 5) : 1);
                    float LdotN = d_max(0.f, dotProduct(lightDir, N));
                    // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
                    auto shadow_res = d_trace(shadowPointOrig, lightDir, scene->objects);
                    bool inShadow = shadow_res.hit_obj && (shadow_res.tNear * shadow_res.tNear < lightDistance2);

                    lightAmt += inShadow ? 0 : LI * LdotN;
                    Vector3f reflectionDirection = reflect(-lightDir, N);

                    specularColor += powf(d_max(0.f, -dotProduct(reflectionDirection, dir)), payload.hit_obj->specularExponent) * LI;
                }

                hitColor = lightAmt * (payload.hit_obj->type == 0 ? payload.hit_obj->diffuseColor : printf("tri")) * payload.hit_obj->Kd + specularColor * payload.hit_obj->Ks;
                break;
            }
        }
        
    }
    return hitColor;
}


__global__ void RayTracing_kernel(Vector3f* fb,Scene* sc,Vector3f eye_pos,Vector3f g, Vector3f u, Vector3f r, float w,float h,float xg,float yg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= WIDTH || j >= HEIGHT) return;
    float x = -w / 2 + xg * i;
    float y = h / 2 - yg * j;
    Vector3f dir = x * r + y * u + g;
    dir=normalize(dir);
    fb[j * WIDTH + i] = castRay(eye_pos, dir, sc, 0);
}

// [comment]
// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void Renderer::Render(Scene* scene,Vector3f* framebuffer,Vector3f eye_pos,Vector3f g,Vector3f u) {
    float scale = std::tan(deg2rad(scene->fov * 0.5f));
    float imageAspectRatio = WIDTH / (float)HEIGHT;
    float height = scale * 2;
    float width = height * imageAspectRatio;
    float x_gap = width / (WIDTH - 1);
    float y_gap = height / (HEIGHT - 1);
    Vector3f r = crossProduct(g, u);
    int m = 0;
    if (GPU) {  //using GPU
        RayTracing_kernel << <dim3((WIDTH+STRIDE-1)/STRIDE,(HEIGHT + STRIDE - 1)/STRIDE), dim3(STRIDE, STRIDE) >> > (framebuffer, scene, eye_pos, g, u, r, width, height, x_gap, y_gap);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
    else        //using CPU
    for (int j = 0; j < HEIGHT; ++j)
    {
        for (int i = 0; i < WIDTH; ++i)
        {
            float x;
            float y; 
            x = -width / 2 + x_gap * i;
            y = height / 2 - y_gap * j;
            Vector3f dir = x*r+y*u+g;
            dir=normalize(dir);
            framebuffer[m++] = castRay(eye_pos, dir, scene, 0);
        }
    }
    
}


