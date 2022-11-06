#include "Scene.hpp"
#include "Sphere.hpp"
#include "Light.hpp"
#include "Renderer.hpp"
#include "global.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"

__host__ __device__ void printV(Vector3f v) {
    printf("(%.3f,%.3f,%.3f)\n", v.x, v.y, v.z);
}

Vector3f eye_pos(0, 0, 0);
Vector3f gaze(0, 0, -1);
Vector3f up(0, 1, 0);
Vector3f gaze_o(0, 0, -1);
Vector3f up_o(0, 1, 0);
Vector3f right_o(1, 0, 0);
float theta=0, phi=0;

void update_pos() {
    /*
    * ca sasb -sacb
    * 0  cb    sb
    * sa -casb cacb
    */
    float a = -theta * M_PI / 180;
    float b = -phi * M_PI / 180;
    Vector3f r1(cos(a), sin(a) * sin(b), -sin(a) * cos(b));
    Vector3f r2(0, cos(b), sin(b));
    Vector3f r3(sin(a), -cos(a) * sin(b), cos(a) * cos(b));
    gaze = Vector3f(dotProduct(r1, gaze_o), dotProduct(r2, gaze_o), dotProduct(r3, gaze_o));
    up = Vector3f(dotProduct(r1, up_o), dotProduct(r2, up_o), dotProduct(r3, up_o));
}


bool keyDown[256];
void getKeyDown(int key) {
    keyDown['q']=keyDown['e']=keyDown['w'] = keyDown['a'] = keyDown['s'] = keyDown['d'] = 0;
    keyDown['f'] = 0;
    if (key == 'w') keyDown['w'] = 1;
    if (key == 'a') keyDown['a'] = 1;
    if (key == 's') keyDown['s'] = 1;
    if (key == 'd') keyDown['d'] = 1;
    if (key == 'q') keyDown['q'] = 1;
    if (key == 'e') keyDown['e'] = 1;
    if (key == 'p') keyDown['p'] = 1;
    if (key == 'f') keyDown['f'] = 1;
}

void move_eye() {
    Vector3f right = crossProduct(gaze, up);
    float speed = 0.5;
    if (keyDown['w']) eye_pos += gaze*speed;
    if (keyDown['s']) eye_pos -= gaze * speed;
    if (keyDown['a']) eye_pos -= right * speed;
    if (keyDown['d']) eye_pos += right * speed;
    if (keyDown['q']) eye_pos += up * speed;
    if (keyDown['e']) eye_pos -= up * speed;
    eye_pos.x = d_clamp(-L_MAX + 0.001, L_MAX - 0.001, eye_pos.x);
    eye_pos.y = d_clamp(-L_MAX + 0.001, L_MAX - 0.001, eye_pos.y);
    eye_pos.z = d_clamp(-L_MAX + 0.001, L_MAX - 0.001, eye_pos.z);
}

__global__ void fbf2img(unsigned char* img, Vector3f* fbf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    int c = blockIdx.z;
    int i = y * WIDTH + x;
    if (c == 0)img[i * 3 + c] = (char)(255 * d_clamp(0, 1, fbf[i].z));
    if (c == 1)img[i * 3 + c] = (char)(255 * d_clamp(0, 1, fbf[i].y));
    if (c == 2)img[i * 3 + c] = (char)(255 * d_clamp(0, 1, fbf[i].x));
}

unsigned char* image;
Vector3f* fb;   //framebuffer

int lmx=-1, lmy=-1,fi=0;
bool canMove = 0;
void on_mouse(int event, int x, int y, int flags, void* userdata){
    switch (event)
    {
        case cv::MouseEventTypes::EVENT_LBUTTONDOWN:
        {
            canMove = 1;
            break;
        }
        case cv::MouseEventTypes::EVENT_LBUTTONUP:
        {
            canMove = 0;
            lmx = x;
            lmy = y;
            break;
        }
    
        case cv::MouseEventTypes::EVENT_MOUSEMOVE:
        {
            if (canMove) {
                theta += (float)(x - lmx) / 10.0;
                phi += (float)(y - lmy) / 10.0;
            }
            lmx = x;
            lmy = y;
            break;
        }

        default:
        {
            break;
        }
    }
}

cudaEvent_t ST, ED;
float  elapsedTime;

void show(const cv::String& winname, cv::InputArray mat) {
    cv::imshow(winname, mat);
}

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline float frnd(float l, float r) {
    int range = (r - l) * 100000;
    return l + ((float)(rnd() % range) / (float)range)*(r-l);
}

// In the main function of the program, we create the scene (create objects and lights).
// We then call the render function().
int main(){
    //create Scene
    Scene* scene;
    HANDLE_ERROR(cudaMallocManaged(&scene, sizeof(Scene)));
    Scene hscene = Scene(WIDTH, HEIGHT);
    HANDLE_ERROR(cudaMemcpy(scene, &hscene, sizeof(Scene), cudaMemcpyHostToDevice));

    //add objects(spheres)
    for (int i = 0; i < SLEN; i++) {
        Sphere sph = Sphere(Vector3f(frnd(-L_MAX, L_MAX), frnd(-L_MAX, L_MAX), frnd(-L_MAX, L_MAX)), frnd(1, 3));
        if (frnd(0, 1) < 0.8) {
            sph.materialType = DIFFUSE_AND_GLOSSY;
            sph.diffuseColor = Vector3f(frnd(0, 1), frnd(0, 1), frnd(0, 1));
        }
        else {
            sph.materialType = REFLECTION_AND_REFRACTION;
            sph.ior = frnd(1,1.5);
        }
        scene->Addo(sph);
    }

    //add lights
    if (NIGHT_MODE) scene->Addl(Light(eye_pos, 1.2, Vector3f(1, 1, -1)));
    scene->Addl(Light(Vector3f(-L_MAX, -L_MAX, L_MAX) * 100, 0.5, Vector3f(1, 1, -1)));
    scene->Addl(Light(Vector3f(L_MAX, L_MAX, L_MAX) * 100, 0.5, Vector3f(-1, -1, -1)));
    scene->Addl(Light(Vector3f(L_MAX, -L_MAX, -L_MAX) * 100, 0.5, Vector3f(-1, 1, 1)));
    scene->Addl(Light(Vector3f(-L_MAX, L_MAX, -L_MAX) * 100, 0.5, Vector3f(1, -1, 1)));


    Renderer r;
    int key;
    int idx=0;
    HANDLE_ERROR(cudaMallocManaged(&image, sizeof(unsigned char) * scene->height * scene->width * 3));
    HANDLE_ERROR(cudaMallocManaged(&fb, sizeof(Vector3f) * scene->height * scene->width));
    auto winName = "Render";
    cv::Mat imageMat;
    HANDLE_ERROR(cudaEventCreate(&ST));
    HANDLE_ERROR(cudaEventCreate(&ED));
    //loop
    while (!keyDown['p']) {
        HANDLE_ERROR(cudaEventRecord(ST, 0));
        idx++;
        key = cv::waitKey(10);
        getKeyDown(key);

        update_pos();
        move_eye();
        if (NIGHT_MODE) {
            if (keyDown['f']) fi ^= 1;
            scene->lights[0].position = eye_pos;
            scene->lights[0].dir = gaze;
            scene->lights[0].intensity = 1.2 * (float)fi;
        }
        
        r.Render(scene,fb,eye_pos,gaze,up);
        dim3 Grid((WIDTH+15)/16, (HEIGHT+15)/16, 3);
        dim3 Block(16, 16);
        fbf2img << <Grid, Block >> > (image, fb);
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        imageMat=cv::Mat(HEIGHT, WIDTH, CV_8UC3, image);

        show(winName, imageMat);
        //break;
        HANDLE_ERROR(cudaEventRecord(ED, 0));
        HANDLE_ERROR(cudaEventSynchronize(ED));
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, ST, ED));
        printf("\rFPS: %3.1f", 1000.0 / elapsedTime);
        if (idx == 1) {
            cv::setMouseCallback(winName, on_mouse, (void*)&imageMat);
        }
    }
    HANDLE_ERROR(cudaEventDestroy(ST));
    HANDLE_ERROR(cudaEventDestroy(ED));
    HANDLE_ERROR(cudaFree(scene));
    HANDLE_ERROR(cudaFree(fb));
    HANDLE_ERROR(cudaFree(image));
    return 0;
}