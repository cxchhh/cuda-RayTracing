#pragma once
#include "cuda_runtime.h"
#ifndef HANDLE_ERROR
static void HandleError(cudaError_t err,const char* file,int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif // !HANDLE_ERROR
#define WIDTH 1280
#define HEIGHT 960
#define MAX_DEPTH 2
#define SLEN 15
#define VLEN 0
#define LLEN 5
#define EPSILON 0.001
#define STRIDE 16
#define L_MAX 10
#define GPU 1
#define NIGHT_MODE 1