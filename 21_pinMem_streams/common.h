#ifndef _COMMON_H_
#define _COMMON_H_

void vecadd_gpu(float *x, float *y, float *z, int N);
void vecadd_gpu_streams(float *x, float *y, float *z, int N);
#endif