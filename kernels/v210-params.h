/* Kernel parameter block, shared by kernels/v210.cu and the host code. */

#pragma once

typedef struct
{
  unsigned long long plane[3];  /* device pointers of the non-v210 frame */
  long long stride[3];          /* their strides in bytes */
  unsigned long long v210;      /* device pointer of the v210 frame */
  long long v210_stride;        /* its stride in bytes */
  int width;
  int height;
} V210Params;
