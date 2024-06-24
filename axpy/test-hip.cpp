#include "hip/hip_runtime.h"
 
int main()
{
  int device_count;
  hipGetDeviceCount(&device_count);
  printf( "%d GPUs available\n", device_count );
}
