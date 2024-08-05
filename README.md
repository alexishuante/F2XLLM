Command to run OpenMP fortran: 
```shell
gfortran -fopenmp -O3 f90FileName -o outputFileName
```

Command to run OpenACC fortran with milan0: 
```shell
nvc -acc  -Minfo=accel -o outputFileName cFileName -lm
```

Command to run OpenMP C: 
```shell
gfortran -fopenmp fileName.c -o outputFileName
```

For C HIP: 
```shell
hipcc input_file.cpp -o output_file
```

For C OpenACC on milan0:
```shell
 nvcc fileName.c -o outputFileName
```

For C CUDA on milan0:
```shell
nvcc -arch=sm_60 cFileName -o outputFileName
```

Command to run 
