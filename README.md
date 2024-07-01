Command to run OpenMP fortran: 
```shell
gfortran -fopenmp file_name -o outputFileName
```
Command to run OpenACC fortran with milan0: 
```shell
nvfortran -fast -Minfo=all -acc=gpu -gpu=cc70 file_name -o outputFileName
```

Command to run OpenACC fortran with cousteau:
```shell
gfortran -fopenacc -fopt-info-optimized file_name -o outputFileName
```

Command to measure the GPU usage over 60 seconds on milan0 and NVIDIA GPU systems (Make sure to copy the entire thing and paste it on a second terminal) 

```shell
#!/bin/bash

# Log GPU usage every second for 60 seconds
for i in {1..60}
do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader >> gpu_usage.log
    sleep 1
done

# Calculate average GPU usage
awk '{ total += $1; count++ } END { print "Average GPU usage:", total/count "%" }' gpu_usage.log
```

Command to measure the GPU usage over 60 seconds on cousteau and AMD GPU systems (Make sure to copy the entire thing and paste it on a second terminal) 
```shell
#!/bin/bash

# Log GPU usage every second for 60 seconds
for i in {1..60}
do
    # Extract the GPU usage percentage using rocm-smi
    rocm-smi --showuse | grep -Eo '[0-9]+%' | sed 's/%//' >> gpu_usage.log
    sleep 1
done

# Calculate average GPU usage
awk '{ total += $1; count++ } END { print "Average GPU usage:", total/count "%" }' gpu_usage.log

```
