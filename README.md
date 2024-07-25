Command to run OpenMP fortran: 
```shell
gfortran -fopenmp file_name -o outputFileName
```
Command to run OpenACC fortran with milan0: 
```shell
nvc -acc  -Minfo=accel -o outputFileName cFileName -lm
```

For C OpenACC on milan0:
```shell
nvcc cFile.c -o outputFileName
```

Command to run OpenACC fortran with cousteau:
```shell
gfortran -fopenacc -fopt-info-optimized file_name -o outputFileName
```

Command to measure the GPU usage over 60 seconds on milan0 and NVIDIA GPU systems (Make sure to copy the entire thing and paste it on a second terminal) 

```shell
#!/bin/bash

# Get the number of GPUs
num_gpus=$(nvidia-smi -L | wc -l)

# Create an array to store usage logs for each GPU
declare -a usage_logs

# Initialize the usage logs
for ((i=0; i<num_gpus; i++)); do
    usage_logs[i]="gpu_usage_$i.log"
    echo -n "" > "${usage_logs[i]}"
done

# Log GPU usage every second for 60 seconds
for i in {1..60}
do
    # Extract GPU usage for each GPU and log it
    for ((j=0; j<num_gpus; j++)); do
        usage=$(nvidia-smi -i $j --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        echo "$usage" >> "${usage_logs[j]}"
    done
    sleep 1
done

# Calculate and display average GPU usage for each GPU
for ((j=0; j<num_gpus; j++)); do
    avg_usage=$(awk '{ total += $1; count++ } END { print total/count }' "${usage_logs[j]}")
    echo "Average GPU usage for GPU $j: $avg_usage%"
done
```

Command to measure the GPU usage over 60 seconds on cousteau and AMD GPU systems (Make sure to copy the entire thing and paste it on a second terminal) 
```shell
#!/bin/bash

# Get the number of GPUs
num_gpus=$(rocm-smi -d | grep -c 'GPU')

# Create an array to store usage logs for each GPU
declare -a usage_logs

# Initialize the usage logs
for ((i=0; i<num_gpus; i++)); do
    usage_logs[i]="gpu_usage_$i.log"
    echo -n "" > "${usage_logs[i]}"
done

# Log GPU usage every second for 60 seconds
for i in {1..60}
do
    # Extract GPU usage for each GPU and log it
    for ((j=0; j<num_gpus; j++)); do
        usage=$(rocm-smi --showuse -d $j | grep -Eo '[0-9]+%' | sed 's/%//')
        echo "$usage" >> "${usage_logs[j]}"
    done
    sleep 1
done

# Calculate and display average GPU usage for each GPU
for ((j=0; j<num_gpus; j++)); do
    avg_usage=$(awk '{ total += $1; count++ } END { print total/count }' "${usage_logs[j]}")
    echo "Average GPU usage for GPU $j: $avg_usage%"
done

```
