Command to run OpenMP fortran: \
```shell
gfortran -fopenmp file_name -o outputFileName
```
Command to run OpenACC fortran with milan0: \
`nvfortran -fast -Minfo=all -acc=gpu -gpu=cc70 file_name -o outputFileName`

Command to run OpenACC fortran with cousteau: \
`gfortran -fopenacc -fopt-info-optimized file_name -o outputFileName`
