Command to run OpenMP fortran: gfortran -fopenmp file_name -o outputFileName

Command to run OpenACC fortran with milan0: nvfortran -fast -Minfo=all -acc=gpu -gpu=cc70 file_name -o outputFileName

Command to run OpenACC fortran with cousteau: gfortran -fopenacc -fopt-info-optimized OpenACC.f90 -o cousteau