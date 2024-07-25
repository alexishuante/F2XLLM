#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <time.h>

#include <cuda_runtime.h>

 

#define PI 3.1415926535897931

#define SQRPI2 (2.0 * pow(PI, -0.5))

#define TOBOHRS 1.889725987722

#define DTOL 1.0e-12

#define RCUT 1.0e-12

#define NGAUSS 3 // Example size, change as needed

 

double get_time() {

    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec * 1e-9;

}

 

// // Your data here

__constant__ double* d_xpnt;

__constant__ double* d_coef;

__device__ double (*d_geom)[3];

__device__ double* d_schwarz;

__device__ double* d_dens;

__device__ double* d_fock;

 

__global__ void eriKernel(int nnnn, int natom, double dtol, double sqrpi2, double rcut,

                          double* xpnt, double* coef, double (*geom)[3],

                          double* schwarz, double* dens, double* fock) {

    size_t ijkl = blockIdx.x * blockDim.x + threadIdx.x + 1;

 

    if (ijkl > nnnn) return;

 

    // Decompose triangular ijkl index into ij >= kl

    int ij = (int)sqrt(2.0 * ijkl);

    int n = (ij * ij + ij) / 2;

    while (n < ijkl) {

        ij++;

        n = (ij * ij + ij) / 2;

    }

    int kl = ijkl - (ij * ij - ij) / 2;

 

    if (ij > natom * (natom + 1) / 2 || kl > natom * (natom + 1) / 2) return;

 

    if (schwarz[ij - 1] * schwarz[kl - 1] > dtol) {

        // Decompose triangular ij index into i >= j

        int i = (int)sqrt(2.0 * ij);

        n = (i * i + i) / 2;

        while (n < ij) {

            i++;

            n = (i * i + i) / 2;

        }

        int j = ij - (i * i - i) / 2;

        // Decompose triangular kl index into k >= l

        int k = (int)sqrt(2.0 * kl);

        n = (k * k + k) / 2;

        while (n < kl) {

            k++;

            n = (k * k + k) / 2;

        }

        int l = kl - (k * k - k) / 2;

 

        if (i > natom || j > natom || k > natom || l > natom) return;

 

        double eri = 0.0;

 

        // Loop structure remains the same with index changes

        for (int ib = 0; ib < NGAUSS; ib++) {

            for (int jb = 0; jb < NGAUSS; jb++) {

                double aij = 1.0 / (xpnt[ib] + xpnt[jb]);

                double dij = coef[ib] * coef[jb] * exp(-xpnt[ib] * xpnt[jb] * aij *

                        (pow(geom[i - 1][0] - geom[j - 1][0], 2) +

                         pow(geom[i - 1][1] - geom[j - 1][1], 2) +

                         pow(geom[i - 1][2] - geom[j - 1][2], 2))) * pow(aij, 1.5);

 

                if (abs(dij) > dtol) {

                    double xij = aij * (xpnt[ib] * geom[i - 1][0] + xpnt[jb] * geom[j - 1][0]);

                    double yij = aij * (xpnt[ib] * geom[i - 1][1] + xpnt[jb] * geom[j - 1][1]);

                    double zij = aij * (xpnt[ib] * geom[i - 1][2] + xpnt[jb] * geom[j - 1][2]);

 

                    for (int kb = 0; kb < NGAUSS; kb++) {

                        for (int lb = 0; lb < NGAUSS; lb++) {

                            double akl = 1.0 / (xpnt[kb] + xpnt[lb]);

                            double dkl = dij * coef[kb] * coef[lb] * exp(-xpnt[kb] * xpnt[lb] * akl *

                                    (pow(geom[k - 1][0] - geom[l - 1][0], 2) +

                                     pow(geom[k - 1][1] - geom[l - 1][1], 2) +

                                     pow(geom[k - 1][2] - geom[l - 1][2], 2))) * pow(akl, 1.5);

 

                            if (abs(dkl) > dtol) {

                                double aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) / (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);

                                double tt = aijkl * (

                                        pow(xij - akl * (xpnt[kb] * geom[k - 1][0] + xpnt[lb] * geom[l - 1][0]), 2) +

                                        pow(yij - akl * (xpnt[kb] * geom[k - 1][1] + xpnt[lb] * geom[l - 1][1]), 2) +

                                        pow(zij - akl * (xpnt[kb] * geom[k - 1][2] + xpnt[lb] * geom[l - 1][2]), 2));

 

                                double f0t = SQRPI2;

                                if (tt > rcut) f0t = (pow(tt, -0.5)) * erf(sqrt(tt));

                                eri += dkl * f0t * sqrt(aijkl);

                            }

                        }

                    }

                }

            }

        }

 

        if (i == j) eri *= 0.5;

        if (k == l) eri *= 0.5;

        if (i == k && j == l) eri *= 0.5;

 

        atomicAdd(&fock[(i-1) * natom + (j-1)], dens[(k-1) * natom + (l-1)] * eri * 4.0);

        atomicAdd(&fock[(k-1) * natom + (l-1)], dens[(i-1) * natom + (j-1)] * eri * 4.0);

        atomicAdd(&fock[(i-1) * natom + (k-1)], -dens[(j-1) * natom + (l-1)] * eri);

        atomicAdd(&fock[(i-1) * natom + (l-1)], -dens[(j-1) * natom + (k-1)] * eri);

        atomicAdd(&fock[(j-1) * natom + (k-1)], -dens[(i-1) * natom + (l-1)] * eri);

        atomicAdd(&fock[(j-1) * natom + (l-1)], -dens[(i-1) * natom + (k-1)] * eri);

    }

}

 

void hartreeFockOperation(int nnnn, int ngauss, int natom, double dtol, double rcut, double *schwarzHost, double *xpntHost, double *coefHost, double (*geomHost)[3], double *densHost, double *fockHost, double *run_time) {

    // Copy data to device 

 

    double *d_xpnt, *d_coef, *d_schwarz, *d_dens, *d_fock;

    double (*d_geom)[3];

    size_t geomSize = natom * 3 * sizeof(double);

    size_t schwarzSize = (natom * natom + natom) / 2 * sizeof(double);

    size_t densSize = pow(natom, 2) * sizeof(double);

    size_t fockSize = pow(natom, 2) * sizeof(double);

  

    // Allocate device memory

    cudaMalloc((void**)&d_xpnt, sizeof(double) * ngauss);

    cudaMalloc((void**)&d_coef, sizeof(double) * ngauss);

    cudaMalloc((void**)&d_geom, geomSize);

    cudaMalloc((void**)&d_schwarz, schwarzSize);

    cudaMalloc((void**)&d_dens, densSize);

    cudaMalloc((void**)&d_fock, fockSize);

 

 

    cudaMemcpy(d_xpnt, xpntHost, sizeof(double) * ngauss, cudaMemcpyHostToDevice);

    cudaMemcpy(d_coef, coefHost, sizeof(double) * ngauss, cudaMemcpyHostToDevice);

    cudaMemcpy(d_geom, geomHost, geomSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_schwarz, schwarzHost, schwarzSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_dens, densHost, densSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_fock, fockHost, fockSize, cudaMemcpyHostToDevice);

 

    // Launch kernel

    int blockSize = 128; // number of threads per block

    int numBlocks = (nnnn + blockSize - 1) / blockSize;

   

    double start_time = get_time();

    eriKernel<<<numBlocks, blockSize>>>(nnnn, natom, DTOL, SQRPI2, rcut, d_xpnt, d_coef, d_geom, d_schwarz, d_dens, d_fock);

    cudaDeviceSynchronize();

    double end_time = get_time();

    *run_time = end_time - start_time;

 

    // Copy result back to host

 

    cudaMemcpy(fockHost, d_fock, fockSize, cudaMemcpyDeviceToHost);

 

    //     // Free device memory

    cudaFree(d_xpnt);

    cudaFree(d_coef);

    cudaFree(d_geom);

    cudaFree(d_schwarz);

    cudaFree(d_dens);

    cudaFree(d_fock);

}

 

 

 

 

void ssss(int i, int j, int k, int l, int ngauss, double* xpnt, double* coef, double (*geom)[3], double* eri) {

    int ib, jb, kb, lb;

    double aij, dij, xij, yij, zij, akl, dkl, aijkl, tt, f0t;

 

 

    *eri = 0.0;

 

 

    for (ib = 0; ib < ngauss; ib++) {

        for (jb = 0; jb < ngauss; jb++) {

            aij = 1.0 / (xpnt[ib] + xpnt[jb]);

            dij = coef[ib] * coef[jb] * exp(-xpnt[ib] * xpnt[jb] * aij *

                (pow(geom[i][0] - geom[j][0], 2) +

                 pow(geom[i][1] - geom[j][1], 2) +

                 pow(geom[i][2] - geom[j][2], 2))) * pow(aij, 1.5);

 

 

            if (fabs(dij) > DTOL) {

                xij = aij * (xpnt[ib] * geom[i][0] + xpnt[jb] * geom[j][0]);

                yij = aij * (xpnt[ib] * geom[i][1] + xpnt[jb] * geom[j][1]);

                zij = aij * (xpnt[ib] * geom[i][2] + xpnt[jb] * geom[j][2]);

 

 

                for (kb = 0; kb < ngauss; kb++) {

                    for (lb = 0; lb < ngauss; lb++) {

                        akl = 1.0 / (xpnt[kb] + xpnt[lb]);

                        dkl = dij * coef[kb] * coef[lb] * exp(-xpnt[kb] * xpnt[lb] * akl *

                            (pow(geom[k][0] - geom[l][0], 2) +

                             pow(geom[k][1] - geom[l][1], 2) +

                             pow(geom[k][2] - geom[l][2], 2))) * pow(akl, 1.5);

 

 

                        if (fabs(dkl) > DTOL) {

                            aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /

                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);

                          

                            tt = aijkl * (

                                pow(xij - akl * (xpnt[kb] * geom[k][0] + xpnt[lb] * geom[l][0]), 2) +

                                pow(yij - akl * (xpnt[kb] * geom[k][1] + xpnt[lb] * geom[l][1]), 2) +

                                pow(zij - akl * (xpnt[kb] * geom[k][2] + xpnt[lb] * geom[l][2]), 2));

 

 

                            f0t = SQRPI2;

                            if (tt > RCUT) {

                                f0t = pow(tt, -0.5) * erf(sqrt(tt));

                            }

 

 

                            *eri += dkl * f0t * sqrt(aijkl);

                        }

                    }

                }

            }

        }

    }

}

 

 

int main() {

    int natom = 8192, ngauss = 3;

    double erep = 0.0;

    double txpnt[3] = {6.3624214, 1.1589230, 0.3136498};

    double tcoef[3] = {0.154328967295, 0.535328142282, 0.444634542185};

 

    double *xpntHost = (double*)malloc(ngauss * sizeof(double));

    double *coefHost = (double*)malloc(ngauss * sizeof(double));

    double (*geomHost)[3] = (double(*)[3])malloc(natom * 3 * sizeof(double));

    double *fockHost = (double*)malloc(natom * natom * sizeof(double));

    double *densHost = (double*)malloc(natom * natom * sizeof(double));

 

    // double tgeom[4][3] = {{0.0, 0.0, 0.0},

    //                     {0.05, 0.0, 1.0},

    //                     {0.1, 1.0, 0.0},

    //                     {1.0, 0.2, 0.0}};

 

    // Initialize arrays

    for (int i = 0; i < ngauss; i++) {

        xpntHost[i] = txpnt[i];

        coefHost[i] = tcoef[i];

    }

 

    srand(12345);  // Set seed for reproducibility

    for (int i = 0; i < natom; i++) {

        for (int j = 0; j < 3; j++) {

            int random_int = rand() % 181;

            double value = (double)random_int / 10.0;

            geomHost[i][j] = value * TOBOHRS;

            // printf("%f ", geomHost[i][j]);

        }

    }

 

    // for (int i = 0; i < natom; i++) {

    //     for (int j = 0; j < 3; j++) {

    //         geomHost[i][j] = tgeom[i][j] * TOBOHRS;

    //     }

    // }

 

 

 

 

    // Build density matrix

    for (int i = 0; i < natom; i++) {

        for (int j = 0; j < natom; j++) {

            densHost[i * natom + j] = 0.1;

            if (i == j) densHost[i * natom + j] = 1.0;

        }

    }

 

 

    // Initialize Fock matrix

    for (int i = 0; i < natom * natom; i++) {

        fockHost[i] = 0.0;

    }

 

 

    // Normalize primitive GTO weights

    for (int i = 0; i < ngauss; i++) {

        coefHost[i] *= pow(2.0 * xpntHost[i], 0.75);

    }

 

 

    // Compute Schwarz Inequality factors

    int nn = (natom * natom + natom) / 2;

    double *schwarzHost = (double*)malloc(nn * sizeof(double));

 

 

    int ij = 0;

    for (int i = 0; i < natom; i++) {

        for (int j = 0; j <= i; j++) {

            double eri;

            ssss(i, j, i, j, ngauss, xpntHost, coefHost, geomHost, &eri);

            schwarzHost[ij] = sqrt(fabs(eri));

            ij++;

        }

    }

 

    // Compute Hartree-Fock

    int nnnn = (nn * nn + nn) / 2;

    // printf("nnnn: %d\n", nnnn);

 

    // Warmup run

    double run_time = 0.0;

    printf("Performing warmup run...\n");

    hartreeFockOperation(nnnn, ngauss, natom, DTOL, RCUT, schwarzHost, xpntHost, coefHost, geomHost, densHost, fockHost, &run_time);

 

    // Timing runs

    double total_time = 0.0;

    int nruns = 10;

    printf("Performing %d timed runs:\n", nruns);

 

    for (int run = 0; run < nruns; run++) {

        // Reset fock matrix

        for (int i = 0; i < natom * natom; i++) {

            fockHost[i] = 0.0;

        }

        run_time = 0.0;

        hartreeFockOperation(nnnn, ngauss, natom, DTOL, RCUT, schwarzHost, xpntHost, coefHost, geomHost, densHost, fockHost, &run_time);

        total_time += run_time;

 

        // Compute 2e- energy

        erep = 0.0;

        for (int i = 0; i < natom; i++) {

            for (int j = 0; j < natom; j++) {

                erep += fockHost[i * natom + j] * densHost[i * natom + j];

 

            }

        }

 

        printf("Run %d: Time = %.6f seconds, 2e- energy = %f\n", run + 1, run_time, erep * 0.5);

    }

    // Calculate and print average time

    double avg_time = total_time / nruns;

    printf("\nAverage time per run: %.6f seconds\n", avg_time);

 

    // Free memory

    free(xpntHost);

    free(coefHost);

    free(geomHost);

    free(fockHost);

    free(densHost);

    free(schwarzHost);

 

 

    return 0;

}
