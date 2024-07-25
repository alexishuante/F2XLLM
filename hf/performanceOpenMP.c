#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>
#include <time.h>
#include <omp.h>

#define PI 3.1415926535897931
#define SQRPI2 (2.0 * pow(PI, -0.5))
#define TOBOHRS 1.889725987722

#define DTOL 1.0e-12
#define RCUT 1.0e-12

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void hartree_fock(double *fock, const double *dens, const double *schwarz, const double *geom, const double *xpnt, const double *coef, int ngauss, int nnnn, int natom, double dtol, double rcut, int nn) {

    int ijkl, ij, i, j, kl, k, l, ib, jb, kb, lb, n;
    double xij, yij, zij, aij, dij, akl, dkl, aijkl, tt, f0t, eri;

    #pragma omp parallel for private(ijkl, ij, i, j, kl, k, l, ib, jb, kb, lb, xij, yij, zij, aij, dij, akl, dkl, aijkl, tt, f0t, eri, n)
    for (ijkl = 1; ijkl <= nnnn; ijkl++) {
        // decompose triangular ijkl index into ij>=kl
        ij = (int)sqrt(2.0 * ijkl);
        n = (ij * ij + ij) / 2;
        while (n < ijkl) {
            ij++;
            n = (ij * ij + ij) / 2;
        }
        kl = ijkl - (ij * ij - ij) / 2;

        if (schwarz[ij - 1] * schwarz[kl - 1] > dtol) {
            // decompose triangular ij index into i>=j
            i = (int)sqrt(2.0 * ij);
            n = (i * i + i) / 2;
            while (n < ij) {
                i++;
                n = (i * i + i) / 2;
            }
            j = ij - (i * i - i) / 2;

            // decompose triangular kl index into k>=l
            k = (int)sqrt(2.0 * kl);
            n = (k * k + k) / 2;
            while (n < kl) {
                k++;
                n = (k * k + k) / 2;
            }
            l = kl - (k * k - k) / 2;

            eri = 0.0;
            for (jb = 0; jb < ngauss; jb++) {
                for (ib = 0; ib < ngauss; ib++) {
                    aij = 1.0 / (xpnt[ib] + xpnt[jb]);
                    dij = coef[ib] * coef[jb] * exp(-xpnt[ib] * xpnt[jb] * aij * 
                            ((geom[3 * (i-1)] - geom[3 * (j-1)]) * (geom[3 * (i-1)] - geom[3 * (j-1)]) +
                             (geom[3 * (i-1) + 1] - geom[3 * (j-1) + 1]) * (geom[3 * (i-1) + 1] - geom[3 * (j-1) + 1]) +
                             (geom[3 * (i-1) + 2] - geom[3 * (j-1) + 2]) * (geom[3 * (i-1) + 2] - geom[3 * (j-1) + 2]))) * pow(aij, 1.5);
                    if (fabs(dij) > dtol) {
                        xij = aij * (xpnt[ib] * geom[3 * (i-1)] + xpnt[jb] * geom[3 * (j-1)]);
                        yij = aij * (xpnt[ib] * geom[3 * (i-1) + 1] + xpnt[jb] * geom[3 * (j-1) + 1]);
                        zij = aij * (xpnt[ib] * geom[3 * (i-1) + 2] + xpnt[jb] * geom[3 * (j-1) + 2]);
                        
                        for (lb = 0; lb < ngauss; lb++) {
                            for (kb = 0; kb < ngauss; kb++) {
                                akl = 1.0 / (xpnt[kb] + xpnt[lb]);
                                dkl = dij * coef[kb] * coef[lb] * exp(-xpnt[kb] * xpnt[lb] * akl *
                                    ((geom[3 * (k-1)] - geom[3 * (l-1)]) * (geom[3 * (k-1)] - geom[3 * (l-1)]) +
                                     (geom[3 * (k-1) + 1] - geom[3 * (l-1) + 1]) * (geom[3 * (k-1) + 1] - geom[3 * (l-1) + 1]) +
                                     (geom[3 * (k-1) + 2] - geom[3 * (l-1) + 2]) * (geom[3 * (k-1) + 2] - geom[3 * (l-1) + 2]))) * pow(akl, 1.5);

                                if (fabs(dkl) > dtol) {
                                    aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) / 
                                            (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
                                    tt = aijkl * ((xij - akl * (xpnt[kb] * geom[3 * (k-1)] + xpnt[lb] * geom[3 * (l-1)])) * (xij - akl * (xpnt[kb] * geom[3 * (k-1)] + xpnt[lb] * geom[3 * (l-1)])) +
                                                  (yij - akl * (xpnt[kb] * geom[3 * (k-1) + 1] + xpnt[lb] * geom[3 * (l-1) + 1])) * (yij - akl * (xpnt[kb] * geom[3 * (k-1) + 1] + xpnt[lb] * geom[3 * (l-1) + 1])) +
                                                  (zij - akl * (xpnt[kb] * geom[3 * (k-1) + 2] + xpnt[lb] * geom[3 * (l-1) + 2])) * (zij - akl * (xpnt[kb] * geom[3 * (k-1) + 2] + xpnt[lb] * geom[3 * (l-1) + 2])));
                                
                                f0t = SQRPI2;
                                if (tt > rcut) f0t = pow(tt, -0.5) * erf(sqrt(tt));
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

            #pragma omp atomic
            fock[(i-1) * natom + (j-1)] += dens[(k-1) * natom + (l-1)] * eri * 4.0;
            #pragma omp atomic
            fock[(k-1) * natom + (l-1)] += dens[(i-1) * natom + (j-1)] * eri * 4.0;
            #pragma omp atomic
            fock[(i-1) * natom + (k-1)] -= dens[(j-1) * natom + (l-1)] * eri;
            #pragma omp atomic
            fock[(i-1) * natom + (l-1)] -= dens[(j-1) * natom + (k-1)] * eri;
            #pragma omp atomic
            fock[(j-1) * natom + (k-1)] -= dens[(i-1) * natom + (l-1)] * eri;
            #pragma omp atomic
            fock[(j-1) * natom + (l-1)] -= dens[(i-1) * natom + (k-1)] * eri;
        }
    }
}

void ssss(int i, int j, int k, int l, int ngauss, double* xpnt, double* coef, double* geom, double* eri) {
    int ib, jb, kb, lb;
    double aij, dij, xij, yij, zij, akl, dkl, aijkl, tt, f0t;

    *eri = 0.0;

    for (ib = 0; ib < ngauss; ib++) {
        for (jb = 0; jb < ngauss; jb++) {
            aij = 1.0 / (xpnt[ib] + xpnt[jb]);
            dij = coef[ib] * coef[jb] * exp(-xpnt[ib] * xpnt[jb] * aij *
                ((geom[3*i] - geom[3*j]) * (geom[3*i] - geom[3*j]) +
                 (geom[3*i+1] - geom[3*j+1]) * (geom[3*i+1] - geom[3*j+1]) +
                 (geom[3*i+2] - geom[3*j+2]) * (geom[3*i+2] - geom[3*j+2]))) * pow(aij, 1.5);

            if (fabs(dij) > DTOL) {
                xij = aij * (xpnt[ib] * geom[3*i] + xpnt[jb] * geom[3*j]);
                yij = aij * (xpnt[ib] * geom[3*i+1] + xpnt[jb] * geom[3*j+1]);
                zij = aij * (xpnt[ib] * geom[3*i+2] + xpnt[jb] * geom[3*j+2]);

                for (kb = 0; kb < ngauss; kb++) {
                    for (lb = 0; lb < ngauss; lb++) {
                        akl = 1.0 / (xpnt[kb] + xpnt[lb]);
                        dkl = dij * coef[kb] * coef[lb] * exp(-xpnt[kb] * xpnt[lb] * akl *
                            ((geom[3*k] - geom[3*l]) * (geom[3*k] - geom[3*l]) +
                             (geom[3*k+1] - geom[3*l+1]) * (geom[3*k+1] - geom[3*l+1]) +
                             (geom[3*k+2] - geom[3*l+2]) * (geom[3*k+2] - geom[3*l+2]))) * pow(akl, 1.5);

                        if (fabs(dkl) > DTOL) {
                            aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb]) /
                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
                           
                            tt = aijkl * (
                                (xij - akl * (xpnt[kb] * geom[3*k] + xpnt[lb] * geom[3*l])) * (xij - akl * (xpnt[kb] * geom[3*k] + xpnt[lb] * geom[3*l])) +
                                (yij - akl * (xpnt[kb] * geom[3*k+1] + xpnt[lb] * geom[3*l+1])) * (yij - akl * (xpnt[kb] * geom[3*k+1] + xpnt[lb] * geom[3*l+1])) +
                                (zij - akl * (xpnt[kb] * geom[3*k+2] + xpnt[lb] * geom[3*l+2])) * (zij - akl * (xpnt[kb] * geom[3*k+2] + xpnt[lb] * geom[3*l+2])));

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
    // int natom = 32768, ngauss = 3;
    int natom = 2048, ngauss = 3;
    double erep = 0.0;
    double txpnt[3] = {6.3624214, 1.1589230, 0.3136498};
    double tcoef[3] = {0.154328967295, 0.535328142282, 0.4446345421855};

    // Allocate memory
    double *xpnt = (double*)malloc(ngauss * sizeof(double));
    double *coef = (double*)malloc(ngauss * sizeof(double));
    double *geom = (double*)malloc(natom * 3 * sizeof(double));
    double *fock = (double*)malloc(natom * natom * sizeof(double));
    double *dens = (double*)malloc(natom * natom * sizeof(double));

    // Initialize arrays
    for (int i = 0; i < ngauss; i++) {
        xpnt[i] = txpnt[i];
        coef[i] = tcoef[i];
    }

    //     double geom_values[] = {
    //     0.0, 0.0, 0.0,
    //     0.05, 0.0, 1.0,
    //     0.1, 1.0, 0.0,
    //     1.0, 0.2, 0.0
    // };

    // for (int i = 0; i < natom * 3; i++) {
    //     geom[i] = geom_values[i] * TOBOHRS;
    //     printf("geom[%d] = %.6f\n", i, geom[i]);
    // }

    //Generate random geometry for larger system
    srand(12345);  // Set seed for reproducibility
    for (int i = 0; i < natom * 3; i++) {
        int random_int = rand() % 181;
        double value = (double)random_int / 10.0;
        geom[i] = value * TOBOHRS;
    }

    // Build density matrix
    for (int i = 0; i < natom; i++) {
        for (int j = 0; j < natom; j++) {
            dens[i * natom + j] = 0.1;
            if (i == j) dens[i * natom + j] = 1.0;
        }
    }

    // Initialize Fock matrix
    for (int i = 0; i < natom * natom; i++) {
        fock[i] = 0.0;
    }

    // Normalize primitive GTO weights
    for (int i = 0; i < ngauss; i++) {
        coef[i] *= pow(2.0 * xpnt[i], 0.75);
    }

    // Compute Schwarz Inequality factors
    int nn = (natom * natom + natom) / 2;
    double *schwarz = (double*)malloc(nn * sizeof(double));
    int ij = 0;
    for (int i = 0; i < natom; i++) {
        for (int j = 0; j <= i; j++) {
            double eri;
            ssss(i, j, i, j, ngauss, xpnt, coef, geom, &eri);
            schwarz[ij] = sqrt(fabs(eri));
            ij++;
        }
    }

    // Compute Hartree-Fock
    int nnnn = (nn * nn + nn) / 2;
    printf("nnnn: %d\n", nnnn);

    // Warmup run
    printf("Performing warmup run...\n");
    hartree_fock(fock, dens, schwarz, geom, xpnt, coef, ngauss, nnnn, natom, DTOL, RCUT, nn);

    // Reset fock matrix
    for (int i = 0; i < natom * natom; i++) {
        fock[i] = 0.0;
    }

        // Timing runs
    double total_time = 0.0;
    int nruns = 10;
    printf("Performing %d timed runs:\n", nruns);

    for (int run = 0; run < nruns; run++) {
        // Reset fock matrix
        for (int i = 0; i < natom * natom; i++) {
            fock[i] = 0.0;
        }

        double start_time = get_time();
        hartree_fock(fock, dens, schwarz, geom, xpnt, coef, ngauss, nnnn, natom, DTOL, RCUT, nn);
        #pragma omp taskwait
        double end_time = get_time();
       
        double run_time = end_time - start_time;
        total_time += run_time;
       
        // Compute 2e- energy
        double erep = 0.0;
        for (int i = 0; i < natom; i++) {
            for (int j = 0; j < natom; j++) {
                erep += fock[i * natom + j] * dens[i * natom + j];
            }
        }
       
        printf("Run %d: Time = %.6f seconds, 2e- energy = %f\n", run + 1, run_time, erep * 0.5);
    }

    // Calculate and print average time
    double avg_time = total_time / nruns;
    printf("\nAverage time per run: %.6f seconds\n", avg_time);





    // Free memory
    free(xpnt);
    free(coef);
    free(geom);
    free(fock);
    free(dens);
    free(schwarz);

    return 0;
}
