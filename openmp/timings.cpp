#include <iostream>
#include <time.h>
#include <omp.h>

int main() {
    const int ntests = 20;
    const int n = 400;
    int nthreads;
    double a[n][n], b[n][n], c[n][n];
    double t1, t2, elapsed_time, cpu_time;
    clock_t tclock1, tclock2;
    int clock_rate;
    int i, j, k, itest;

    nthreads = omp_get_max_threads();  // Get the number of threads from OpenMP
    std::cout << "Using " << nthreads << " threads (set by environment variable)\n";

    std::cout << "Will multiply n by n matrices, with n=400\n";

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            a[i][j] = 1.0;
            b[i][j] = 1.0;
            c[i][j] = 0.0;
        }
    }

    tclock1 = clock();  // start wall timer
    t1 = omp_get_wtime();   // start cpu timer
    for (itest = 1; itest <= ntests; ++itest) {
        #pragma omp parallel for private(j, k) shared(a,b,c)
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                c[i][j] = 0.0;
                for (k = 0; k < n; ++k) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    t2 = omp_get_wtime();   // end cpu timer
    tclock2 = clock();
    elapsed_time = t2 - t1;
    cpu_time = ((double)(tclock2 - tclock1)) / CLOCKS_PER_SEC;

    std::cout << "Performed " << ntests << " matrix multiplies: CPU time = " << cpu_time << " seconds\n";
    std::cout << "Elapsed time = " << elapsed_time << " seconds\n";

/*
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
     
            std::cout << c[i][j] << '\t';
        }
    }
*/
    
    return 0;
}
