/*
 * attn_fused.c -- Q3c: row-wise fusion (the N x N matrix is never stored).
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    // (void)Q;
    // (void)K;
    // (void)V;
    // (void)O;
    // (void)N;
    // (void)d;
    // printf("attn_fused: not implemented yet\n");
    // exit(0); /* no output file is written, so `make check` skips this variant */

    const float scale = 1.0f / sqrtf((float)d);

    /* Process each row i independently - never allocate the full N×N S matrix */
    for (int i = 0; i < N; i++)
    {
        const float *Qi = Q + (size_t)i * d;
        float *Oi = O + (size_t)i * d;

        /* Allocate one row of S (only N floats, not N²) */
        float *S_row = malloc(N * sizeof(float));
        if (!S_row)
        {
            perror("malloc S_row");
            exit(1);
        }

        /* Step 1: Compute one row of S = Q[i] @ K^T */
        for (int j = 0; j < N; j++)
        {
            const float *Kj = K + (size_t)j * d;
            float acc = 0.0f;
            for (int k = 0; k < d; k++)
                acc += Qi[k] * Kj[k];
            S_row[j] = acc * scale;
        }

        /* Step 2: Softmax on this single row */
        float max = S_row[0];
        for (int j = 1; j < N; j++)
            if (S_row[j] > max)
                max = S_row[j];

        float sum = 0.0f;
        for (int j = 0; j < N; j++)
        {
            S_row[j] = expf(S_row[j] - max);
            sum += S_row[j];
        }
        for (int j = 0; j < N; j++)
            S_row[j] /= sum;

        /* Step 3: Multiply S_row @ V to get O[i] */
        for (int k = 0; k < d; k++)
        {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += S_row[j] * V[(size_t)j * d + k];
            Oi[k] = acc;
        }

        free(S_row);
    }
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in)
    {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1)
    {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n)
    {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++)
    {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n)
    {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
