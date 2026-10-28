/*
 * attn_tiled.c -- Q3b: cache and register tiling.
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
    // (void)Q; (void)K; (void)V; (void)O; (void)N; (void)d;
    // printf("attn_tiled: not implemented yet\n");
    // exit(0); /* no output file is written, so `make check` skips this variant */

    const float scale = 1.0f / sqrtf((float)d);

    /* Tile sizes: tune these based on your cache size */
    const int Br = 64;   /* block rows of attention matrix */
    const int Bc = 64;   /* block cols of attention matrix */
    const int Br_v = 64; /* register tile for V dimension */

    float *S = malloc((size_t)Br * Bc * sizeof(float));
    float *l = malloc((size_t)Br * sizeof(float)); /* row max */
    float *m = malloc((size_t)Br * sizeof(float)); /* row sum for softmax */

    /* Process attention in row blocks */
    for (int i_start = 0; i_start < N; i_start += Br)
    {
        int i_end = (i_start + Br < N) ? i_start + Br : N;
        int Br_actual = i_end - i_start;

        /* Initialize accumulators for this row block */
        for (int i = 0; i < Br_actual; i++)
        {
            l[i] = -1e9f; /* row max for softmax stability */
            m[i] = 0.0f;  /* row sum for normalization */
        }

        float *O_block = O + (size_t)i_start * d;
        memset(O_block, 0, (size_t)Br_actual * d * sizeof(float));

        /* Column-wise blocking: process attention matrix in column blocks */
        for (int j_start = 0; j_start < N; j_start += Bc)
        {
            int j_end = (j_start + Bc < N) ? j_start + Bc : N;
            int Bc_actual = j_end - j_start;

            /* Step 1: Compute S = Q[i_start:i_end] @ K[j_start:j_end]^T * scale */
            for (int i = i_start; i < i_end; i++)
            {
                for (int j = j_start; j < j_end; j++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < d; k++)
                    {
                        acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
                    }
                    S[(size_t)(i - i_start) * Bc + (j - j_start)] = acc * scale;
                }
            }

            /* Step 2: Online softmax (Welford-style) for numerical stability */
            for (int i = 0; i < Br_actual; i++)
            {
                float *row = S + (size_t)i * Bc;
                for (int j = 0; j < Bc_actual; j++)
                {
                    float val = row[j];
                    if (val > l[i])
                    {
                        /* Update with new max */
                        float diff = l[i] - val;
                        m[i] = m[i] * expf(diff) + expf(0.0f);
                        l[i] = val;
                    }
                    else
                    {
                        m[i] += expf(val - l[i]);
                    }
                    row[j] = val; /* keep for V multiply */
                }
            }
        }

        /* Step 3: Apply normalized softmax and multiply by V */
        for (int i_rel = 0; i_rel < Br_actual; i_rel++)
        {
            int i = i_start + i_rel;
            for (int k = 0; k < d; k++)
            {
                float acc = 0.0f;
                for (int j_start = 0; j_start < N; j_start += Bc)
                {
                    int j_end = (j_start + Bc < N) ? j_start + Bc : N;
                    int Bc_actual = j_end - j_start;

                    float *row = S + (size_t)i_rel * Bc;
                    for (int j = j_start; j < j_end; j++)
                    {
                        /* Softmax weight: exp(S[i,j] - max) / sum */
                        float weight = expf(S[(size_t)i_rel * Bc + (j - j_start)] - l[i_rel]) / m[i_rel];
                        acc += weight * V[(size_t)j * d + k];
                    }
                }
                O[(size_t)i * d + k] = acc;
            }
        }
    }

    free(S);
    free(l);
    free(m);
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
