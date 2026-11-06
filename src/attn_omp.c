/*
 * attn_omp.c -- Q4: your best variant, parallelised with OpenMP.
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

#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

#define TK 8  /* register tile for k-dimension (fits 8 floats = 1 AVX register) */
#define TI 16 /* query row block (fits in L1) */
#define TJ 64 /* key/value row block (fits in L1) */

/* ------------------------------------------------------------------------- */
/* attention(): OpenMP parallelized implementation                             */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);

    /* Allocate and initialize S matrix */
    float *S = malloc((size_t)N * N * sizeof(float));
    if (!S)
    {
        perror("malloc S");
        exit(1);
    }

    /* Initialize S to zero */
    memset(S, 0, (size_t)N * N * sizeof(float));

/* Step 1: S = Q @ K^T * scale - PARALLELIZED over ti (query rows) */
#pragma omp parallel for schedule(dynamic, 1)
    for (int ti = 0; ti < N; ti += TI)
    {
        int ti_end = ti + TI < N ? ti + TI : N;

        for (int tj = 0; tj < N; tj += TJ)
        {
            int tj_end = tj + TJ < N ? tj + TJ : N;

            /* Prefetch next tile of K (optional optimization) */
            if (tj_end < N)
            {
                __builtin_prefetch(&K[tj_end * d], 0, 3);
            }

            /* Compute dot products for this (ti, tj) tile */
            for (int i = ti; i < ti_end; i++)
            {
                const float *Qi = Q + (size_t)i * d;
                float *Si = S + (size_t)i * N;

                for (int j = tj; j < tj_end; j++)
                {
                    const float *Kj = K + (size_t)j * d;

#if USE_AVX && (TK >= 8)
                    /* AVX2 vectorized dot product */
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    int limit = (d / TK) * TK;

                    /* Process complete AVX tiles */
                    for (; k < limit; k += TK)
                    {
                        __m256 vq = _mm256_loadu_ps(Qi + k);
                        __m256 vk = _mm256_loadu_ps(Kj + k);
                        acc = _mm256_fmadd_ps(vq, vk, acc);
                    }

                    /* Horizontal sum of accumulator */
                    float temp[8];
                    _mm256_storeu_ps(temp, acc);
                    float sum = temp[0] + temp[1] + temp[2] + temp[3] +
                                temp[4] + temp[5] + temp[6] + temp[7];

                    /* Handle remaining elements */
                    for (; k < d; k++)
                    {
                        sum += Qi[k] * Kj[k];
                    }

                    Si[j] = sum * scale;
#else
                    /* Scalar fallback */
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++)
                    {
                        sum += Qi[k] * Kj[k];
                    }
                    Si[j] = sum * scale;
#endif
                }
            }
        }
    }

/* Step 2: Softmax per row - PARALLELIZED over rows */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        float *row = S + (size_t)i * N;

        /* Find max for numerical stability */
        float max = row[0];
        for (int j = 1; j < N; j++)
        {
            if (row[j] > max)
                max = row[j];
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
        {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }

        /* Normalize */
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; j++)
        {
            row[j] *= inv_sum;
        }
    }

    /* Step 3: O = S @ V - PARALLELIZED over ti (output rows) */
    memset(O, 0, (size_t)N * d * sizeof(float));

#pragma omp parallel for schedule(dynamic, 1)
    for (int ti = 0; ti < N; ti += TI)
    {
        int ti_end = ti + TI < N ? ti + TI : N;

        for (int tj = 0; tj < N; tj += TJ)
        {
            int tj_end = tj + TJ < N ? tj + TJ : N;

            /* Prefetch next tile of V */
            if (tj_end < N)
            {
                __builtin_prefetch(&V[tj_end * d], 0, 3);
            }

            for (int i = ti; i < ti_end; i++)
            {
                const float *Si = S + (size_t)i * N;
                float *Oi = O + (size_t)i * d;

                for (int j = tj; j < tj_end; j++)
                {
                    float s = Si[j];
                    const float *Vj = V + (size_t)j * d;

#if USE_AVX && (d >= 8)
                    /* AVX2 scaled accumulate */
                    __m256 vs = _mm256_set1_ps(s);
                    int k = 0;
                    int dk = (d / TK) * TK;

                    for (; k < dk; k += TK)
                    {
                        __m256 vj = _mm256_loadu_ps(Vj + k);
                        __m256 vo = _mm256_loadu_ps(Oi + k);
                        vo = _mm256_fmadd_ps(vs, vj, vo);
                        _mm256_storeu_ps(Oi + k, vo);
                    }

                    /* Handle remaining elements */
                    for (; k < d; k++)
                    {
                        Oi[k] += s * Vj[k];
                    }
#else
                    /* Scalar fallback */
                    for (int k = 0; k < d; k++)
                    {
                        Oi[k] += s * Vj[k];
                    }
#endif
                }
            }
        }
    }

    free(S);
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