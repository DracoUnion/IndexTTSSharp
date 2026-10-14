using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// A F32 GEMM/GEMV that works on the raw weight pointers <c>Linear</c> layers
/// need. Many IndexTTS submodels (conformer, perceiver, DiT, BigVGAN) have
/// F32 (or dequant-friendly) weights; this keeps a single, correct, simple
/// path that the Model layer can call without the quant-panel complexity of
/// <see cref="MulMatQ4K"/>. Threaded over output rows.
/// </summary>
public static unsafe class MatmulF32
{
    /// <summary>C[m, n] = A[m, k] · B[k, n] + (bias!=null ? bias : 0). Row-major.</summary>
    public static void GEMM(float* a, int k, float* b, float* c, int m, int n, bool addBias, float* bias, CpuThreadPool? pool)
    {
        void Body(int worker, int workers)
        {
            int begin = m * worker / workers;
            int end = m * (worker + 1) / workers;
            for (int r = begin; r < end; r++)
            {
                float* aRow = a + r * k;
                float* cRow = c + r * n;
                GEMMRow(aRow, b, cRow, k, n, addBias, bias);
            }
        }

        if (pool == null || m <= 1)
            Body(0, 1);
        else
            pool.For(m, Body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GEMMRow(float* aRow, float* b, float* cRow, int k, int n, bool addBias, float* bias)
    {
        if (addBias)
        {
            for (int i = 0; i < n; i++)
                cRow[i] = bias[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                cRow[i] = 0f;
        }

        for (int ki = 0; ki < k; ki++)
        {
            float av = aRow[ki];
            float* bRow = b + ki * n;
            if (Avx.IsSupported && Fma.IsSupported && (n & 7) == 0)
            {
                for (int j = 0; j < n; j += 8)
                {
                    Vector256<float> bv = Avx.LoadVector256(bRow + j);
                    Vector256<float> cv = Avx.LoadVector256(cRow + j);
                    cv = Fma.MultiplyAdd(Vector256.Create(av), bv, cv);
                    Avx.Store(cRow + j, cv);
                }
            }
            else
            {
                for (int j = 0; j < n; j++)
                    cRow[j] += av * bRow[j];
            }
        }
    }

    /// <summary>Single row GEMV: y[n] = x[k] · B[k,n] (+bias).</summary>
    public static void GEMV(float* x, float* b, float* y, int k, int n, bool addBias, float* bias, CpuThreadPool? pool)
    {
        void Body(int worker, int workers)
        {
            int begin = n * worker / workers;
            int end = n * (worker + 1) / workers;
            for (int col = begin; col < end; col++)
            {
                float acc = addBias && bias != null ? bias[col] : 0f;
                float* bc = b + col * k;
                for (int ki = 0; ki < k; ki++)
                    acc += x[ki] * bc[ki];
                y[col] = acc;
            }
        }

        if (pool == null || n <= 32)
            Body(0, 1);
        else
            pool.For(n, Body);
    }
}
