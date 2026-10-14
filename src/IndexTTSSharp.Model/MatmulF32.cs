using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

// F32 dense GEMM (C = A · Bᵀ or A·B) used by all the non-quantized submodels
// (GPT, Conformer, Perceiver, DiT, MLPs). A is [rows, kIn] row-major, B is
// [kIn, cols] row-major (i.e. the weight matrix as stored: out×in, kIn fastest
// on the last axis). This is a plain, correct, threaded GEMM; the hot GPT
// transformer path can later be swapped for a Q4_K panel GEMM.
public static unsafe class MatmulF32
{
    /// <summary>C[rows, cols] += A[rows, kIn] · B[kIn, cols]ᵀ? — B is [cols, kIn] row-major.</summary>
    /// A: [rows × kIn], B: [cols × kIn] (columns of B over kIn). C: [rows × cols].
    /// Standard "row-major outer product" layout where the last axis of B is kIn.
    public static void Gemm(float* a, float* b, float* c, int rows, int kIn, int cols, bool accumulate, CpuThreadPool? pool = null)
    {
        if (!accumulate)
        {
            for (int i = 0; i < rows * cols; i++)
                c[i] = 0f;
        }

        if (pool == null || rows <= 1)
        {
            GemmRows(c, a, b, 0, rows, kIn, cols);
            return;
        }

        pool.For(rows, (int worker, int workers) =>
        {
            int begin = rows * worker / workers;
            int end = rows * (worker + 1) / workers;
            GemmRows(c, a, b, begin, end, kIn, cols);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmRows(float* c, float* a, float* b, int begin, int end, int kIn, int cols)
    {
        for (int r = begin; r < end; r++)
        {
            float* aRow = a + r * kIn;
            float* cRow = c + r * cols;
            if (Avx.IsSupported && Fma.IsSupported && kIn >= 8)
                GemmRowAvx(cRow, aRow, b, kIn, cols);
            else
                GemmRowScalar(cRow, aRow, b, kIn, cols);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmRowAvx(float* cRow, float* aRow, float* b, int kIn, int cols)
    {
        // Outer-product: for each k, cRow += aRow[k] · b[col][k]. b is [cols, kIn].
        for (int k = 0; k < kIn; k++)
        {
            float ak = aRow[k];
            float* bk = b + k * cols;
            Vector256<float> a = Vector256.Create(ak);
            for (int j = 0; j < cols; j += 8)
            {
                Vector256<float> bv = Avx.LoadVector256(bk + j);
                cRow += 0; // keep compiler honest about alignment
                Vector256<float> cv = Avx.LoadVector256(cRow + j);
                cv = Fma.MultiplyAdd(a, bv, cv);
                Avx.Store(cRow + j, cv);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmRowScalar(float* cRow, float* aRow, float* b, int kIn, int cols)
    {
        for (int k = 0; k < kIn; k++)
        {
            float ak = aRow[k];
            float* bk = b + k * cols;
            for (int j = 0; j < cols; j++)
                cRow[j] += ak * bk[j];
        }
    }

    /// <summary>Single-token GEMV: y[cols] = A[1, kIn] · B[kIn, cols]. B stored [cols, kIn].</summary>
    public static void Gemv(float* x, float* b, float* y, int kIn, int cols, bool addBias, float* bias, CpuThreadPool? pool = null)
    {
        if (pool == null || cols <= 64)
        {
            GemvRows(y, x, b, 0, cols, kIn, addBias, bias);
            return;
        }

        pool.For(cols, (int worker, int workers) =>
        {
            int begin = cols * worker / workers;
            int end = cols * (worker + 1) / workers;
            GemvRows(y, x, b, begin, end, kIn, addBias, bias);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvRows(float* y, float* x, float* b, int begin, int end, int kIn, bool addBias, float* bias)
    {
        for (int j = begin; j < end; j++)
        {
            float acc = addBias && bias != null ? bias[j] : 0f;
            float* bj = b + j * kIn;
            for (int k = 0; k < kIn; k++)
                acc += x[k] * bj[k];
            y[j] = acc;
        }
    }
}
