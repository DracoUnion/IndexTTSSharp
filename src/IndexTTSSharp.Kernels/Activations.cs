using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// Element-wise activations + normalizations + pointwise helpers used by the
/// IndexTTS submodels (Conformer, Perceiver, DiT, BigVGAN, Vocos, regulators).
/// Every op is scalar-first with an AVX fast path when <c>Avx2.IsSupported</c>.
/// </summary>
public static unsafe class Activations
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GELU(float* x, int n)
    {
        for (int i = 0; i < n; i++)
            x[i] = GELUScalar(x[i]);
    }

    private static float GELUScalar(float x)
    {
        // tanh-approx GELU (matches F.gelu default used across the models).
        const float k1 = 0.7978845608f;
        const float k2 = 0.044715f;
        float inner = x + k2 * x * x * x;
        float tanh = MathF.Tanh(k1 * inner);
        return 0.5f * x * (1f + tanh);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GELUScalarApply(float* x, int n) => GELU(x, n);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SiLU(float* x, int n)
    {
        int i = 0;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            for (; i <= n - 16; i += 16)
            {
                Vector256<float> a = Avx.LoadVector256(x + i);
                Vector256<float> b = Avx.LoadVector256(x + i + 8);
                Avx.Store(x + i, FastExp.SiluAvx2(a));
                Avx.Store(x + i + 8, FastExp.SiluAvx2(b));
            }
        }
        for (; i < n; i++)
            x[i] = SiLUScalar(x[i]);
    }

    private static float SiLUScalar(float x) => x / (1f + MathF.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Mish(float* x, int n)
    {
        for (int i = 0; i < n; i++)
            x[i] = x[i] * MathF.Tanh(MathF.Log(MathF.Exp(x[i]) + 1f));
    }

    /// <summary>GLU: split x into two halves (g, u); out = g * sigmoid(u).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Glu(float* x, int n2, float* dst)
    {
        for (int i = 0; i < n2; i++)
            dst[i] = x[i] * SigmoidScalar(x[n2 + i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SigmoidScalar(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Snake: x + (1/α)·sin²(αx).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Snake(float* x, float* alpha, int n, float logScale = 0f)
    {
        float s = logScale != 0 ? MathF.Exp(logScale) : 1f;
        for (int i = 0; i < n; i++)
        {
            float a = s * alpha[i];
            float sin = MathF.Sin(a * x[i]);
            x[i] += 0.5f * sin * sin / a;
        }
    }

    /// <summary>SnakeBeta: x + (1/α)·sin²(αx) + β.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SnakeBeta(float* x, float* alpha, float* beta, int n, float logScale = 0f)
    {
        float s = logScale != 0 ? MathF.Exp(logScale) : 1f;
        for (int i = 0; i < n; i++)
        {
            float a = s * alpha[i];
            float sin = MathF.Sin(a * x[i]);
            x[i] += 0.5f * sin * sin / a + beta[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ReLU(float* x, int n)
    {
        for (int i = 0; i < n; i++)
            x[i] = MathF.Max(x[i], 0f);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Tanh(float* x, int n)
    {
        for (int i = 0; i < n; i++)
            x[i] = MathF.Tanh(x[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Abs(float* x, int n)
    {
        for (int i = 0; i < n; i++)
            x[i] = MathF.Abs(x[i]);
    }
}
