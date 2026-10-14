using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// Normalization + pointwise ops that the IndexTTS submodels need but that
/// HyMT2Sharp's <c>Ops</c> does not provide (LayerNorm, GroupNorm, BatchNorm,
/// general Softmax, GELU, Mish, Sigmoid). All scalar-first; AVX where useful.
/// Inputs are channel-major or row-major per the call site.
/// </summary>
public static unsafe class Norms
{
    /// <summary>LayerNorm over the last dim. <paramref name="x"/>: [rows, dim]; weight/bias: [dim].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void LayerNorm(float* x, float* weight, float* bias, int rows, int dim, float eps, CpuThreadPool? pool = null)
    {
        if (pool == null || rows <= 1)
        {
            LayerNormRange(x, weight, bias, 0, rows, dim, eps);
            return;
        }

        pool.For(rows, (int worker, int workers) =>
        {
            int begin = rows * worker / workers;
            int end = rows * (worker + 1) / workers;
            LayerNormRange(x, weight, bias, begin, end, dim, eps);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void LayerNormRange(float* x, float* weight, float* bias, int begin, int end, int dim, float eps)
    {
        for (int r = begin; r < end; r++)
        {
            float* row = x + r * dim;
            float mean = 0f;
            for (int i = 0; i < dim; i++)
                mean += row[i];
            mean /= dim;
            float var = 0f;
            for (int i = 0; i < dim; i++)
            {
                float d = row[i] - mean;
                var += d * d;
            }
            var /= dim;
            float inv = 1f / MathF.Sqrt(var + eps);
            for (int i = 0; i < dim; i++)
            {
                float n = (row[i] - mean) * inv;
                row[i] = n * (weight != null ? weight[i] : 1f) + (bias != null ? bias[i] : 0f);
            }
        }
    }

    /// <summary>
    /// GroupNorm: input <paramref name="x"/> is [inChannels, time] channel-major;
    /// split inChannels into numGroups groups of inChannels/numGroups channels.
    /// weight/bias are per-channel [inChannels].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GroupNorm(float* x, float* weight, float* bias, int inChannels, int time, int numGroups, float eps)
    {
        int g = inChannels / numGroups;
        int perGroupTime = g * time;
        for (int gr = 0; gr < numGroups; gr++)
        {
            float* grpBase = x + gr * perGroupTime;
            float mean = 0f;
            for (int i = 0; i < perGroupTime; i++)
                mean += grpBase[i];
            mean /= perGroupTime;
            float var = 0f;
            for (int i = 0; i < perGroupTime; i++)
            {
                float d = grpBase[i] - mean;
                var += d * d;
            }
            var /= perGroupTime;
            float inv = 1f / MathF.Sqrt(var + eps);
            for (int c = 0; c < g; c++)
            {
                int ci = gr * g + c;
                float* row = x + ci * time;
                float w = weight != null ? weight[ci] : 1f;
                float b = bias != null ? bias[ci] : 0f;
                for (int t = 0; t < time; t++)
                    row[t] = (row[t] - mean) * inv * w + b;
            }
        }
    }

    /// <summary>BatchNorm (inference): [inChannels, time], running mean/var + affine.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void BatchNorm(float* x, float* runningMean, float* runningVar, float* weight, float* bias, int inChannels, int time, float eps)
    {
        for (int c = 0; c < inChannels; c++)
        {
            float mean = runningMean[c];
            float inv = 1f / MathF.Sqrt(runningVar[c] + eps);
            float w = weight != null ? weight[c] : 1f;
            float b = bias != null ? bias[c] : 0f;
            float* row = x + c * time;
            for (int t = 0; t < time; t++)
                row[t] = (row[t] - mean) * inv * w + b;
        }
    }

    /// <summary>Softmax over the last dim. x: [rows, dim]; out: same (in-place if out==x).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Softmax(float* x, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float* row = x + r * dim;
            float max = -80f;
            for (int i = 0; i < dim; i++)
                if (row[i] > max)
                    max = row[i];
            float sum = 0f;
            for (int i = 0; i < dim; i++)
            {
                row[i] = MathF.Exp(row[i] - max);
                sum += row[i];
            }
            float inv = sum > 0 ? 1f / sum : 0f;
            for (int i = 0; i < dim; i++)
                row[i] *= inv;
        }
    }
}
