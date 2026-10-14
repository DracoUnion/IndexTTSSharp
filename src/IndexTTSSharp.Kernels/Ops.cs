using System.Runtime.CompilerServices;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// Small pointwise / shape ops the IndexTTS submodels need that are not covered
/// by the GEMM/conv/norm kernels: nearest-interpolation (length regulator /
/// wavenet), cumulative sum (DiT mask path), statistics pool (CAMPPlus), and
/// embedding row-gather (quant/F32).
/// </summary>
public static unsafe class Ops
{
    /// <summary>Zero <paramref name="bytes"/> at <paramref name="dst"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Memset(void* dst, nuint bytes)
    {
        byte* p = (byte*)dst;
        nuint i = 0;
        for (; i + 7 < bytes; i += 8)
        {
            p[i] = 0; p[i + 1] = 0; p[i + 2] = 0; p[i + 3] = 0;
            p[i + 4] = 0; p[i + 5] = 0; p[i + 6] = 0; p[i + 7] = 0;
        }
        for (; i < bytes; i++)
            p[i] = 0;
    }

    /// <summary>
    /// Nearest-neighbor temporal up/downsample. <paramref name="src"/>: [C, srcLen]
    /// channel-major; <paramref name="dst"/>: [C, dstLen]. Each output frame t maps to
    /// the nearest input frame <c>round(t * srcLen / dstLen)</c>. Matches
    /// <c>F.interpolate(x, size=dstLen, mode='nearest')</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void InterpolateNearest(float* src, int inChannels, int srcLen,
        float* dst, int dstLen)
    {
        for (int c = 0; c < inChannels; c++)
        {
            float* s = src + c * srcLen;
            float* d = dst + c * dstLen;
            for (int t = 0; t < dstLen; t++)
            {
                int srcIdx = (int)(Math.Round((double)t * srcLen / dstLen));
                if (srcIdx >= srcLen) srcIdx = srcLen - 1;
                d[t] = s[srcIdx];
            }
        }
    }

    /// <summary>Cumulative sum over the last dim. x: [rows, dim].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Cumsum(float* x, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float* row = x + r * dim;
            float acc = 0f;
            for (int i = 0; i < dim; i++)
            {
                acc += row[i];
                row[i] = acc;
            }
        }
    }

    /// <summary>
    /// Statistics pooling: mean and unbiased-std over the time dim, concatenated
    /// as <c>[mean(C); std(C)]</c>. <paramref name="x"/>: [inChannels, time];
    /// <paramref name="outMean"/>/<paramref name="outStd"/>: [inChannels] each.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void StatsPool(float* x, int inChannels, int time,
        float* outMean, float* outStd)
    {
        float inv = 1f / time;
        float unbiased = time > 1 ? time / (time - 1) : 1f;
        for (int c = 0; c < inChannels; c++)
        {
            float* row = x + c * time;
            float mean = 0f;
            for (int t = 0; t < time; t++)
                mean += row[t];
            mean *= inv;
            float var = 0f;
            for (int t = 0; t < time; t++)
            {
                float d = row[t] - mean;
                var += d * d;
            }
            var *= inv * unbiased;
            outMean[c] = mean;
            outStd[c] = MathF.Sqrt(var);
        }
    }

    /// <summary>
    /// F32 embedding row-gather: <paramref name="dst"/>[i, :] =
    /// <paramref name="weight"/>[ids[i], :]. <paramref name="weight"/>:
    /// [numRows, dim] row-major.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void EmbedGatherF32(float* weight, int dim, int* ids, int count, float* dst)
    {
        for (int i = 0; i < count; i++)
        {
            int id = ids[i];
            float* src = weight + id * dim;
            float* outRow = dst + i * dim;
            for (int d = 0; d < dim; d++)
                outRow[d] = src[d];
        }
    }

    /// <summary>
    /// Softmax over a single row in place (max-subtract for stability).
    /// Used by Perceiver/DiT non-causal attention when qLen==1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SoftmaxRow(float* row, int n)
    {
        float max = -80f;
        for (int i = 0; i < n; i++)
            if (row[i] > max) max = row[i];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            row[i] = MathF.Exp(row[i] - max);
            sum += row[i];
        }
        float inv = sum > 0 ? 1f / sum : 0f;
        for (int i = 0; i < n; i++)
            row[i] *= inv;
    }
}
