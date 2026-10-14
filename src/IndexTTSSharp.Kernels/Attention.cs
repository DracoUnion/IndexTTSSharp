using System.Runtime.CompilerServices;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// General multi-head attention (non-causal by default) used by Perceiver,
/// Conformer, and DiT. GPT-2 attention reuses the quantized kernels +
/// <c>SoftmaxCausal</c> from HyMT2Sharp's Ops. This handles arbitrary
/// q/kv head counts (GQA) and optional padding masks.
///
/// Layouts are the same as HyMT2Sharp's Ops.AttentionScores/Combine:
///   q: [qLen, heads, headDim]  (batch=1, no heads-major)
///   k: [kvLen, kvHeads, headDim]
///   v: [kvLen, kvHeads, headDim]
///   out: [qLen, heads, headDim]
/// </summary>
public static unsafe class Attention
{
    /// <summary>Scores = (q·kᵀ)/scale. Non-causal: full kvLen visible.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scores(
        float* q, float* k, float* scores,
        int heads, int kvHeads, int headDim,
        int qLen, int kvLen, float scale,
        CpuThreadPool? pool = null)
    {
        int group = heads / kvHeads;
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                int kvh = h / group;
                for (int qt = 0; qt < qLen; qt++)
                {
                    float* qh = q + (qt * heads + h) * headDim;
                    float* row = scores + (h * qLen + qt) * kvLen;
                    for (int kt = 0; kt < kvLen; kt++)
                    {
                        float* kh = k + (kt * kvHeads + kvh) * headDim;
                        float acc = 0f;
                        for (int d = 0; d < headDim; d++)
                            acc += qh[d] * kh[d];
                        row[kt] = acc * scale;
                    }
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }

    /// <summary>Combine = softmax(scores) · v. Softmax over kvLen row, then out = Σ w·v.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Combine(
        float* v, float* scores, float* result,
        int heads, int kvHeads, int headDim,
        int qLen, int kvLen,
        CpuThreadPool? pool = null)
    {
        int group = heads / kvHeads;
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                int kvh = h / group;
                for (int qt = 0; qt < qLen; qt++)
                {
                    float* row = scores + (h * qLen + qt) * kvLen;
                    // softmax
                    float max = -80f;
                    for (int kt = 0; kt < kvLen; kt++)
                        if (row[kt] > max) max = row[kt];
                    float sum = 0f;
                    for (int kt = 0; kt < kvLen; kt++)
                    {
                        row[kt] = MathF.Exp(row[kt] - max);
                        sum += row[kt];
                    }
                    float inv = sum > 0 ? 1f / sum : 0f;
                    for (int kt = 0; kt < kvLen; kt++)
                        row[kt] *= inv;
                    // result[qt,h] = Σ row[kt] · v[kt,kvh]
                    float* outH = result + (qt * heads + h) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int kt = 0; kt < kvLen; kt++)
                            acc += row[kt] * v[(kt * kvHeads + kvh) * headDim + d];
                        outH[d] = acc;
                    }
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }
}
