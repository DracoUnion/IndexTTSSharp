using System.Runtime.CompilerServices;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// Llama-style RoPE: precomputed cos/sin tables + rotate-half application.
/// Matches <c>precompute_freqs_cis</c> / <c>apply_rotary_emb</c> in the DiT
/// (<c>gpt_fast</c>) transformer.
/// </summary>
public static unsafe class Rope
{
    public static void Precompute(int seqLen, int nElem, float baseFreq, out float[] cos, out float[] sin)
    {
        int half = nElem / 2;
        cos = new float[seqLen * half];
        sin = new float[seqLen * half];
        float[] freqs = new float[half];
        for (int i = 0; i < half; i++)
            freqs[i] = 1f / MathF.Pow(baseFreq, (float)i / nElem);
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float angle = pos * freqs[i];
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>
    /// Apply rotate-half RoPE in place to <paramref name="qk"/> [rows, heads, headDim],
    /// where headDim = 2*half and rotate pairs are (i, i+half).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Apply(float* qk, int rows, int heads, int headDim, int pos, float[] cos, float[] sin)
    {
        int half = headDim / 2;
        fixed (float* cRow = &cos[pos * half], sRow = &sin[pos * half])
        {
            for (int r = 0; r < rows; r++)
            {
                for (int h = 0; h < heads; h++)
                {
                    float* row = qk + (r * heads + h) * headDim;
                    for (int i = 0; i < half; i++)
                    {
                        float c = cRow[i];
                        float s = sRow[i];
                        float x0 = row[i];
                        float x1 = row[i + half];
                        row[i] = x0 * c - x1 * s;
                        row[i + half] = x0 * s + x1 * c;
                    }
                }
            }
        }
    }
}
