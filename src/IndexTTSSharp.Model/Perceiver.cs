using System.Runtime.CompilerServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// PerceiverResampler (from <c>indextts/gpt/perceiver.py</c>).
///
/// Two variants are used in the pipeline:
///  - spk/conditioning perceiver: <c>perceiver.*</c>, latent dim == model_dim,
///    outputs 32 latents used as GPT conditioning.
///  - emo perceiver: <c>emo_perceiver.*</c>, 1 latent, context dim = conformer out.
///
/// Tensor naming (GGUF torch-module path):
///   perceiver.latents                      [num_latents, dim]
///   perceiver.proj_context.weight|bias     [dim, dim_context]
///   perceiver.layers.{i}.attn.to_q.weight [dim*heads, dim]
///   perceiver.layers.{i}.attn.to_kv.weight[2*dim*heads, dim_context]
///   perceiver.layers.{i}.attn.to_out.weight[dim, dim*heads]
///   perceiver.layers.{i}.ff.{i}.weight|bias
///   perceiver.norm.{i}
/// </summary>
public unsafe sealed class Perceiver
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly ScratchArena _scratch = new();

    private readonly float* _latents;
    private readonly int _dim;
    private readonly int _numLatents;
    private readonly int _heads;

    public Perceiver(WeightStore w, CpuThreadPool? pool, string prefix, int dim, int numLatents, int heads)
    {
        _w = w;
        _pool = pool;
        _dim = dim;
        _numLatents = numLatents;
        _heads = heads;
        _latents = w.GetFloat(prefix + "latents", out _);
    }

    /// <summary>
    /// Run the perceiver over a context (ctxLen, ctxDim) → (numLatents, dim).
    /// <paramref name="dst"/> is caller-owned, ≥ numLatents*dim floats.
    /// </summary>
    public void Forward(float* ctx, int ctxLen, int ctxDim, float* mask, float* dst)
    {
        int layers = 2; // PerceiverResampler default depth
        int headDim = _dim / _heads;
        int dimInner = headDim * _heads;

        // x = proj_context(ctx) if ctxDim != dim, else ctx
        float* x = (float*)_scratch.A((nuint)(ctxLen * _dim * sizeof(float)));
        if (ctxDim != _dim)
        {
            float* projW = _w.GetFloat("perceiver.proj_context.weight", out _);
            float* projB = _w.TryGetFloat("perceiver.proj_context.bias", out _);
            MatmulF32.Gemm(ctx, projW, x, ctxLen, ctxDim, _dim, false, _pool);
            AddBias(x, projB, ctxLen, _dim, projB != null);
        }
        else
        {
            Buffer.MemoryCopy(ctx, x, (nuint)(ctxLen * ctxDim * sizeof(float)), (nuint)(ctxLen * _dim * sizeof(float)));
        }

        // latents [numLatents, dim]
        float* lat = (float*)_scratch.B((nuint)(_numLatents * _dim * sizeof(float)));
        Buffer.MemoryCopy(_latents, lat, (nuint)(_numLatents * _dim * sizeof(float)), (nuint)(_numLatents * _dim * sizeof(float)));

        float* attnQ = (float*)_scratch.C((nuint)(_numLatents * dimInner * sizeof(float)));
        float* attnKV = (float*)_scratch.D((nuint)((ctxLen + _numLatents) * 2 * dimInner * sizeof(float)));
        float* attnOut = (float*)_scratch.E((nuint)(_numLatents * dimInner * sizeof(float)));
        float* ffn = (float*)_scratch.F((nuint)(_numLatents * 4 * _dim * sizeof(float)));

        for (int li = 0; li < layers; li++)
        {
            string p = $"perceiver.layers.{li}.";
            float* toQ = _w.GetFloat(p + "attn.to_q.weight", out _);
            float* toKV = _w.GetFloat(p + "attn.to_kv.weight", out _);
            float* toOut = _w.GetFloat(p + "attn.to_out.weight", out _);

            // context is latents concatenated with x
            int totalCtx = _numLatents + ctxLen;
            float* ctxConcat = attnKV; // reuse attnKV for the concatenated context
            Buffer.MemoryCopy(lat, ctxConcat, (nuint)(_numLatents * _dim * sizeof(float)), (nuint)(_numLatents * _dim * sizeof(float)));
            Buffer.MemoryCopy(x, ctxConcat + _numLatents * _dim, (nuint)(ctxLen * _dim * sizeof(float)), (nuint)(ctxLen * _dim * sizeof(float)));

            // q = toQ(latents) [numLatents, dimInner]
            MatmulF32.Gemm(lat, toQ, attnQ, _numLatents, _dim, dimInner, false, _pool);
            // kv = toKV(ctxConcat) [totalCtx, 2*dimInner]
            MatmulF32.Gemm(ctxConcat, toKV, attnKV + _numLatents * 2 * dimInner, totalCtx, _dim, 2 * dimInner, false, _pool);
            float* k = attnKV + _numLatents * 2 * dimInner;
            float* v = k + totalCtx * dimInner;

            // cross attention: q (numLatents, heads, headDim) vs k/v (totalCtx, ...)
            float scale = 1f / MathF.Sqrt(headDim);
            float* scores = (float*)_scratch.C((nuint)(_numLatents * totalCtx * sizeof(float)));
            Attention.Scores(attnQ, k, scores,
                _heads, _heads, headDim, _numLatents, totalCtx, scale, _pool);
            Attention.Combine(v, scores, attnOut, _heads, _heads, headDim, _numLatents, totalCtx, _pool);

            // to_out: [numLatents, dim] = attnOut · toOutᵀ
            float* head = (float*)_scratch.F((nuint)(_numLatents * _dim * sizeof(float)));
            MatmulF32.Gemm(attnOut, toOut, head, _numLatents, dimInner, _dim, false, _pool);
            // residual
            for (int i = 0; i < _numLatents * _dim; i++)
                lat[i] += head[i];

            // ffn: GELU(gamma·ff1 + beta)·ff2
            float* ff1 = _w.GetFloat(p + "ff.0.weight", out _);
            float* ff1B = _w.TryGetFloat(p + "ff.0.bias", out _);
            float* ff2 = _w.GetFloat(p + "ff.2.weight", out _);
            float* ff2B = _w.TryGetFloat(p + "ff.2.bias", out _);
            int ffInner = 4 * _dim * 2 / 3;
            MatmulF32.Gemm(lat, ff1, ffn, _numLatents, _dim, ffInner, false, _pool);
            AddBias(ffn, ff1B, _numLatents, ffInner, ff1B != null);
            GELU(ffn, _numLatents * ffInner);
            MatmulF32.Gemm(ffn, ff2, head, _numLatents, ffInner, _dim, false, _pool);
            AddBias(head, ff2B, _numLatents, _dim, ff2B != null);
            for (int i = 0; i < _numLatents * _dim; i++)
                lat[i] += head[i];
        }

        // norm: RMSNorm(latents)
        float* gamma = _w.GetFloat("perceiver.norm.0", out _);
        RMSNorm(lat, gamma, _numLatents, _dim);
        if (dst != null)
            Buffer.MemoryCopy(lat, dst, (nuint)(_numLatents * _dim * sizeof(float)), (nuint)(_numLatents * _dim * sizeof(float)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RMSNorm(float* x, float* gamma, int rows, int dim)
    {
        float scale = MathF.Sqrt(dim);
        for (int r = 0; r < rows; r++)
        {
            float* row = x + r * dim;
            float sq = 0f;
            for (int i = 0; i < dim; i++)
                sq += row[i] * row[i];
            float inv = 1f / MathF.Sqrt(sq + 1e-5f);
            for (int i = 0; i < dim; i++)
                row[i] = row[i] * inv * scale * gamma[i];
        }
    }

    private static void AddBias(float* x, float* bias, int rows, int cols, bool addBias)
    {
        if (bias == null || !addBias)
            return;
        for (int r = 0; r < rows; r++)
        {
            float* row = x + r * cols;
            for (int c = 0; c < cols; c++)
                row[c] += bias[c];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GELU(float* x, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float v = x[i];
            float t = MathF.Tanh(v * 0.7978845608f + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + t);
        }
    }
}
