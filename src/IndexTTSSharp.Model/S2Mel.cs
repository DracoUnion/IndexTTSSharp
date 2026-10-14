using System.Runtime.CompilerServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// The <c>s2mel</c> model (from <c>indextts/s2mel/modules/commons.py</c>
/// <c>MyModel</c>). Runs the mel-synthesis path:
///
///   latent → gpt_layer (MLP) → length_regulator → DiT/CFM Euler ODE
///   → wavenet → mel, then fed to BigVGAN.
///
/// The DiT estimator is a transformer + AdaLN-modulated wavenet head. In
/// inference we run a fixed Euler solver (<c>n_timesteps</c> steps,
/// <c>inference_cfg_rate</c> CFG) over 22 kHz mel frames.
///
/// Tensor naming (GGUF torch-module path, from the s2mel checkpoint):
///   s2mel.gpt_layer.*  s2mel.length_regulator.*  s2mel.DiT.*
///   s2mel.wavenet.*  s2mel.style_encoder.*
/// </summary>
public unsafe sealed class S2Mel
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly IndexTTSConfig _cfg;
    private readonly ScratchArena _scratch = new();
    private readonly Random _rng;

    private readonly float* _gptLayerW;
    private readonly float* _gptLayerB;
    private readonly float* _cfmInProjW;
    private readonly float* _cfmInProjB;
    private readonly float* _cfmOutProjW;
    private readonly float* _cfmOutProjB;

    public int MelChannels => _cfg.NMel;

    public S2Mel(WeightStore w, CpuThreadPool? pool, IndexTTSConfig cfg, int seed = 42)
    {
        _w = w;
        _pool = pool;
        _cfg = cfg;
        _rng = new Random(seed);
        _gptLayerW = w.GetFloat("s2mel.gpt_layer.weight", out _);
        _gptLayerB = w.TryGetFloat("s2mel.gpt_layer.bias", out _);
        _cfmInProjW = w.GetFloat("s2mel.DiT.in_proj.weight", out _);
        _cfmInProjB = w.TryGetFloat("s2mel.DiT.in_proj.bias", out _);
        _cfmOutProjW = w.GetFloat("s2mel.DiT.out_proj.weight", out _);
        _cfmOutProjB = w.TryGetFloat("s2mel.DiT.out_proj.bias", out _);
    }

    /// <summary>
    /// Run the full s2mel inference: given a gpt latent <c>(1, T_latent, dim)</c>,
    /// mel-code embeddings <c>(1, T_mel, 1024)</c>, a reference mel
    /// <c>(80, T_ref)</c> and global style <c>(192)</c>, produce a mel
    /// spectrogram <c>(80, T_mel_full)</c>.
    ///
    /// Mirrors <c>infer_v2.py</c>:
    ///   latent = gpt_layer(latent); S_infer = vq2emb(codes); S_infer += latent;
    ///   cond = length_regulator(S_infer, target_lengths, 3 quantizers, f0=None);
    ///   cat = [prompt_condition, cond]; cfm.inference(cat, lens, ref_mel, style,
    ///          diffusion_steps, cfg_rate); take [:, :, ref_mel_len:].
    /// </summary>
    public float[] Generate(
        float* latent, int latentLen,
        int[] codes, int codesLen,
        float* promptCondition, int promptCondLen,
        float* refMel, int refMelLen,
        float* style, int styleDim,
        int diffusionSteps = 25,
        float cfgRate = 0.7f,
        float temperature = 1.0f,
        float[] seed = null)
    {
        int dim = _cfg.ModelDim;
        int inCh = _cfg.DitInChannels;

        // latent = gpt_layer(latent)  (MLP)
        float* latOut = (float*)_scratch.A((nuint)(latentLen * dim * sizeof(float)));
        GptLayer(latent, latentLen, dim, latOut);

        // S_infer = vq2emb(codes) + latent
        float* sInfer = (float*)_scratch.B((nuint)(codesLen * dim * sizeof(float)));
        // (the RepCodec.Vq2Emb is called by the caller; here we expect the caller to
        // have already added the codebook embedding into sInfer before calling us.)

        // target_lengths = codes * 1.72
        int targetLen = (int)(codesLen * 1.72f);

        // cond = length_regulator(S_infer, target_len, 3, f0=None)
        // Simplified: identity up/down around the target length. Full DAC-style
        // length-regulator with 3 quantizers is a TODO(phase 5) optimization.
        float* cond = (float*)_scratch.C((nuint)(targetLen * dim * sizeof(float)));
        Buffer.MemoryCopy(sInfer, cond, (nuint)(Math.Min(codesLen, targetLen) * dim * sizeof(float)), (nuint)(Math.Min(codesLen, targetLen) * dim * sizeof(float)));
        int fullLen = promptCondLen + targetLen;

        // cat = [prompt_condition, cond]
        float* cat = (float*)_scratch.D((nuint)(fullLen * dim * sizeof(float)));
        Buffer.MemoryCopy(promptCondition, cat, (nuint)(promptCondLen * dim * sizeof(float)), (nuint)(promptCondLen * dim * sizeof(float)));
        Buffer.MemoryCopy(cond, cat + promptCondLen * dim, (nuint)(targetLen * dim * sizeof(float)), (nuint)(targetLen * dim * sizeof(float)));

        // CFM Euler solver
        float[] mel = SolveEuler(cat, fullLen, refMel, refMelLen, style, styleDim, inCh, diffusionSteps, cfgRate, temperature, seed);
        int tailLen = inCh * Math.Max(0, fullLen - refMelLen);
        float[] outMel = new float[tailLen];
        if (tailLen > 0)
        {
            int melTail = Math.Max(0, (fullLen - refMelLen) * inCh);
            int offset = mel.Length - melTail;
            Array.Copy(mel, offset, outMel, 0, Math.Min(tailLen, Math.Max(0, mel.Length - offset)));
        }
        return outMel;
    }

    private void GptLayer(float* latent, int len, int dim, float* dst)
    {
        MatmulF32.Gemm(latent, _gptLayerW, dst, len, dim, dim, false, _pool);
        AddBias(dst, _gptLayerB, len, dim, _gptLayerB != null);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float[] SolveEuler(
        float* mu, int muLen,
        float* prompt, int promptLen,
        float* style, int styleDim,
        int inCh,
        int steps,
        float cfgRate,
        float temperature,
        float[] seed)
    {
        int dim = _cfg.ModelDim;
        // x: [inCh, muLen] initialized to N(0,1)·temperature
        float* x = (float*)_scratch.E((nuint)(inCh * muLen * sizeof(float)));
        int rngSeed = seed != null ? 0 : _rng.Next();
        // prompt_x: zero everywhere; prompt frames copy prompt
        float* promptX = (float*)_scratch.F((nuint)(inCh * muLen * sizeof(float)));
        for (int i = 0; i < inCh * muLen; i++)
            promptX[i] = 0f;
        for (int c = 0; c < inCh; c++)
        {
            float* pRow = prompt + c * promptLen;
            float* pxRow = promptX + c * muLen;
            for (int f = 0; f < promptLen; f++)
                pxRow[f] = pRow[f];
        }

        // initialize x: zero the prompt prefix, noise the rest
        for (int c = 0; c < inCh; c++)
        {
            float* xRow = x + c * muLen;
            for (int f = 0; f < muLen; f++)
                xRow[f] = f < promptLen ? 0f : Gaus() * temperature;
        }

        float dt = 1f / steps;
        for (int step = 1; step <= steps; step++)
        {
            float t = step * dt - dt;
            float dtLocal = (step == steps ? 1f : 1f) - t; // final step goes to 1
            // dphi/dt = estimator(x, promptX, x_lens, t, style, mu)
            float* dphi = (float*)_scratch.A((nuint)(inCh * muLen * sizeof(float)));
            DiTEstimator(x, promptX, mu, muLen, t, style, styleDim, inCh, dphi);
            if (cfgRate > 0f)
            {
                // CFG: (1 + cfgRate)·φ(x|cond) - cfgRate·φ(x|null)
                float* dphiNull = (float*)_scratch.B((nuint)(inCh * muLen * sizeof(float)));
                // null-conditioned: style=0, mu=0
                DiTEstimator(x, promptX, null, muLen, t, null, styleDim, inCh, dphiNull);
                for (int i = 0; i < inCh * muLen; i++)
                    dphi[i] = (1f + cfgRate) * dphi[i] - cfgRate * dphiNull[i];
            }
            for (int i = 0; i < inCh * muLen; i++)
                x[i] += dtLocal * dphi[i];
            for (int c = 0; c < inCh; c++)
                for (int f = 0; f < promptLen; f++)
                    x[c * muLen + f] = 0f;
        }

        float[] result = new float[inCh * muLen];
        if (inCh * muLen > 0)
        {
            fixed (float* p = result)
                Buffer.MemoryCopy(x, p, (nuint)(inCh * muLen * sizeof(float)), (nuint)(inCh * muLen * sizeof(float)));
        }
        return result;
    }

    /// <summary>
    /// DiT forward pass: in_proj → (8×) transformer blocks → out_proj.
    /// TODO(phase 5): the 8-block transformer with AdaLN (timestep + style)
    /// conditioning. For now this applies the in/out projections and a simple
    /// per-step estimate so the solver terminates; the full DiT is the heavy
    /// piece of the s2mel.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void DiTEstimator(float* x, float* promptX, float* mu, int muLen,
        float t, float* style, int styleDim, int inCh, float* dst)
    {
        int dim = _cfg.ModelDim;
        float* hidden = (float*)_scratch.C((nuint)(muLen * dim * sizeof(float)));
        // in_proj: (inCh, muLen) → (muLen, inCh) → (muLen, dim)
        for (int f = 0; f < muLen; f++)
        {
            for (int d = 0; d < dim; d++)
            {
                float acc = 0f;
                for (int c = 0; c < inCh; c++)
                    acc += x[c * muLen + f] * _cfmInProjW[d * inCh + c];
                acc += _cfmInProjB != null ? _cfmInProjB[d] : 0f;
                hidden[f * dim + d] = acc;
            }
        }
        // (transformer + wavenet skipped — TODO(phase 5))
        // out_proj: (muLen, dim) → (muLen, inCh)
        for (int f = 0; f < muLen; f++)
        {
            for (int c = 0; c < inCh; c++)
            {
                float acc = 0f;
                for (int d = 0; d < dim; d++)
                    acc += hidden[f * dim + d] * _cfmOutProjW[c * dim + d];
                acc += _cfmOutProjB != null ? _cfmOutProjB[c] : 0f;
                dst[c * muLen + f] = acc;
            }
        }
    }

    private float Gaus()
    {
        // Box-Muller with a bounded rejection for NaN safety.
        float u1 = 1f - (float)_rng.NextDouble();
        float u2 = 1f - (float)_rng.NextDouble();
        if (u1 < 1e-6f) u1 = 1e-6f;
        float mag = MathF.Sqrt(-2f * MathF.Log(u1));
        return mag * MathF.Cos(2f * 3.14159265358979323846f * u2);
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
}
