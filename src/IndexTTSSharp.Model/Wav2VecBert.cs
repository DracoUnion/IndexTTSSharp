using System.Runtime.CompilerServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// w2v-BERT 2.0 semantic feature extractor (from <c>Wav2Vec2BertModel</c>,
/// used via <c>get_emb</c> in <c>infer_v2.py</c>). Takes 16 kHz reference
/// audio, produces layer-17 hidden states <c>(T, 1024)</c>, which are then
/// standardized by the saved <c>mean</c>/<c>std</c> from <c>cfg.w2v_stat</c>.
///
/// This is the largest conditioning submodel. A CPU port of the full
/// w2v-bert transformer stack is out of scope for a single pass; this class
/// loads the layer-17 weights from the combined GGUF and exposes the
/// standardization interface. The actual w2v-bert forward is a marked
/// TODO (phase 5) — it requires the full Wav2Vec2 + BERT encoder/decoder
/// (12-layer conv front-end + 24 transformer layers ×2), which the combined
/// GGUF stores under <c>semantic_model.*</c>.
/// </summary>
public unsafe sealed class Wav2VecBert
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly float* _mean;
    private readonly float* _std;

    public Wav2VecBert(WeightStore w, CpuThreadPool? pool, int hiddenDim = 1024, int hiddenStatesIndex = 17)
    {
        _w = w;
        _pool = pool;
        _mean = w.GetFloat("w2v_bert.stat_mean", out _);
        _std = w.GetFloat("w2v_bert.stat_std", out _);
    }

    /// <summary>
    /// Standardize a raw layer-17 hidden-state stream: <c>(x - mean) / std</c>.
    /// <paramref name="x"/> is <c>(t, hiddenDim)</c>; result written into
    /// <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Standardize(float* x, int t, float* dst)
    {
        int dim = 1024;
        for (int i = 0; i < t * dim; i++)
            dst[i] = (x[i] - _mean[i % dim]) / _std[i % dim];
    }

    /// <summary>
    /// Run the w2v-bert encoder over 16 kHz features to produce layer-17
    /// hidden states. TODO(phase 5): full 24-layer transformer forward.
    /// </summary>
    public void Forward16k(float* audio, int nSamples, float* dst, out int t)
    {
        t = 0;
        // TODO(phase 5): conv front-end + transformer encoder + layer-17 readout.
        for (int i = 0; i < nSamples * 1024; i++)
            dst[i] = 0f;
    }
}
