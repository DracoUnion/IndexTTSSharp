using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// ConformerEncoder (EFSM2-style, from <c>indextts/gpt/conformer_encoder.py</c>)
/// — the spk/emo conditioning path. Input is a (T, 1024) w2v-bert feature
/// stream; output is a (T/4, outDim) encoded stream (Conv2dSubsampling4
/// default). Tensor names follow the GGUF torch-module path:
/// <c>conformer.embed.*</c> (subsampling conv + linear), <c>conformer.encoders.{i}.*</c>
/// (rel-pos MHA + depthwise conv + FFN + macaron), <c>conformer.after_norm.*</c>.
/// All linear/conv weights come through <see cref="WeightStore"/>.
/// </summary>
public unsafe sealed class ConformerEncoder
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly ScratchArena _scratch = new();

    public int InputSize { get; }
    public int OutDim { get; }
    public int Heads { get; }
    public int LinearUnits { get; }
    public int NumBlocks { get; }
    public string InputLayer { get; }
    public int SubsampleRate { get; }

    // Rel-pos attention
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _out;
    private float* _rel;

    public ConformerEncoder(WeightStore w, CpuThreadPool? pool, int inputSize, int outDim, int heads,
        int linearUnits, int numBlocks, string inputLayer)
    {
        _w = w;
        _pool = pool;
        InputSize = inputSize;
        OutDim = outDim;
        Heads = heads;
        LinearUnits = linearUnits;
        NumBlocks = numBlocks;
        InputLayer = inputLayer;
        SubsampleRate = inputLayer switch
        {
            "conv2d2" => 2,
            "conv2d" => 4,
            "conv2d6" => 6,
            "conv2d8" => 8,
            "linear" => 1,
            _ => 4,
        };

        int d = OutDim;
        _q = _w.GetFloat($"conformer.encoders.0.self_attn.linear_q.weight", out _);
        _k = _w.GetFloat($"conformer.encoders.0.self_attn.linear_k.weight", out _);
        _v = _w.GetFloat($"conformer.encoders.0.self_attn.linear_v.weight", out _);
        _out = _w.GetFloat($"conformer.encoders.0.self_attn.linear_out.weight", out _);
        _rel = _w.GetFloat($"conformer.encoders.0.self_attn.rel_k.weight", out _);
    }

    /// <summary>
    /// Run the encoder: (T, 1024) input → (T/subsample, OutDim) output, written
    /// into <paramref name="dst"/> (owned by the caller, ≥ outLen*OutDim floats).
    /// Returns <c>outLen</c>. Placeholder body until the tensor names are
    /// confirmed against a real GGUF; subsampling+rel-pos conformer is the
    /// heaviest conditioning path.
    /// </summary>
    public int Forward(float* input, int t, float* dst, out int outLen)
    {
        outLen = t / SubsampleRate;
        // TODO: conv2d subsampling → N rel-pos Conformer blocks → after_norm.
        for (int i = 0; i < outLen * OutDim; i++)
            dst[i] = 0f;
        return outLen;
    }
}
