using System.Runtime.CompilerServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// BigVGAN (from <c>indextts/s2mel/modules/bigvgan/bigvgan.py</c>). Mel
/// spectrogram <c>(80, T)</c> → waveform <c>(T · 2 · 4 · 8 · 8)</c> samples
/// via a stack of 10 anti-aliased conv-transpose / resblock layers.
///
/// Tensor naming (GGUF torch-module path, from the BigVGAN checkpoint):
///   bigvgan.0.weight|bias  bigvgan.{i}.0.*  bigvgan.{i}.1.*  bigvgan.{i}.2.*
///
/// A full BigVGAN port is a large conv stack; this class exposes the
/// sampling-ratio interface and a linear-interpolation resampling stub
/// (TODO(phase 6)) that returns a zero-filled waveform of the correct length.
/// </summary>
public unsafe sealed class BigVgan
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly IndexTTSConfig _cfg;
    private readonly ScratchArena _scratch = new();

    private readonly float* _convW;
    private readonly float* _convB;

    public int SampleRate => _cfg.Sr;
    public int HopLength => _cfg.HopLength;
    public int NMel => _cfg.NMel;

    public BigVgan(WeightStore w, CpuThreadPool? pool, IndexTTSConfig cfg)
    {
        _w = w;
        _pool = pool;
        _cfg = cfg;
        _convW = w.TryGetFloat("bigvgan.0.weight", out _);
        _convB = w.TryGetFloat("bigvgan.0.bias", out _);
    }

    /// <summary>
    /// Mel spectrogram <c>(80, T)</c> → 16-bit-PCM-equivalent float waveform
    /// in <c>[-1, 1]</c>. The full conv-transpose stack is a TODO(phase 6);
    /// this currently returns a zero waveform of the correct length so the
    /// pipeline plumbing is exercised end-to-end.
    /// </summary>
    public float[] MelToWave(float* mel, int melFrames)
    {
        int outLen = melFrames * HopLength;
        float* dst = (float*)_scratch.A((nuint)(outLen * sizeof(float)));
        // TODO(phase 6): 10-layer anti-aliased conv-transpose stack.
        for (int i = 0; i < outLen; i++)
            dst[i] = 0f;
        float[] result = new float[outLen];
        if (outLen > 0)
        {
            fixed (float* p = result)
                Buffer.MemoryCopy(dst, p, (nuint)(outLen * sizeof(float)), (nuint)(outLen * sizeof(float)));
        }
        return result;
    }
}
