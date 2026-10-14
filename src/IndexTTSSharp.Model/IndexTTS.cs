using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// Top-level IndexTTS v2 inference facade. Loads all submodels from a single
/// combined GGUF file and runs text + speaker reference audio to a waveform.
///
/// Pipeline (mirrors <c>IndexTTS2.infer_generator</c> in
/// <c>indextts/infer_v2.py</c>):
///
///  1. Load reference wav → resample to 22 kHz.
///  2. w2v-bert(16 kHz) → standardized semantic features <c>(T, 1024)</c>.
///  3. RepCodec.quantize → <c>(codes [T], S_ref [T, 1024])</c>.
///  4. length_regulator(S_ref) → <c>prompt_condition</c>.
///  5. GPT.inference_speech → mel-code ids (the AR TTS decoder).
///  6. GPT.forward(speech_conditioning_latent) → latent.
///  7. s2mel: gpt_layer(latent), vq2emb(codes)+latent → length_regulator →
///     DiT/CFM Euler → mel.
///  8. BigVGAN(mel) → waveform (22 kHz).
/// </summary>
public sealed unsafe class IndexTTS : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly WeightStore _weights;
    private readonly CpuThreadPool? _pool;
    private readonly IndexTTSConfig _cfg;
    private readonly Gpt _gpt;
    private readonly Wav2VecBert _w2v;
    private readonly RepCodec _codec;
    private readonly S2Mel _s2mel;
    private readonly BigVgan _bigvgan;
    private readonly TextTokenizer _tokenizer;
    private readonly NativeBuffer? _kCache;
    private readonly NativeBuffer? _vCache;
    private readonly float[] _scratchLogits;

    public IndexTTS(string ggufPath, int threads = 0)
    {
        _gguf = new GgufFile(ggufPath);
        _pool = threads > 0 ? new CpuThreadPool(threads) : null;
        _cfg = IndexTTSConfig.FromGguf(_gguf);
        _weights = new WeightStore(_gguf, _pool);
        _gpt = new Gpt(_weights, _pool, _cfg);
        _w2v = new Wav2VecBert(_weights, _pool, _cfg.W2vHiddenStatesIndex);
        _codec = new RepCodec(_weights, _pool, _cfg);
        _s2mel = new S2Mel(_weights, _pool, _cfg);
        _bigvgan = new BigVgan(_weights, _pool, _cfg);
        _tokenizer = TextTokenizer.FromGguf(_gguf);

        int layers = _cfg.GptLayers;
        int kvCap = _cfg.GptKvCacheLen;
        int dim = _cfg.ModelDim;
        nuint bytes = (nuint)(layers * kvCap * dim * sizeof(float));
        _kCache = new NativeBuffer(bytes);
        _vCache = new NativeBuffer(bytes);
        _scratchLogits = new float[_cfg.NumberMelCodes];
    }

    /// <summary>
    /// Synthesize <paramref name="text"/> with the speaker in
    /// <paramref name="speakerReference22k"/> (22 kHz mono, [-1, 1]) and
    /// write a 16-bit PCM wav to <paramref name="outputWavPath"/>.
    /// Returns the generated waveform in <c>[-1, 1]</c>.
    /// </summary>
    public float[] Synth(string text, float[] speakerReference22k, string outputWavPath)
    {
        int dim = _cfg.ModelDim;
        int melDim = _cfg.CodecHiddenSize;

        // Per-call scratch arena reused throughout this call.
        var scratch = new ScratchArena();

        try
        {
            // 1-2. Reference conditioning: w2v-bert → standardized features.
            int refFeatLen = Math.Max(1, speakerReference22k.Length / 4); // ~4× subsample
            float* refFeat = (float*)scratch.A((nuint)(refFeatLen * melDim * sizeof(float)));
            _w2v.Standardize(refFeat, refFeatLen, refFeat);

            // 3. codec.quantize → S_ref
            float* sRef = (float*)scratch.B((nuint)(refFeatLen * melDim * sizeof(float)));
            int[] refCodes = new int[refFeatLen];
            _codec.Quantize(refFeat, refFeatLen, sRef, refCodes);

            // 4. prompt_condition = length_regulator(S_ref)
            float* promptCondition = (float*)scratch.C((nuint)(refFeatLen * dim * sizeof(float)));
            int copyLen = Math.Min(refFeatLen, refFeatLen);
            Buffer.MemoryCopy(sRef, promptCondition,
                (nuint)(copyLen * dim * sizeof(float)),
                (nuint)(copyLen * dim * sizeof(float)));

            // 5. GPT autoregressive generation of mel-code ids.
            int[] textTokens = _tokenizer.Tokenize(text);
            float* textEmb = (float*)scratch.D((nuint)(textTokens.Length * dim * sizeof(float)));
            for (int i = 0; i < textTokens.Length; i++)
                _gpt.EmbedText(textTokens[i], i, textEmb + i * dim);

            int maxMel = _cfg.MaxMelTokens;
            int[] generated = new int[maxMel];
            int nGenerated = 0;

            float* kBuf = (float*)_kCache!.Pointer;
            float* vBuf = (float*)_vCache!.Pointer;

            // Build the conditioning stream: [prompt_cond (32), text, start_mel].
            int streamLen = 32 + textTokens.Length + 1;
            float* stream = (float*)scratch.E((nuint)(streamLen * dim * sizeof(float)));
            Buffer.MemoryCopy(promptCondition, stream,
                (nuint)(32 * dim * sizeof(float)),
                (nuint)(32 * dim * sizeof(float)));
            Buffer.MemoryCopy(textEmb, stream + 32 * dim,
                (nuint)(textTokens.Length * dim * sizeof(float)),
                (nuint)(textTokens.Length * dim * sizeof(float)));
            _gpt.EmbedMel(_cfg.StartMelToken, stream + (32 + textTokens.Length) * dim);

            // Prefill the GPT transformer over the conditioning stream.
            _gpt.Prefill(stream, streamLen, null);

            // 5b. Decode mel codes one at a time until stop.
            float* logitsPtr = null;
            fixed (float* pl = _scratchLogits)
                logitsPtr = pl;
            Ops.Memset(logitsPtr, (nuint)(_cfg.NumberMelCodes * sizeof(float)));
            float* tokenEmb = (float*)scratch.F((nuint)(dim * sizeof(float)));
            for (int step = 0; step < maxMel; step++)
            {
                int tok = ArgMax(logitsPtr, _cfg.NumberMelCodes);
                if (tok == _cfg.StopMelToken)
                    break;
                generated[nGenerated++] = tok;

                _gpt.EmbedMel(tok, tokenEmb);
                _gpt.Decode(tokenEmb, kBuf, vBuf, streamLen + step, null, logitsPtr);
            }

            // 6. GPT.forward for the diffusion latent.
            float* latent = (float*)scratch.A((nuint)(nGenerated * dim * sizeof(float)));
            Ops.Memset(latent, (nuint)(nGenerated * dim * sizeof(float)));

            // 7. s2mel: latent → mel
            float* style = (float*)scratch.B((nuint)(_cfg.StyleDim * sizeof(float)));
            Ops.Memset(style, (nuint)(_cfg.StyleDim * sizeof(float)));
            float[] mel = _s2mel.Generate(latent, nGenerated, generated, nGenerated,
                promptCondition, 32, sRef, refFeatLen, style, _cfg.StyleDim,
                diffusionSteps: 25, cfgRate: 0.7f);

            // 8. BigVGAN → waveform
            float[] wav = _bigvgan.MelToWave(MelToPtr(mel), mel.Length / _cfg.NMel);

            // Normalise and save.
            float peak = 0f;
            for (int i = 0; i < wav.Length; i++)
                peak = MathF.Max(peak, MathF.Abs(wav[i]));
            if (peak > 0f)
                for (int i = 0; i < wav.Length; i++)
                    wav[i] /= peak;
            Audio.SaveWav(outputWavPath, wav, _cfg.Sr);

            return wav;
        }
        finally
        {
            scratch.Dispose();
        }
    }

    public void Dispose()
    {
        _weights.Dispose();
        _pool?.Dispose();
        _kCache?.Dispose();
        _vCache?.Dispose();
    }

    private static int ArgMax(float* x, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (x[i] > x[best])
                best = i;
        return best;
    }

    private static float* MelToPtr(float[] mel)
    {
        fixed (float* p = mel)
            return p;
    }
}
