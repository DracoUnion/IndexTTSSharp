using Flygon.IndexTTSSharp.Gguf;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// Hyperparameters for the IndexTTS v2 pipeline, read from GGUF metadata with
/// sensible defaults matching the reference config. Missing keys fall back to
/// defaults so a partially-specified GGUF still loads.
/// </summary>
public sealed class IndexTTSConfig
{
    // --- GPT / UnifiedVoice ---
    public int GptLayers { get; init; } = 8;
    public int ModelDim { get; init; } = 512;
    public int GptHeads { get; init; } = 8;
    public int GptKvCacheLen { get; init; } = 1024;
    public int MaxTextTokens { get; init; } = 120;
    public int MaxMelTokens { get; init; } = 250;
    public int NumberTextTokens { get; init; } = 256;
    public int NumberMelCodes { get; init; } = 8194;
    public int StartMelToken { get; init; } = 8192;
    public int StopMelToken { get; init; } = 8193;
    public int StartTextToken { get; init; } = 0;
    public int StopTextToken { get; init; } = 1;
    public int ConditionNumLatent { get; init; } = 32;
    public string SpkCondMode { get; init; } = "conformer";
    public string ConditionType { get; init; } = "conformer_perceiver";

    // --- Conformer (spk + emo conditioning) ---
    public int ConformerOutputSize { get; init; } = 512;
    public int ConformerLinearUnits { get; init; } = 2048;
    public int ConformerAttentionHeads { get; init; } = 8;
    public int ConformerNumBlocks { get; init; } = 6;
    public string ConformerInputLayer { get; init; } = "conv2d";

    // --- Perceiver ---
    public int PerceiverMult { get; init; } = 4;
    public int PerceiverHeads { get; init; } = 8;

    // --- Semantic codec (RepCodec / Vocos) ---
    public int CodebookSize { get; init; } = 8192;
    public int CodebookDim { get; init; } = 8;
    public int CodecHiddenSize { get; init; } = 1024;
    public int VocosDim { get; init; } = 384;
    public int VocosIntermediateDim { get; init; } = 2048;
    public int VocosNumLayers { get; init; } = 12;
    public int NumQuantizers { get; init; } = 1;

    // --- s2mel (DiT / wavenet / length regulator) ---
    public int DitDepth { get; init; } = 8;
    public int DitNumHeads { get; init; } = 8;
    public int DitHiddenDim { get; init; } = 512;
    public int DitInChannels { get; init; } = 80;
    public string DitFinalLayerType { get; init; } = "wavenet";
    public bool DitStyleCondition { get; init; } = true;
    public bool LongSkipConnection { get; init; } = true;
    public int WavenetHiddenDim { get; init; } = 256;
    public int WavenetNumLayers { get; init; } = 4;
    public int WavenetKernelSize { get; init; } = 3;
    public int WavenetDilationRate { get; init; } = 2;
    public int StyleDim { get; init; } = 192;
    public int ContentDim { get; init; } = 1024;
    public int LengthRegulatorChannels { get; init; } = 512;
    public int[] LengthRegulatorSamplingRatios { get; init; } = [];
    public int NCodebooks { get; init; } = 3;

    // --- CAMPPlus ---
    public int CampplusFeatDim { get; init; } = 80;
    public int CampplusEmbeddingSize { get; init; } = 192;

    // --- w2v-bert ---
    public int W2vHiddenStatesIndex { get; init; } = 17;

    // --- mel / audio ---
    public int Sr { get; init; } = 22050;
    public int NMel { get; init; } = 80;
    public int NfFt { get; init; } = 1024;
    public int HopLength { get; init; } = 256;
    public int WinLength { get; init; } = 1024;

    /// <summary>Read config from GGUF metadata (top-level <c>gpt.*</c>, <c>s2mel.*</c>, …).</summary>
    public static IndexTTSConfig FromGguf(GgufFile gguf)
    {
        int G(string key, int dflt) => gguf.GetInt32(key, dflt);
        float F(string key, float dflt) => gguf.GetFloat32(key, dflt);
        string S(string key, string dflt) => gguf.GetString(key, dflt);

        int[] ratios = gguf.GetInt32Array("s2mel.length_regulator.sampling_ratios");

        return new IndexTTSConfig
        {
            GptLayers = G("gpt.layers", 8),
            GptKvCacheLen = G("gpt.kv_cache_len", 1024),
            ModelDim = G("gpt.model_dim", 512),
            GptHeads = G("gpt.heads", 8),
            MaxTextTokens = G("gpt.max_text_tokens", 120),
            MaxMelTokens = G("gpt.max_mel_tokens", 250),
            NumberTextTokens = G("gpt.number_text_tokens", 256),
            NumberMelCodes = G("gpt.number_mel_codes", 8194),
            StartMelToken = G("gpt.start_mel_token", 8192),
            StopMelToken = G("gpt.stop_mel_token", 8193),
            ConditionNumLatent = G("gpt.condition_num_latent", 32),
            SpkCondMode = S("gpt.spk_cond_mode", "conformer"),
            ConditionType = S("gpt.condition_type", "conformer_perceiver"),

            ConformerOutputSize = G("gpt.condition_module.output_size", 512),
            ConformerLinearUnits = G("gpt.condition_module.linear_units", 2048),
            ConformerAttentionHeads = G("gpt.condition_module.attention_heads", 8),
            ConformerNumBlocks = G("gpt.condition_module.num_blocks", 6),
            ConformerInputLayer = S("gpt.condition_module.input_layer", "conv2d"),

            PerceiverMult = G("gpt.condition_module.perceiver_mult", 4),
            PerceiverHeads = G("gpt.condition_module.attention_heads", 8),

            CodebookSize = G("semantic_codec.codebook_size", 8192),
            CodebookDim = G("semantic_codec.codebook_dim", 8),
            CodecHiddenSize = G("semantic_codec.hidden_size", 1024),
            VocosDim = G("semantic_codec.vocos_dim", 384),
            VocosIntermediateDim = G("semantic_codec.vocos_intermediate_dim", 2048),
            VocosNumLayers = G("semantic_codec.vocos_num_layers", 12),
            NumQuantizers = G("semantic_codec.num_quantizers", 1),

            DitDepth = G("s2mel.DiT.depth", 8),
            DitNumHeads = G("s2mel.DiT.num_heads", 8),
            DitHiddenDim = G("s2mel.DiT.hidden_dim", 512),
            DitInChannels = G("s2mel.DiT.in_channels", 80),
            DitFinalLayerType = S("s2mel.DiT.final_layer_type", "wavenet"),
            DitStyleCondition = G("s2mel.DiT.style_condition", 1) != 0,
            LongSkipConnection = G("s2mel.DiT.long_skip_connection", 1) != 0,
            WavenetHiddenDim = G("s2mel.wavenet.hidden_dim", 256),
            WavenetNumLayers = G("s2mel.wavenet.num_layers", 4),
            WavenetKernelSize = G("s2mel.wavenet.kernel_size", 3),
            WavenetDilationRate = G("s2mel.wavenet.dilation_rate", 2),
            StyleDim = G("s2mel.style_encoder.dim", 192),
            ContentDim = G("s2mel.DiT.content_dim", 1024),
            LengthRegulatorChannels = G("s2mel.length_regulator.channels", 512),
            LengthRegulatorSamplingRatios = ratios.Length > 0 ? ratios : [],
            NCodebooks = G("s2mel.length_regulator.n_codebooks", 3),

            CampplusFeatDim = G("campplus.feat_dim", 80),
            CampplusEmbeddingSize = G("campplus.embedding_size", 192),

            W2vHiddenStatesIndex = G("w2v_bert.hidden_states_index", 17),

            Sr = G("s2mel.preprocess_params.sr", 22050),
            NMel = G("s2mel.preprocess_params.spect_params.n_mels", 80),
            NfFt = G("s2mel.preprocess_params.spect_params.n_fft", 1024),
            HopLength = G("s2mel.preprocess_params.spect_params.hop_length", 256),
            WinLength = G("s2mel.preprocess_params.spect_params.win_length", 1024),
        };
    }
}
