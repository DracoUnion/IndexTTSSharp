using System.Runtime.CompilerServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// GPT-2 transformer stack (from <c>indextts/gpt/model_v2.py</c> +
/// <c>transformers_gpt2.py</c>). 8 layers, dim 512, 8 heads. Uses F32 GEMM via
/// <see cref="MatmulF32"/>.
///
/// Tensor naming follows the GGUF torch-module path:
///   gpt.h.{i}.ln_1.*  gpt.h.{i}.attn.c_attn.*  gpt.h.{i}.attn.c_proj.*
///   gpt.h.{i}.ln_2.*  gpt.h.{i}.mlp.c_fc.*     gpt.h.{i}.mlp.c_proj.*
///   gpt.ln_f.*  text_embedding.weight  mel_embedding.weight
///   mel_pos_embedding.emb.weight  text_pos_embedding.emb.weight
///   emo_layer.*  emovec_layer.*  speed_emb.weight
/// </summary>
public unsafe sealed class Gpt : IDisposable
{
    private readonly WeightStore _w;
    private readonly CpuThreadPool? _pool;
    private readonly IndexTTSConfig _cfg;

    // per-layer weight arrays (indexed by layer)
    private float* _lnF;
    private float* _lnFB;
    private float* _lmHead;   // [8194, dim]
    private float* _lmHeadB;

    // embeddings
    private readonly float* _textEmb;  // [257, dim]
    private readonly float* _melEmb;   // [8194, dim]
    private readonly float* _melPos;   // [max_mel+2+cond, dim]
    private readonly float* _textPos;  // [max_text+2, dim]
    private readonly float* _emoLayer;
    private readonly float* _emoVecLayer;
    private readonly float* _speedEmb;

    // scratch for attention
    private readonly ScratchArena _scratch = new();

    public Gpt(WeightStore w, CpuThreadPool? pool, IndexTTSConfig cfg)
    {
        _w = w;
        _pool = pool;
        _cfg = cfg;
        int layers = cfg.GptLayers;
        _layerLn1 = new float*[layers];
        _layerLn1B = new float*[layers];
        _layerCAttn = new float*[layers];
        _layerCAttnB = new float*[layers];
        _layerCProj = new float*[layers];
        _layerCProjB = new float*[layers];
        _layerLn2 = new float*[layers];
        _layerLn2B = new float*[layers];
        _layerCFc = new float*[layers];
        _layerCFcB = new float*[layers];
        _layerMlpProj = new float*[layers];
        _layerMlpProjB = new float*[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"gpt.h.{i}.";
            _layerLn1[i] = w.GetFloat(p + "ln_1.weight", out _);
            _layerLn1B[i] = w.TryGetFloat(p + "ln_1.bias", out _);
            _layerCAttn[i] = w.GetFloat(p + "attn.c_attn.weight", out _);
            _layerCAttnB[i] = w.TryGetFloat(p + "attn.c_attn.bias", out _);
            _layerCProj[i] = w.GetFloat(p + "attn.c_proj.weight", out _);
            _layerCProjB[i] = w.TryGetFloat(p + "attn.c_proj.bias", out _);
            _layerLn2[i] = w.GetFloat(p + "ln_2.weight", out _);
            _layerLn2B[i] = w.TryGetFloat(p + "ln_2.bias", out _);
            _layerCFc[i] = w.GetFloat(p + "mlp.c_fc.weight", out _);
            _layerCFcB[i] = w.TryGetFloat(p + "mlp.c_fc.bias", out _);
            _layerMlpProj[i] = w.GetFloat(p + "mlp.c_proj.weight", out _);
            _layerMlpProjB[i] = w.TryGetFloat(p + "mlp.c_proj.bias", out _);
        }
        _lnF = w.GetFloat("gpt.ln_f.weight", out _);
        _lnFB = w.TryGetFloat("gpt.ln_f.bias", out _);
        _lmHead = w.GetFloat("gpt.lm_head.weight", out _);
        _lmHeadB = w.TryGetFloat("gpt.lm_head.bias", out _);
        _textEmb = w.GetFloat("text_embedding.weight", out _);
        _melEmb = w.GetFloat("mel_embedding.weight", out _);
        _melPos = w.GetFloat("mel_pos_embedding.emb.weight", out _);
        _textPos = w.GetFloat("text_pos_embedding.emb.weight", out _);
        _emoLayer = w.GetFloat("emo_layer.weight", out _);
        _emoVecLayer = w.GetFloat("emovec_layer.weight", out _);
        _speedEmb = w.GetFloat("speed_emb.weight", out _);
    }

    public int Dim => _cfg.ModelDim;
    public int Heads => _cfg.GptHeads;
    public int HeadDim => _cfg.ModelDim / _cfg.GptHeads;
    public int NumberMelCodes => _cfg.NumberMelCodes;
    public int StopMelToken => _cfg.StopMelToken;

    /// <summary>
    /// Look up a single embedding row by token id (used by generation to embed
    /// one new mel token at a time).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EmbedMel(int token, float* dst)
    {
        int dim = _cfg.ModelDim;
        float* row = _melEmb + token * dim;
        for (int i = 0; i < dim; i++)
            dst[i] = row[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EmbedText(int token, int pos, float* dst)
    {
        int dim = _cfg.ModelDim;
        float* textRow = _textEmb + token * dim;
        float* posRow = _textPos + pos * dim;
        for (int i = 0; i < dim; i++)
            dst[i] = textRow[i] + posRow[i];
    }

    /// <summary>
    /// One forward pass over the embedding stream (prefill). Computes the GPT
    /// transformer over <paramref name="seq"/> input rows and writes the final
    /// hidden state (after <c>ln_f</c>) into <paramref name="hiddenOut"/>.
    /// </summary>
    public void Prefill(float* inputEmbs, int seq, float* hiddenOut)
    {
        int dim = _cfg.ModelDim;
        int layers = _cfg.GptLayers;
        int heads = _cfg.GptHeads;
        int hDim = _cfg.ModelDim / _cfg.GptHeads;

        // Working buffer for the sequence being transformed.
        float* h = (float*)_scratch.A((nuint)(seq * dim * sizeof(float)));
        Buffer.MemoryCopy(inputEmbs, h, (nuint)(seq * dim * sizeof(float)), (nuint)(seq * dim * sizeof(float)));

        // qkv for current layer: [seq, 3*dim]. We compute in place into B.
        float* qkv = (float*)_scratch.B((nuint)(seq * 3 * dim * sizeof(float)));
        float* attnOut = (float*)_scratch.C((nuint)(seq * dim * sizeof(float)));
        float* scores = (float*)_scratch.D((nuint)(heads * seq * seq * sizeof(float)));
        float* ffn = (float*)_scratch.E((nuint)(seq * 4 * dim * sizeof(float)));

        float scale = 1f / MathF.Sqrt(hDim);
        for (int l = 0; l < layers; l++)
        {
            // RMSNorm-style: GPT2 uses LayerNorm (eps=1e-5) in PyTorch transformers.
            // The HF GPT2 blocks use LAYERnorm with the layer's ln_1, ln_2.
            Norms.LayerNorm(h, GetLn1(l), GetLn1B(l), seq, dim, 1e-5f, _pool);

            // qkv = h · cAttnᵀ + cAttnB  → [seq, 3*dim]
            float* cAttn = GetCAttn(l);
            float* cAttnB = GetCAttnB(l);
            MatmulF32.Gemm(h, cAttn, qkv, seq, dim, 3 * dim, accumulate: false, _pool);
            AddBias(qkv, cAttnB, seq, 3 * dim, addBias: true);

            // split qkv into q,k,v and run attention (causal, GPT2).
            RunAttention(qkv, attnOut, scores, seq, heads, hDim, scale, GetCProj(l), GetCProjB(l));

            // residual: h += attnOut
            AddInPlace(h, attnOut, seq * dim);

            // ln_2
            Norms.LayerNorm(h, GetLn2(l), GetLn2B(l), seq, dim, 1e-5f, _pool);

            // ffn: [seq, 4*dim] = GELU(h · cFcᵀ + cFcB)
            float* cFc = GetCFc(l);
            float* cFcB = GetCFcB(l);
            MatmulF32.Gemm(h, cFc, ffn, seq, dim, 4 * dim, accumulate: false, _pool);
            AddBias(ffn, cFcB, seq, 4 * dim, addBias: true);
            GELU(ffn, seq * 4 * dim);

            // out = GELU(ffn) · c_projᵀ + c_projB
            float* cProj = GetMlpProj(l);
            float* cProjB = GetMlpProjB(l);
            float* mlpOut = (float*)_scratch.F((nuint)(seq * dim * sizeof(float)));
            MatmulF32.Gemm(ffn, cProj, mlpOut, seq, 4 * dim, dim, accumulate: false, _pool);
            AddBias(mlpOut, cProjB, seq, dim, addBias: true);
            AddInPlace(h, mlpOut, seq * dim);
        }

        Norms.LayerNorm(h, _lnF, _lnFB, seq, dim, 1e-5f, _pool);
        if (hiddenOut != null)
            Buffer.MemoryCopy(h, hiddenOut, (nuint)(seq * dim * sizeof(float)), (nuint)(seq * dim * sizeof(float)));
    }

    /// <summary>
    /// Single-token GPT forward pass over a KV cache. Writes the final hidden
    /// state for the new token into <paramref name="lastOut"/>.
    /// </summary>
    public void Decode(float* tokenEmb, float* kCache, float* vCache, int kvLen, float* lastOut, float* logits)
    {
        int dim = _cfg.ModelDim;
        int layers = _cfg.GptLayers;
        int heads = _cfg.GptHeads;
        int hDim = _cfg.ModelDim / _cfg.GptHeads;

        // h starts as the single new embedding.
        float* h = (float*)_scratch.A((nuint)(dim * sizeof(float)));
        Buffer.MemoryCopy(tokenEmb, h, (nuint)(dim * sizeof(float)), (nuint)(dim * sizeof(float)));

        float* qkv = (float*)_scratch.B((nuint)(3 * dim * sizeof(float)));
        float* attnOut = (float*)_scratch.C((nuint)(dim * sizeof(float)));
        float* ffn = (float*)_scratch.D((nuint)(4 * dim * sizeof(float)));
        float* mlpOut = (float*)_scratch.E((nuint)(dim * sizeof(float)));
        float* scores = (float*)_scratch.F((nuint)(heads * kvLen * sizeof(float)));

        float scale = 1f / MathF.Sqrt(hDim);
        for (int l = 0; l < layers; l++)
        {
            Norms.LayerNorm(h, GetLn1(l), GetLn1B(l), 1, dim, 1e-5f, _pool);

            float* cAttn = GetCAttn(l);
            float* cAttnB = GetCAttnB(l);
            MatmulF32.Gemv(h, cAttn, qkv, dim, 3 * dim, addBias: true, cAttnB, _pool);

            // split q,k,v for this single query.
            float* q = qkv;
            float* kNew = qkv + dim;
            float* vNew = qkv + 2 * dim;

            // Write this new K,V into the cache at position kvLen.
            float* kl = kCache + l * _cfg.GptKvCacheLen * dim;
            float* vl = vCache + l * _cfg.GptKvCacheLen * dim;
            Buffer.MemoryCopy(kNew, kl + kvLen * dim, (nuint)(dim * sizeof(float)), (nuint)(dim * sizeof(float)));
            Buffer.MemoryCopy(vNew, vl + kvLen * dim, (nuint)(dim * sizeof(float)), (nuint)(dim * sizeof(float)));

            // Causal attention against [kl .. kl+kvLen+1] (the +1 is the token we just appended).
            RunDecodeAttention(q, vl, kl, attnOut, scores, kvLen + 1, heads, hDim, scale, GetCProj(l), GetCProjB(l));

            AddInPlace(h, attnOut, dim);

            Norms.LayerNorm(h, GetLn2(l), GetLn2B(l), 1, dim, 1e-5f, _pool);

            float* cFc = GetCFc(l);
            float* cFcB = GetCFcB(l);
            MatmulF32.Gemv(h, cFc, ffn, dim, 4 * dim, addBias: true, cFcB, _pool);
            GELU(ffn, 4 * dim);

            float* cProj = GetMlpProj(l);
            float* cProjB = GetMlpProjB(l);
            MatmulF32.Gemv(ffn, cProj, mlpOut, 4 * dim, dim, addBias: true, cProjB, _pool);
            AddInPlace(h, mlpOut, dim);
        }

        Norms.LayerNorm(h, _lnF, _lnFB, 1, dim, 1e-5f, _pool);
        if (lastOut != null)
            Buffer.MemoryCopy(h, lastOut, (nuint)(dim * sizeof(float)), (nuint)(dim * sizeof(float)));

        // logits = h · lmHeadᵀ + lmHeadB
        int codes = _cfg.NumberMelCodes;
        MatmulF32.Gemv(h, _lmHead, logits, dim, codes, addBias: true, _lmHeadB, _pool);
    }

    /// <summary>Attention over a full KV cache. K/V already include the new token.</summary>
    private void RunDecodeAttention(
        float* q, float* v, float* k,
        float* attnOut, float* scores,
        int kvLen, int heads, int hDim, float scale,
        float* cProj, float* cProjB)
    {
        int dim = _cfg.ModelDim;
        // q is [hDim] per head for this one token.
        float* qHead = q;
        float* outHead = attnOut;
        for (int h = 0; h < heads; h++)
        {
            float* kq = k + h * hDim;
            for (int t = 0; t < kvLen; t++)
            {
                float acc = 0f;
                for (int d = 0; d < hDim; d++)
                    acc += qHead[d] * kq[d];
                scores[h * kvLen + t] = acc * scale;
                kq += dim;
            }
            Norms.Softmax(scores + h * kvLen, 1, kvLen);
            float* outRow = outHead + h * hDim;
            for (int d = 0; d < hDim; d++)
            {
                float acc = 0f;
                float* vRow = v + h * hDim + d;
                for (int t = 0; t < kvLen; t++)
                    acc += scores[h * kvLen + t] * vRow[t * dim];
                outRow[d] = acc;
            }
        }
        // c_proj + bias
        MatmulF32.Gemv(attnOut, cProj, attnOut, dim, dim, addBias: true, cProjB, _pool);
    }

    private void RunAttention(
        float* qkv, float* attnOut, float* scores,
        int seq, int heads, int hDim, float scale,
        float* cProj, float* cProjB)
    {
        int dim = _cfg.ModelDim;
        float* q = qkv;                 // [seq, dim] (q part)
        float* k = qkv + dim;           // [seq, dim]
        float* v = qkv + 2 * dim;       // [seq, dim]
        // scores[h, i, j] = q[i]·k[j]·scale for j<=i (causal)
        float* s = scores;
        int perHead = seq * seq;
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * hDim;
            for (int i = 0; i < seq; i++)
            {
                float* qRow = q + i * dim + hOff;
                float* sRow = s + (h * seq + i) * seq;
                // causal: j <= i
                for (int j = 0; j < seq; j++)
                {
                    if (j > i)
                    {
                        sRow[j] = float.MinValue;
                        continue;
                    }
                    float* kRow = k + j * dim + hOff;
                    float acc = 0f;
                    for (int d = 0; d < hDim; d++)
                        acc += qRow[d] * kRow[d];
                    sRow[j] = acc * scale;
                }
            }
            // softmax per row (over the full seq, causal entries are -inf)
            for (int i = 0; i < seq; i++)
            {
                float* sRow = s + (h * seq + i) * seq;
                float mx = -80f;
                for (int j = 0; j < seq; j++)
                    if (sRow[j] > mx) mx = sRow[j];
                float sum = 0f;
                for (int j = 0; j < seq; j++)
                {
                    sRow[j] = MathF.Exp(sRow[j] - mx);
                    sum += sRow[j];
                }
                float inv = sum > 0f ? 1f / sum : 0f;
                for (int j = 0; j < seq; j++)
                    sRow[j] *= inv;
            }
            // out[i, h] = Σ_j sRow[i,j] · v[j, h]
            for (int i = 0; i < seq; i++)
            {
                float* sRow = s + (h * seq + i) * seq;
                float* outRow = attnOut + i * dim + hOff;
                for (int d = 0; d < hDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < seq; j++)
                        acc += sRow[j] * v[j * dim + hOff + d];
                    outRow[d] = acc;
                }
            }
        }
        // c_proj + bias
        MatmulF32.Gemm(attnOut, cProj, attnOut, seq, dim, dim, accumulate: false, _pool);
        AddBias(attnOut, cProjB, seq, dim, addBias: true);
    }

    private static void AddInPlace(float* a, float* b, int n)
    {
        for (int i = 0; i < n; i++)
            a[i] += b[i];
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
            // tanh-approx GELU
            float t = MathF.Tanh(v * 0.7978845608f + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + t);
        }
    }

    // Layer accessors (indices into the 8 stored sets).
    private readonly float*[] _layerLn1;
    private readonly float*[] _layerLn1B;
    private readonly float*[] _layerCAttn;
    private readonly float*[] _layerCAttnB;
    private readonly float*[] _layerCProj;
    private readonly float*[] _layerCProjB;
    private readonly float*[] _layerLn2;
    private readonly float*[] _layerLn2B;
    private readonly float*[] _layerCFc;
    private readonly float*[] _layerCFcB;
    private readonly float*[] _layerMlpProj;
    private readonly float*[] _layerMlpProjB;

    private float* GetLn1(int l) => _layerLn1[l];
    private float* GetLn1B(int l) => _layerLn1B[l];
    private float* GetCAttn(int l) => _layerCAttn[l];
    private float* GetCAttnB(int l) => _layerCAttnB[l];
    private float* GetCProj(int l) => _layerCProj[l];
    private float* GetCProjB(int l) => _layerCProjB[l];
    private float* GetLn2(int l) => _layerLn2[l];
    private float* GetLn2B(int l) => _layerLn2B[l];
    private float* GetCFc(int l) => _layerCFc[l];
    private float* GetCFcB(int l) => _layerCFcB[l];
    private float* GetMlpProj(int l) => _layerMlpProj[l];
    private float* GetMlpProjB(int l) => _layerMlpProjB[l];

    public void Dispose() { }
}
