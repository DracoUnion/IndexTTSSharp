# 04 · IndexTTSSharp.Model 模块详细设计

**版本**: 1.0 · **更新**: 2026-10-15 · **代码路径**: `src/IndexTTSSharp.Model/`

## 1. 模块职责

全部 TTS 子模型 + 顶层推理编排（`IndexTTS`）+ 音频前端（`Audio`）+ 配置（`IndexTTSConfig`）+ 权重存储（`WeightStore`）。本模块是项目的主体，向上对 `Cli` 暴露极简门面：`new IndexTTS(ggufPath, threads)` → `Synth(text, refAudio22k, outPath)`。

## 2. 文件清单

| 文件 | 角色 |
|------|------|
| `IndexTTS.cs` | 顶层门面 + 推理编排（§3） |
| `IndexTTSConfig.cs` | 超参数配置（§4） |
| `WeightStore.cs` | 权重加载 + 去量化缓存（§5） |
| `Gpt.cs` | GPT-2 自回归 TTS 解码器（§6） |
| `RepCodec.cs` | 语义编解码器 quantize/vq2emb（§7） |
| `S2Mel.cs` | mel 合成：gpt_layer + 长度调节器 + DiT/CFM Euler（§8） |
| `Wav2VecBert.cs` | w2v-bert 语义特征提取 stub（§9） |
| `BigVgan.cs` | BigVGAN 声码器 stub（§10） |
| `Perceiver.cs` | PerceiverResampler（§11） |
| `ConformerEncoder.cs` | Conformer 条件编码 stub（§12） |
| `TextTokenizer.cs` | SentencePiece BPE 分词器（§13） |
| `Audio.cs` | WAV 读取/写出、重采样、mel stub（§14） |
| `MatmulF32.cs` | F32 GEMM/GEMV（封装 `Kernels.MatmulF32`，同语义） |

> 说明：`Model` 命名空间内也有一份 `MatmulF32`，与 `Kernels` 中同名类并存，供非量化子模型直接引用；二者数值语义一致。

## 3. 顶层编排：`IndexTTS`

### 3.1 构造
```csharp
IndexTTS(string ggufPath, int threads = 0)
```
1. `new GgufFile(ggufPath)` 打开并解析 GGUF。
2. `threads > 0` 时创建 `CpuThreadPool`；否则 `null`（各内核退化为串行或查询默认拓扑）。
3. `IndexTTSConfig.FromGguf(_gguf)` 读配置。
4. 实例化所有子模型：`Gpt`、`Wav2VecBert`、`RepCodec`、`S2Mel`、`BigVgan`、`TextTokenizer`。
5. 分配 **KV cache**：K、V 各 `layers * kvCacheLen * dim * 4B`（默认 8·1024·512·4 = 16 MiB）`NativeBuffer`。
6. 分配 `_scratchLogits`（`[NumberMelCodes] = 8194`，decode 每步的 logits）。

### 3.2 `Synth` 全流程
调用链与缓冲区（`A..F` 为 `ScratchArena` 槽位）：

```
float[] Synth(text, speakerReference22k, outputWavPath)
  scratch = new ScratchArena()                       // 每调用一个实例，finally 释放
  ├─ refFeat  (A, refFeatLen·1024)  = _w2v.Standardize(refAudio 子采样)
  ├─ sRef     (B, refFeatLen·1024)  + refCodes[] = _codec.Quantize(refFeat)
  ├─ promptCondition (C, 32·512) = 拷贝 sRef 前 32 帧
  ├─ textTokens = _tokenizer.Tokenize(text)
  ├─ textEmb   (D, len·512)  .EmbedText(textTokens[i], i)
  ├─ stream    (E, (32+len+1)·512) = [promptCondition | textEmb | start_mel_emb]
  ├─ _gpt.Prefill(stream, streamLen, null)
  ├─ decode 循环（≤ MaxMelTokens=250）：
  │    tok = ArgMax(logits, 8194)；若 == StopMelToken(8193) 则停
  │    EmbedMel(tok) → _gpt.Decode(tokenEmb, kCache, vCache,
  │                        streamLen+step, null, logits)
  ├─ latent   (A, nGenerated·512) = 0
  ├─ style    (B, 192) = 0
  ├─ mel = _s2mel.Generate(latent, nGenerated, generated, nGenerated,
  │            promptCondition, 32, sRef, refFeatLen, style, 192,
  │            diffusionSteps: 25, cfgRate: 0.7f)
  ├─ wav = _bigvgan.MelToWave(mel, mel.Length/NMel)
  ├─ 归一化峰值 → Audio.SaveWav(outputWavPath, wav, Sr)
  └─ 返回 wav（float[]，[-1,1]）
```

当前实现的已知简化（对照参考 `infer_v2.py`）：
- **w2v-bert 前向未实现**（`Wav2VecBert.Forward16k` 是 TODO），`refFeat` 实际来自调用方提供的引用特征/占位数据路径；参考中步骤 2 的语义特征本应来自 w2v-bert。
- **GPT forward 条件 latent** 参考中由 `Gpt.forward(speech_conditioning_latent)` 输出，此处 `latent` 直接清零，由 `S2Mel.Generate` 内部路径衔接。
- **完整 DiT / BigVGAN** 未实现（stub 见 §8/§10）。

## 4. 配置：`IndexTTSConfig`

`init` 属性 + 默认值，`FromGguf` 从元数据读取（缺失回退默认）。字段分组：

| 分组 | 关键字段（默认值） |
|------|---------------------|
| GPT | `GptLayers=8, ModelDim=512, GptHeads=8, GptKvCacheLen=1024, MaxTextTokens=120, MaxMelTokens=250, NumberTextTokens=256, NumberMelCodes=8194, StartMelToken=8192, StopMelToken=8193, ConditionNumLatent=32, SpkCondMode="conformer", ConditionType="conformer_perceiver"` |
| Conformer | `ConformerOutputSize=512, LinearUnits=2048, AttentionHeads=8, NumBlocks=6, InputLayer="conv2d"` |
| Perceiver | `PerceiverMult=4, PerceiverHeads=8` |
| codec | `CodebookSize=8192, CodebookDim=8, CodecHiddenSize=1024, VocosDim=384, VocosNumLayers=12, NumQuantizers=1` |
| s2mel | `DitDepth=8, DitNumHeads=8, DitHiddenDim=512, DitInChannels=80, DitFinalLayerType="wavenet", StyleDim=192, ContentDim=1024, WavenetHiddenDim=256, WavenetNumLayers=4, LengthRegulatorChannels=512, NCodebooks=3` |
| w2v | `W2vHiddenStatesIndex=17` |
| mel/audio | `Sr=22050, NMel=80, NfFt=1024, HopLength=256, WinLength=1024` |

GGUF 的 key 形如 `gpt.layers`、`semantic_codec.codebook_size`、`s2mel.DiT.depth`、`s2mel.preprocess_params.spect_params.n_mels` 等（见 `FromGguf` 映射）。

## 5. 权重存储：`WeightStore`

### 5.1 职责
从单个 combined GGUF **按名称**加载张量，F32 直接拷贝、量化张量首次访问时**解量化到 F32**，结果缓存于 `_buffers`（`NativeBuffer`）。子模型通过 `GetFloat`/`TryGetFloat` 获得稳定的 `float*` 视图。

### 5.2 关键 API
```csharp
float* GetFloat(string name, out int[] shape)   // 必选；缺失 → KeyNotFoundException
float* TryGetFloat(string name, out int[] shape) // 可选；缺失记入 Unmapped，返回 null
bool Has(name) / GgufTensorInfo Info(name)
Unmapped : IReadOnlyList<string>
```

- 形状处理：GGUF `ulong[]` → `int[]`，空形状按 `[1]`；元素数 `checked((int))`。
- `Load`：F32 → `Buffer.MemoryCopy`；否则 `Dequantize(type, src, dst, count)`。
- `Dequantize` 支持：`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1`（内联逐块实现）、`F16`；其余类型 → `NotSupportedException`（TODO 扩展 K 系列等）。
- `Dispose`：释放所有 `NativeBuffer` + `GgufFile`。

### 5.3 去量化块布局
- Q4_0：每 32 个 float，`d(f16) + 16B 打包 4bit`，`val = (quarter − 8)·d`。
- Q4_1：`d + m + 16B`，`val = d·q + m`。
- Q5_0：`d + qh(1B) + 20B`，高 5 bit 进位用 `qh`。
- Q5_1：`d + m + qh(1B) + 20B`。
- Q8_0：`d(f16) + 32×int8`，`val = d·q`。
- Q8_1：`d + m + qs_sum(i16) + 32×int8`，`val = d·q + m`。

## 6. 子模型：`Gpt`

### 6.1 结构
GPT-2 decoder（`model_v2.py` + `transformers_gpt2.py`），8 层 pre-LN，dim 512，8 头。全部权重来自 `WeightStore`，按 GGUF 路径命名：

```
gpt.h.{i}.ln_1.* | attn.c_attn.* | attn.c_proj.* | ln_2.* | mlp.c_fc.* | mlp.c_proj.*
gpt.ln_f.*  gpt.lm_head.*  text_embedding.weight  mel_embedding.weight
mel_pos_embedding.emb.weight  text_pos_embedding.emb.weight
emo_layer.weight  emovec_layer.weight  speed_emb.weight
```

### 6.2 关键操作
- `EmbedText(token, pos, dst)`：`text_emb[token] + text_pos[pos]`。
- `EmbedMel(token, dst)`：`mel_emb[token]`。
- `Prefill(inputEmbs, seq, hiddenOut)`：整段条件流前向。
  - 逐层：`LayerNorm(ln_1)` → `QKV = h·c_attnᵀ + b` → `RunAttention`（因果）→ 残差 → `LayerNorm(ln_2)` → `GELU(h·c_fcᵀ+b)` → `mlpOut + 残差`；最后 `LayerNorm(ln_f)`。
  - `RunAttention`：分头 q/k/v，因果掩码（`j>i → -inf`），per-row softmax（数值稳定 max-shift），`out = Σ s·v`，再 `c_proj + bias`。
  - 并行：`MatmulF32.Gemm` 行并行；attention 内循环串行。
- `Decode(tokenEmb, kCache, vCache, kvLen, lastOut, logits)`：单 token 前向。
  - qkv 单 token；新 K/V 写入 cache 第 `kvLen` 槽的层层偏移；`RunDecodeAttention` 对整个 KV cache 因果 attention；最后 `logits = h·lm_headᵀ + lm_headB` 写入 `[NumberMelCodes]`。
- `NumberMelCodes` / `StopMelToken` 暴露给编排器。

## 7. 子模型：`RepCodec`

### 7.1 用途
- `Quantize(feats, t, dstEmb, dstCodes)`：参考语义特征 `(T,1024)` → 码 `[T]` + 重建嵌入 `[T,1024]`（GPT 条件用）。
- `Vq2Emb(codes, t, dst)`：mel-code id → 码本嵌入 `[T,1024]`（s2mel DiT 条件用）。

### 7.2 量化器实现（`num_quantizers == 1` 退化为 L2 FVQ）
```
feats → in_proj((T,1024)→(T,8)) + bias
      → L2 归一化每行
      → 与 L2 归一化的码本行做最近邻（欧氏距离）
      → 码 id；重建 = out_proj(码本行) → (T,1024)
```
权重：`semantic_codec.quantizer.{in_proj,out_proj,vq.codebooks}`。
Vocos backbone（12 层残差卷积）是 TODO；当前直接用量化器（数值主导步骤）。

## 8. 子模型：`S2Mel`

### 8.1 职责
`latent → gpt_layer(MLP) → 长度调节 + vq2emb → DiT/CFM Euler → mel`。参考 `MyModel`（`s2mel/modules/commons.py`）。

### 8.2 `Generate`
- `gpt_layer(latent)`：`Gemm + bias`（`s2mel.gpt_layer.*`）。
- `targetLen = codesLen * 1.72`（长度调节器采样比的简化，3-quantizer 的 DAC 式调节是 TODO）。
- `cond = sInfer` 的前 `min(codesLen, targetLen)` 帧拷贝（identity 近似）。
- `cat = [prompt_condition(32) | cond]`，`fullLen = 32 + targetLen`。
- `SolveEuler(cat, fullLen, refMel, refMelLen, style, 192, inCh=80, steps=25, cfgRate=0.7)`。

### 8.3 `SolveEuler`（CFM Euler ODE）
- 初始 `x`：`N(0,1)·temperature`；prompt 区帧强制为 0（`x` 与 `promptX` 两路：被去噪变量 + 恒定 prompt 条件）。
- 每步：`dphi = DiTEstimator(x, promptX, mu, t, style)`；开启 CFG 时再算 `dphiNull`（`mu=null, style=null`），合成 `(1+cfgRate)·dphi − cfgRate·dphiNull`。
- `x += dtLocal·dphi`；之后把 prompt 区帧再次清零。
- 噪声源：`Random(seed=42)`，Box-Muller 高斯（下界裁剪防 NaN）。`seed` 参数保留（软种子数组，当前仅用于占位）。

### 8.4 `DiTEstimator`（**重数值路径中的简化**）
参考是 8 层 transformer + AdaLN + wavenet head；当前仅实现：
```
in_proj((inCh)·dim): (f,d) 行循环 → hidden (muLen, dim)
（transformer + wavenet 跳过 → TODO(phase 5)）
out_proj((dim)·inCh) → dst (inCh, muLen)
```
权重：`s2mel.DiT.in_proj.weight|bias`、`s2mel.DiT.out_proj.weight|bias`。

## 9. 子模型：`Wav2VecBert`（stub）

- 职责：16 kHz 参考音频 → layer-17 隐藏状态 `(T,1024)` → `mean/std` 标准化。
- 已实现：`Standardize(x, t, dst)`：`dst = (x − mean) / std`，`_mean`/`_std` 来自 `w2v_bert.stat_mean` / `w2v_bert.stat_std`。
- 未实现：`Forward16k` 完整 24 层 transformer（TODO(phase 5)；combined GGUF 中的 `semantic_model.*` 存放该模型）。

## 10. 子模型：`BigVgan`（stub）

- 职责：mel `(80,T)` → 波形 `(T·1024)`（2·4·8·8 上采样 + resblock）。
- 已实现：仅读 `bigvgan.0.weight|bias`；`MelToWave` 返回 `melFrames * HopLength` 长度的**零波形**（管线端点可运行）。
- 未实现：10 层 anti-aliased conv-transpose / resblock（TODO(phase 6)）。

## 11. 子模型：`Perceiver`

### 11.1 用途
- 说话人条件 perceiver（`perceiver.*`）：latent dim == model_dim，输出 32 latent（GPT 条件）。
- emo perceiver（`emo_perceiver.*`）：1 latent。

### 11.2 `Forward(ctx, ctxLen, ctxDim, mask, dst)`
- 深度固定 2 层（PerceiverResampler 默认）。
- `x = proj_context(ctx)`（ctxDim ≠ dim 时线性映射）→ 与可学习 `latents` 拼接为总上下文。
- 逐层 cross-attention：`q = to_q(latents)`、`kv = to_kv(ctxConcat)` → `Attention.Scores/Combine` → `to_out` + 残差 → GEGLU 风格 FFN（`ff.0` / `ff.2`，`dimInner = 4·dim·2/3`）→ 残差。
- 结束 `RMSNorm(latents)`（`perceiver.norm.0`）。

## 12. 子模型：`ConformerEncoder`（stub）

- 配置：input 1024 → `OutDim`（512），`Head/LinearUnits/NumBlocks` 来自配置，`InputLayer` 决定 `SubsampleRate`（`conv2d→4, conv2d2→2, conv2d6→6, conv2d8→8, linear→1`）。
- 已实现：定位 `conformer.encoders.0.self_attn.linear_{q,k,v,out,rel_k}` 权重；`Forward` 返回 `outLen = t / SubsampleRate` 并写零（占位）。
- 未实现：Conv2d 下采样 + 6 个 rel-pos Conformer 块 + after_norm（TODO，见代码注释——等待对真实 GGUF 确认张量名）。

## 13. 子模型：`TextTokenizer`

### 13.1 加载（`FromGguf`）
- `text_tokenizer.unk_token`（默认 128）、`start_text_token`（0）、`stop_text_token`（1）。
- `text_tokenizer.vocab`（字符串数组 → id 字典）。
- `text_tokenizer.merges`（"a b" 串 → 优先级 = 数组下标）。

### 13.2 归一化（`Normalize`）
全角标点 → ASCII：`：；；,;，、→ ,`、`。→ .`、`！→ !`、`？→ ?`、`\n\r → 空格`、`·—～~ → -`、引号/括号族 → `'`。

### 13.3 BPE（`BpeTokenize`）
- 字符为基底单元（CJK 单字一个 token；控制字符/空白跳过）。
- 按**合并优先级递增**扫描，命中则合并并回退指针，直至无变化或优先级耗尽。

### 13.4 查表回退（`Lookup`）
- 命中 vocab 直接返回；否则 UTF-8 字节 fallback（`<XX>` 形式，如 `<E4>`）；仍失败 → `_unk`。
- `Tokenize` 返回 `[start, ...pieces, stop]`。

## 14. 音频前端：`Audio`

### 14.1 WAV 读取（`LoadWav22k`）
- 手工解析 RIFF/WAVE：遍历 chunk 找 `fmt ` 与 `data`。
- 支持 PCM 16/24/32-bit 与 **float32**；取第一声道（mono）。
- `bitsPerSample` 不识别 → 0 采样。
- 采样率 ≠ 22050 时线性重采样到 22050。

### 14.2 mel 计算（`ComputeMel`，stub）
- 参考非中心 STFT + hann 窗 + log10 + 80 mel basis 的完整实现是 TODO(phase 6)；
- 当前按帧输出「能量和 ×0.1」的近似 mel（保证形状非零）。

### 14.3 WAV 写出（`SaveWav`）
- 16-bit PCM 单声道；`sample = Clamp(s·32767, −32767, 32767)`；写 RIFF/WAVE + `fmt ` + `data`。

## 15. 设计要点 / 约束

1. **所有权与生命周期**：`IndexTTS` 拥有所有子模型与 KV cache，`Dispose` 级联释放（`WeightStore` ⊆ GGUF；线程池；两段 KV cache）。
2. **GPU 无关**：全程 CPU float 运算，`unsafe` 指针 + `ScratchArena` 零分配热路径。
3. **宽松配置**：`FromGguf` 缺 key 回退默认，保证不同产出的 GGUF 可加载；`TryGetFloat` 对可选权重（bias 等）宽容。
4. **确定性输出**：`S2Mel` 默认种子 42，CFM 噪声与参考一致。

## 16. 已知 stub / TODO 汇总

| 位置 | 缺什么 |
|------|--------|
| `Wav2VecBert.Forward16k` | w2v-bert 24 层 transformer（layer-17 读取） |
| `ConformerEncoder.Forward` | Conv2d 下采样 + rel-pos Conformer 块 |
| `S2Mel.DiTEstimator` | DiT 8 层 transformer + AdaLN + wavenet head |
| `S2Mel.Generate` | 完整 DAC 式长度调节器（3 quantizer） |
| `RepCodec.Quantize` | Vocos 12 层骨干 |
| `BigVgan.MelToWave` | BigVGAN 上采样栈 |
| `Audio.ComputeMel` | 完整 STFT + DCT mel-basis |
| `Gpt` 条件 latent | `speed_emb`/`emo` 条件流（权重已加载未接线） |