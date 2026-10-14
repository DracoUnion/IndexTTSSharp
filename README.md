# IndexTTSSharp

纯 C# 的 [IndexTTS v2](https://github.com/Idx-BF/idx-tts) **非官方** CPU 推理实现。不依赖 llama.cpp / ONNX / PyTorch，自带 F32 GEMM 内核，面向进程内调用。

架构、GGUF 读取器与算子库参考 [HyMT2Sharp](https://github.com/sdcb/HyMT2Sharp)。

## 模型

权重需自行下载并打包为**单一 combined GGUF**（NuGet 包不含 GGUF）：

```
gpt.*            # GPT-2 8 层 512-dim 自回归 TTS 解码器
text_embedding / mel_embedding / mel_pos_embedding / text_pos_embedding
emo_layer / emovec_layer / speed_emb
perceiver.*      # 说话人/情绪条件 PerceiverResampler
conformer.*      # w2v-bert 特征上的 Conformer 编码（spk/emo 条件）
w2v_bert.*       # Wav2Vec2Bert 语义特征提取（layer-17 读层 + mean/std 归一化）
semantic_codec.* # RepCodec 语义编解码（Vocos + ResidualVQ）
s2mel.*          # DiT / CFM / 长度调节器 / wavenet
bigvgan.*        # BigVGAN 声码器（mel → 波形）
text_tokenizer.* # SentencePiece BPE 分词（vocab + merges）
```

## 快速开始

```powershell
dotnet run --project src/IndexTTSSharp.Cli -c Release -- `
  --model "D:\_\model\IndexTTS-v2.gguf" `
  --ref "examples\voice_01.wav" `
  --text "欢迎大家来体验indextts2，并给予我们意见与反馈。" `
  --out out.wav [--threads 8]
```

## 作为库使用

```powershell
dotnet add package Flygon.IndexTTSSharp.Model
```

```csharp
using Flygon.IndexTTSSharp.Model;

using IndexTTS tts = new(@"D:\_\model\IndexTTS-v2.gguf");
float[] refAudio = Audio.LoadWav22k(@"examples\voice_01.wav");
float[] wav = tts.Synth("欢迎大家来体验indextts2。", refAudio, "out.wav");
```

## 实现要点

- **GPT-2 解码器**（`Gpt.cs`）：8 层 pre-LN decoder，QKV fused `c_attn`，因果注意力 + GELU MLP，prefill/decode 两路，AVX2 F32 GEMM 走 `MatmulF32`。
- **Perceiver**（`Perceiver.cs`）：cond_perceiver（32 latents）+ emo_perceiver（1 latent），cross-attention + GEGLU FFN + RMSNorm。
- **Conformer**（`ConformerEncoder.cs`）：w2v-bert 特征上的 EFSM2 风格编码（Conv2dSubsample4 + rel-pos MHA + depthwise conv + macaron FFN）。
- **语义编解码**（`RepCodec.cs`）：Vocos 骨干 + L2 归一化 FVQ，`quantize`（ref 特征→codes）与 `vq2emb`（codes→1024-d 嵌入）。
- **s2mel / DiT / CFM**（`S2Mel.cs`）：Euler ODE 求解器 + CFG，8 层 DiT + wavenet head（重数值路径留作 phase 5 优化）。
- **w2v-bert**（`Wav2VecBert.cs`）：layer-17 读层 + mean/std 归一化（完整 24 层 transformer 是最大子模型，留作 phase 5）。
- **BigVGAN**（`BigVgan.cs`）：mel → 波形（10 层 anti-aliased conv-transpose 留作 phase 6）。
- **音频前端**（`Audio.cs`）：16/24/32-bit & float WAV 读取 + 线性重采样到 22050 Hz，mel STFT 桩，PCM WAV 写出。

## 项目结构

```
src/
  IndexTTSSharp.Gguf/    # GGUF v2/v3 reader（HyMT2Sharp 风格，内存映射）
  IndexTTSSharp.Kernels/  # AVX2 F32 GEMM/GEMV + conv/norm/attention/activation
  IndexTTSSharp.Model/    # 所有子模型 + 顶层 IndexTTS 编排 + Audio 前端
  IndexTTSSharp.Cli/      # 命令行入口
tests/
  IndexTTSSharp.Tests/    # xunit
```

## 许可证

Apache-2.0
