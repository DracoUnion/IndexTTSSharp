# 01 · 总体架构设计

**版本**: 1.0 · **更新**: 2026-10-15 · **适用范围**: 全项目

## 1. 项目定位

**IndexTTSSharp** 是 [IndexTTS v2](https://github.com/Idx-BF/idx-tts) 的**纯 C# 非官方 CPU 推理实现**，在**不依赖 llama.cpp / ONNX / PyTorch** 的前提下，把「文本 + 说话人参考音频」转换成 22.05 kHz 的 WAV 波形。

- 目标运行时：.NET 10（`net10.0`），纯 CPU，进程内调用。
- 架构、GGUF 读取器与算子库参考 [HyMT2Sharp](https://github.com/sdcb/HyMT2Sharp)。
- 使用说明见仓库 `README.md`；本文档专注设计层面。

## 2. 技术栈与构建

| 项 | 值 |
|----|----|
| 目标框架 | `net10.0` |
| 语言 | C#（`LangVersion=latest`，`Nullable` 开启，`AllowUnsafeBlocks=true`） |
| SIMD | AVX2 + FMA（`System.Runtime.Intrinsics`） |
| 测试 | xUnit 2.9.3 + .NET Test SDK 17.14.1 |
| 打包 | 三个可打包库独立版本号（见 `Directory.Build.props` / `.targets`） |

### 2.1 独立 NuGet 版本控制

`Directory.Build.props` 定义三个独立版本属性：

```
IndexTTSSharpGgufVersion     = 1.0.0
IndexTTSSharpKernelsVersion  = 1.0.0
IndexTTSSharpModelVersion    = 1.0.0
```

`Directory.Build.targets` 中的 `FixPackProjectReferenceVersions` target 解决了一个 NuGet 打包问题：`dotnet pack -p:Version=` 是全局属性，若不处理会**流入被引用项目的编译**，并且会被 `_GetProjectVersion` 当作包依赖版本返回，导致打包 Model 1.0.1 却依赖 Gguf/Kernels 1.0.1。该 target 用 `GlobalPropertiesToRemove=Version;PackageVersion` 移除全局属性，并在 `_GetProjectReferenceVersions` 之后用各项目自己的版本号覆写引用版本。

### 2.2 解决方案

`IndexTTSSharp.slnx`（新式 XML 解决方案）包含 5 个项目：

```
src/IndexTTSSharp.Cli/        # Exe，引用 Model + Gguf
src/IndexTTSSharp.Gguf/       # 库，独立版本 1.0.0
src/IndexTTSSharp.Kernels/    # 库，独立版本 1.0.0
src/IndexTTSSharp.Model/      # 库，独立版本 1.0.0，引用 Kernels + Gguf
tests/IndexTTSSharp.Tests/    # xUnit，不可打包
```

## 3. 推理 Pipeline 总览

顶层门面是 `IndexTTS`（`src/IndexTTSSharp.Model/IndexTTS.cs`），其 `Synth(text, speakerReference22k, outputWavPath)` 严格按照参考实现 `IndexTTS2.infer_generator`（`indextts/infer_v2.py`）编排：

| 步骤 | 子模型 | 输入 | 输出 | 实现状态 |
|------|--------|------|------|----------|
| 1 | `Audio.LoadWav22k` | 参考 WAV 文件 | 22 kHz 单声道 `float[]` `[-1,1]` | ✅ 完整 |
| 2 | `Wav2VecBert.Standardize` | 22 kHz 音频（~4× 下采样长度） | 标准化语义特征 `(T,1024)` | ⚠️ 语义特征实际来自外部；内部前向是 TODO |
| 3 | `RepCodec.Quantize` | 语义特征 | `refCodes [T]` + `S_ref [T,1024]` | ✅ 量化器完整，Vocos 骨干 TODO |
| 4 | `IndexTTS.Synth` 内部 | `S_ref` | `prompt_condition [32,512]`（拷贝前 32 帧） | ✅ |
| 5 | `Gpt`（自回归） | `textTokens` + `prompt_condition` + start_mel | mel-code 序列 | ✅ 完整 |
| 6 | `Gpt` / `S2Mel` | 生成的 codes | diffusion latent | ⚠️ latent 由 `S2Mel.Generate` 内部生成 |
| 7 | `S2Mel.Generate` | latent + codes + prompt + style | mel 谱 `(80,T)` | ⚠️ Euler+CFG 完整；DiT 8 层 transformer TODO |
| 8 | `BigVgan.MelToWave` | mel | 22 kHz 波形 | ⚠️ stub（返回正确长度的零波形） |
| — | `Audio.SaveWav` | 波形（归一化到峰值 1） | 16-bit PCM WAV | ✅ 完整 |

### 3.1 各步骤数据流细节（来自 `IndexTTS.Synth`）

1. `scratch`（`ScratchArena`）为本调用内复用的临时缓冲区，按 `A..F` 六个槽位分配。
2. `refFeatLen = speakerReference22k.Length / 4`（约 4× 下采样）。
3. 参考条件：`_w2v.Standardize` → `RepCodec.Quantize` → 得到 `sRef` 与 `refCodes`。
4. `promptCondition` = 前 `32`（`ConditionNumLatent`）帧的 `S_ref`（此处即 length regulator 的简化实现）。
5. GPT 自回归：
   - `_tokenizer.Tokenize(text)` → `textTokens`（含 start/stop）。
   - 每个 token 经 `EmbedText(token, pos)` 得到 `textEmb`（词 + 位置嵌入之和）。
   - 组装条件流 `stream = [prompt_condition(32), text_emb, start_mel_emb]`，长度 `32 + textTokens.Length + 1`。
   - `_gpt.Prefill(stream, streamLen, null)` 完成 prefill。
   - 循环最多 `MaxMelTokens` 次：取 `logits` argmax → 若为 stop 则停止 → 记录 `generated` → `EmbedMel` → `Decode`（写 KV cache，位置 `streamLen + step`）。
   - logits 缓冲为 `_scratchLogits`（大小 `NumberMelCodes = 8194`）。
6. `latent` 清零为 `(nGenerated, 512)`（GPT forward 的实际输出在参考中由 `Gpt.forward(speech_conditioning_latent)` 产生，当前走 `S2Mel` 内部路径）。
7. `S2Mel.Generate`：
   - `gpt_layer(latent)`（MLP）→ `latOut`。
   - `sInfer` 应为 `vq2emb(codes) + latent`（由调用方预先写入）。
   - `targetLen = codesLen * 1.72`（长度调节器采样比的简化）。
   - 拼接 `cat = [prompt_condition, cond]`。
   - `SolveEuler`：`dt=1/steps`（默认 25 步），每步调 `DiTEstimator`，开启 CFG 时再算 `dphiNull`（空条件），合成 `(1+cfgRate)·φ(cond) − cfgRate·φ(null)`；初始 x 为高斯噪声，prompt 区帧强制为 0。
   - 截取尾部 `fullLen − refMelLen` 帧作为最终 mel。
8. `BigVgan.MelToWave`：当前返回 `melFrames * HopLength` 长度的**零波形**（stub，见 `04-Model-Module.md` §BigVgan）。
9. 归一化到峰值 1 → `Audio.SaveWav`（16-bit PCM 单声道）。

## 4. 模块分层与依赖

```
┌──────────────────────────────────────────────────────────┐
│  IndexTTSSharp.Cli         命令行参数 → IndexTTS 调用      │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  IndexTTSSharp.Model       子模型 + 顶层编排 + 音频前端    │
│  引用: Kernels, Gguf                                     │
└──────┬──────────────────────┬────────────────────────────┘
       ▼                      ▼
┌─────────────┐      ┌──────────────────┐
│  Kernels    │      │  Gguf            │
│  CPU 算子库 │      │  GGUF 读取器     │
│  (无依赖)   │      │  (无依赖)        │
└─────────────┘      └──────────────────┘
```

依赖规则：
- `Kernels` 与 `Gguf` 是叶子模块，无内部项目依赖。
- `Model` 同时引用 `Kernels`（算子）与 `Gguf`（权重读取），通过 `WeightStore` 解耦「张量定位」与「子模型使用」。
- `Cli` 引用 `Model`（推理）+ `Gguf`（仅编译期传递）。
- `Tests` 引用全部三个库。

## 5. 内存与并发模型

### 5.1 原生内存：`NativeBuffer`
- `NativeMemory.AlignedAlloc(64)` 分配、`AlignedFree` 释放，附终结器兜底。
- 所有权明确：所有权由持有方（`WeightStore`、各子模型、KV cache）负责释放；临时缓冲交给 `ScratchArena`。

### 5.2 临时缓冲：`ScratchArena`
- 一个实例持有多达 6 个 `NativeBuffer`（槽位 `A..F`）。
- `A(nuint bytes)` 等按需扩容：已有缓冲区足够大则复用指针，否则释放重分配。
- 调用方**覆盖**所需字节，不复位整个区域，避免反复 memset。
- 生命周期：`IndexTTS.Synth` 每次调用创建、finally 中释放；子模型各自持有长期 `ScratchArena` 复用。

### 5.3 KV cache
- `IndexTTS` 构造时按 `layers * kvCacheLen * dim * sizeof(float)` 为 K、V 各分配一个 `NativeBuffer`（默认 `8 * 1024 * 512 * 4B = 16 MiB`）。
- `Gpt.Decode` 把新 token 的 K/V 写入第 `kvLen` 槽位，逐层就地覆盖读取。

### 5.4 线程池：`CpuThreadPool`
- 专用后台工作线程，线程数默认取 `CpuTopology.PreferPCoreCount`（P 核优先，Windows 下设置线程亲和性）。
- `For(count, (worker, workers) => …)`：ggml 风格 job-id 轮询 + `AutoResetEvent` 休眠（避免顺序小任务空转烧 P 核）。
- `Barrier()`：ggml 风格自旋屏障，用于一次 `For` 内的 quantize→GEMM 同步。
- `MatmulF32` / `GemmQ4K` 等将行块分给 worker；`rows <= 1` 或线程数 1 时退化为串行。

### 5.5 指针约定
- 权重统一解量化到 `float*`（`WeightStore.GetFloat`），模型代码用 `unsafe` + `Buffer.MemoryCopy`。
- 行优先布局；GGUF 维度约定 dim0 为最内层（行）长度（见 `02-Gguf-Module.md`）。

## 6. 配置体系：`IndexTTSConfig`

- 超参数枚举，`init` 属性带默认值，与参考 Python 配置一致。
- `IndexTTSConfig.FromGguf(GgufFile)`：按 GGUF metadata key 读取，**缺失即回退默认值**，使部分指定的 GGUF 也能加载。
- 分组：
  - `gpt.*`：层数、dim、头数、KV 长度、文本/mel token 上限、特殊 token id、条件 latent 数、spk 条件模式。
  - `gpt.condition_module.*`：Conformer 尺寸。
  - `semantic_codec.*`：RepCodec / Vocos 尺寸。
  - `s2mel.*`：DiT、wavenet、style/content dim、长度调节器。
  - `campplus.*`、`w2v_bert.*`、`s2mel.preprocess_params.*`（采样率、mel 参数）。
- 完整字段见 `04-Model-Module.md` §IndexTTSConfig。

## 7. 实现要点

1. **纯 CPU 数值路径**：默认所有权重以 F32 参与计算（`WeightStore` 解量化到 float）；量化内核（Q4K 等）已实现但 GPT 主路径当前走 `MatmulF32` F32 GEMM。
2. **AVX2 + FMA**：`MatmulF32.GemmRowAvx` 外积形式，逐 k 广播 `Vector256`，`Fma.MultiplyAdd` 累加。
3. **严格对齐参考实现**：`IndexTTS` 逐步骤镜像 `infer_v2.py`，张量名直接对应 GGUF 中的 torch-module 路径。
4. **失败语义**：`WeightStore` 对缺失张量记入 `Unmapped` 列表（非致命）；配置缺失回退默认。
5. **确定性**：`S2Mel` 种子默认 42；`Deterministic=true` 构建。

## 8. 实现状态与 Roadmap

代码中以 `TODO(phase N)` 标记未完成重数值路径：

| Phase | 内容 | 现状 |
|-------|------|------|
| — | GPT-2 解码器（prefill/decode、KV cache） | ✅ 完整 |
| — | RepCodec 量化器（L2 FVQ） | ✅ 完整（Vocos 骨干未实现） |
| — | S2Mel Euler ODE + CFG | ✅ 求解器完整 |
| 5 | w2v-bert 完整 24 层 transformer | ⚠️ 仅 `Standardize`（读层-17 + mean/std） |
| 5 | Conformer 完整编码（EFSM2 风格） | ⚠️ 仅尺寸/权重定位 stub |
| 5 | s2mel DiT 8 层 transformer + AdaLN + wavenet head | ⚠️ `DiTEstimator` 仅 in/out 投影 |
| 5 | RepCodec Vocos backbone（12 层残差卷积） | ⚠️ 未实现 |
| 6 | BigVGAN 10 层 anti-aliased conv-transpose | ⚠️ stub（返回零波形） |
| 6 | 完整 STFT + DCT mel-basis | ⚠️ `Audio.ComputeMel` 为能量近似 |

## 9. 使用方式

- 库用法：`using Flygon.IndexTTSSharp.Model; IndexTTS tts = new(ggufPath); float[] wav = tts.Synth(text, refAudio22k, "out.wav");`
- CLI 用法：见 `05-Cli-Module.md`。
- 权重：需自备**单一 combined GGUF**（`gpt.*`, `semantic_codec.*`, `s2mel.*`, `bigvgan.*`, `w2v_bert.*`, `text_tokenizer.*` 等），NuGet 包不含 GGUF。
