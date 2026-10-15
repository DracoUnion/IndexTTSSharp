# 03 · IndexTTSSharp.Kernels 模块详细设计

**版本**: 1.0 · **更新**: 2026-10-15 · **代码路径**: `src/IndexTTSSharp.Kernels/`

## 1. 模块职责

CPU 推理算子库，为上层模型提供全部数值原语。按用途分为六组：

1. **F32 稠密矩阵乘**（`MatmulF32`）—— 非量化子模型（GPT/Conformer/Perceiver/DiT/MLP）的主力 GEMM/GEMV。
2. **量化 GEMM/GEMV 体系** —— 针对 ggml 量化权重（Q4K/Q6K/Q2/STQ/Q2_0C）的高性能内核，含量化、重打包、块点积、panel GEMM。
3. **神经网络原语** —— 卷积（`Conv1d`/`Conv2d`）、归一化（`Norms`）、激活（`Activations`）、注意力（`Attention`）、RoPE（`Rope`）。
4. **通用运算**（`Ops`）与**快速数学**（`FastExp`）。
5. **基础设施** —— `NativeBuffer`（原生内存）、`ScratchArena`（临时缓冲池）、`CpuThreadPool`（线程池）、`CpuTopology`（拓扑/亲和）、`BlockLayout`（量化块布局）。

## 2. 文件清单与职责总览

| 文件 | 职责 |
|------|------|
| `MatmulF32.cs` | F32 GEMM/GEMV，AVX2+FMA 外积，行并行 |
| `BlockLayout.cs` | 量化块布局结构体（Q4K/Q8K/Q5K/Q6K/Q2_0C/STQ 及其 x8/x4 打包变体）+ 常量 `Qk` + `HalfBits` 工具 |
| `Q4K.cs` / `Q5K.cs` / `Q6K.cs` / `Q8K.cs` / `Q2_0C.cs` / `STQ1_0.cs` | 各类块的解量化、块点积（标量/AVX2）、GEMV/GEMM/量化 |
| `VecDotQ4K.cs` | Q4K·Q8K 块点积（标量 / AVX2 / 水平归约） |
| `RepackQ4K.cs` / `RepackQ6K.cs` / `RepackQ2.cs` / `RepackSTQ.cs` | 将磁盘块重打包为 x8 交错的 SIMD 友好布局，`RepackQ4K.BuildMeta` 生成元数据 |
| `QuantizeQ8Kx4.cs` | 激活量化到 Q8K 4×8 块（含 SiLU 融合、AVX2/标量） |
| `GemmQ4K.cs` / `GemmQ4K.Avx2.cs` / `GemmQ6K.cs` | Q4K/Q6K 8×8 微内核 GEMM（标量/AVX2） |
| `MulMatQ4K.cs` / `MulMatQ6K.cs` / `MulMatQ2.cs` / `MulMatSTQ.cs` / `MulMatPanel.cs` | 高层量化矩阵乘：GEMV/GEMM/prequant、panel 融合（quantize+GEMM、SiLU+quantize+GEMM） |
| `Q2Panel.cs` / `STQPanel.cs` | Q2/STQ panel 的 GEMV/GEMM |
| `Conv1d.cs` / `Conv2d.cs` | 一维/二维卷积，`Conv1d.FoldWeightNorm` 权重归一化折叠 |
| `Norms.cs` | LayerNorm / GroupNorm / BatchNorm / Softmax |
| `Activations.cs` | GELU / SiLU / Mish / GLU / Snake / SnakeBeta / ReLU / Tanh / Abs |
| `Attention.cs` | 通用多头注意力 `Scores` / `Combine`（支持 GQA、padding mask） |
| `Rope.cs` | RoPE `Precompute` / `Apply` |
| `Ops.cs` | `Memset` / `Cumsum` / `EmbedGatherF32` / `SoftmaxRow` |
| `FastExp.cs` | AVX2 快速 exp / SiLU |
| `NativeBuffer.cs` | 64 字节对齐原生缓冲区（分配/释放/终结器） |
| `ScratchArena.cs` | A..F 六槽位临时缓冲复用池 |
| `CpuThreadPool.cs` | 专用工作线程 + job 轮询 + 自旋屏障 + P 核亲和 |
| `CpuTopology.cs` | CPU 拓扑探测（逻辑/P/E 核、SMT、虚拟化）、线程数选择、亲和掩码 |

## 3. 核心数据结构：量化块布局（`BlockLayout.cs`）

`Qk = 256`（ggml 块大小）。磁盘块 → SIMD 友好块：

| 结构 | 含义 |
|------|------|
| `BlockQ4K` | 磁盘 Q4_K 块（256 元素） |
| `BlockQ5K` | 磁盘 Q5_K 块 |
| `BlockQ6K` | 磁盘 Q6_K 块 |
| `BlockQ8K` | 磁盘 Q8_K 块（激活量化目标） |
| `BlockQ2_0C` | 自定义 2-bit 压缩块 |
| `BlockSTQ1_0` | STQ 1.0 块 |
| `BlockSTQ1_0x8` / `BlockQ2x8` / `BlockQ4Kx8` / `BlockQ6Kx8` | 8 块交错的打包布局（x8，SIMD 读 8 行） |
| `BlockQ4Kx8Meta` | Q4K x8 打包的每块元数据（scale/min 解包偏移等） |
| `BlockQ8Kx4` | 激活按 4 个 token 打包的 Q8 块（GEMM 的 A 侧） |
| `HalfBits` | 打包 Half 位操作工具 |
| `BlockLayout.Load8(ushort*)` | 加载 8 个 half（`Vector256`） |

## 4. 量化 GEMM 数据流

以 Q4K 为例（`MulMatQ4K` / `GemmQ4K`）：

```
权重（磁盘 Q4_K）
  │  RepackQ4K.MakeBlockX8 / Rows / BuildMeta
  ▼
BlockQ4Kx8 权重面板（x8 交错） + BlockQ4Kx8Meta 元数据
                                   │
激活 x[tokens, nIn]  ── QuantizeQ8Kx4.Quantize4x8 ──► BlockQ8Kx4 激活面板（Q8）
                                   │
                                   ▼
GemmQ4K.Gemm8x8 / GemmAvx2（8×8 微内核，行并行于 CpuThreadPool）
                                   ▼
输出 y[tokens, nOut]
```

- **微内核**：每次计算 8 输出行 × 8 输入行（8×8 tile），标量与 AVX2 两个变体，`GemmQ4K.Avx2.cs` 提供 `GemmAvx2`。
- **Prequant 路径**：`MulMatQ4K.GemvPrequant` / `GemmPrequant` 接受已量化的激活（`BlockQ8K*` / `BlockQ8Kx4*`），避免在 decode 阶段重复量化同一激活。
- **Panel 融合**：`MulMatPanel.QuantizeAndGemm`（quantize→GEMM 单次 `For`）与 `SiluQuantizeAndGemm`（gate/up 两输入先 SiLU 再融合量化），一次并行调度完成多步，减少同步点。
- `PanelWeight` / `Q2PanelWeight` / `STQPanelWeight`：封装「已打包权重 + 解量化目标 + nOut」的权重视图，供 panel GEMM 使用。

## 5. F32 矩阵乘（`MatmulF32.cs`）

- `Gemm(a, b, c, rows, kIn, cols, accumulate, pool)`：`C[rows,cols] = A[rows,kIn] · B[kIn,cols]`，B 按 `[cols, kIn]` 行优先存储（权重矩阵 as-stored，kIn 为最内轴）。
- 行并行：`pool.For(rows, …)` 均分行块；无池或单行 → 串行。
- 单行内核：`kIn >= 8` 且支持 AVX2+FMA 时走 `GemmRowAvx`（外积：逐 k 广播 `Vector256.Create(ak)`，`Fma.MultiplyAdd` 累加），否则 `GemmRowScalar`。
- `Gemv(x, b, y, kIn, cols, addBias, bias, pool)`：单 token 的 GEMV；`cols > 64` 时列并行。

## 6. 神经网络原语

### 6.1 卷积
- `Conv1d`：一维卷积 + `FoldWeightNorm`（把 weight norm 的 `g` 折入 `w`，推理期省一次归一化）。
- `Conv2d`：二维卷积（Conformer 下采样 `conv2d` 输入层用）。

### 6.2 归一化（`Norms`）
- `LayerNorm(x, weight, bias, rows, dim, eps, pool)`：GPT2 式 LayerNorm（参考实现 HF 块用 LN，eps=1e-5）。
- `GroupNorm` / `BatchNorm`：GroupNorm 支持 `(inChannels, time)` 布局，BatchNorm 用 running stats 推理期路径。
- `Softmax(x, rows, dim)`：按行 softmax。

### 6.3 激活（`Activations`）
`GELU`（tanh 近似）、`SiLU`、`Mish`、`Glu(x, n2, dst)`、`Snake`/`SnakeBeta`（anti-aliased BigVGAN 用）、`ReLU`、`Tanh`、`Abs`。

### 6.4 注意力（`Attention.cs`）
- 通用多头注意力，**非因果**（Perceiver/Conformer/DiT 用）；GPT-2 的因果注意力由模型内部实现（见 `04-Model-Module.md` §Gpt）。
- `Scores(q, k, scores, heads, kvHeads, headDim, qLen, kvLen, scale, pool)`：支持 GQA（`group = heads / kvHeads`），输出 `[heads, qLen, kvLen]`。
- `Combine(v, scores, result, …)`：先对 kvLen 行 softmax，再 `out = Σ w·v`，输出 `[qLen, heads, headDim]`。
- 布局：`q: [qLen, heads, headDim]`、`k/v: [kvLen, kvHeads, headDim]`（batch=1，非 heads-major），与 HyMT2Sharp `Ops.AttentionScores/Combine` 相同。

### 6.5 RoPE（`Rope.cs`）
- `Precompute(seqLen, nElem, baseFreq, out cos, out sin)`：旋转角预计算。
- `Apply(qk, rows, heads, headDim, pos, cos, sin)`：就地旋转。

## 7. 通用运算与快速数学

- `Ops`：`Memset`、`Cumsum`（长度调节器用）、`EmbedGatherF32`（embedding 查表）、`SoftmaxRow`。
- `FastExp`：AVX2 快速 `ExpAvx2` 与 `SiluAvx2`（多项式近似，用于 softmax / SiLU 热路径）。

## 8. 基础设施

### 8.1 `NativeBuffer`
- `NativeMemory.AlignedAlloc(bytes, 64)`，`Clear` 归零，`AlignedFree` 释放。
- 暴露 `Pointer` / `Bytes` / `Span`；终结器兜底防泄漏。

### 8.2 `ScratchArena`
- 六槽位 `A..F`，`Ensure` 按需扩容（够用复用、否则重分配），调用方覆盖使用。

### 8.3 `CpuThreadPool`
- 专用后台线程，`Name = "hymt2-cpu-{i}"`，`AboveNormal` 优先级。
- `For`：`jobId` 自增 → 唤醒 worker → 主线程自旋等待 `_remaining == 0`（每 4096 次自旋 `Thread.Yield`）。
- `Barrier`：ggml 风格 `barrierCount`/`barrierGen` 自旋屏障。
- Windows 下按 `CpuTopology` 目标列表设置 `SetThreadAffinityMask`（P 核优先 → 物理核 → 逻辑核；`index % targets.Count` 环形映射）。
- `Dispose`：置 `_stop`、`Interlocked.Increment(_jobId)`、唤醒所有 worker、`Join`。

### 8.4 `CpuTopology`
- 探测逻辑/物理核数、P 核数（`CountPCores`）、SMT、虚拟化（`LooksLikeVirtualFirmware`）、解析 CPU list（`ParseCpuList`）。
- `ChooseWorkerCount(logical, physical, pCore, sawSmt, vm)`：线程数启发式选择。
- `PreferPCoreCount`：默认线程数（P 核优先）。
- 亲和目标：`PCoreLeaders` / `PhysicalLeaders` / `PhysicalLogicalIds`。

## 9. 性能要点

1. **避免分配**：热路径全走 `ScratchArena` / `NativeBuffer`，无 GC 压力。
2. **SIMD 布局**：权重重打包为 x8 交错，让一次 `LoadVector256` 覆盖 8 行数据，GEMM 微内核 8×8 tile。
3. **并行粒度**：GEMM 按行、GEMV 按列切分到线程池；融合 kernel（quantize+GEMM）减少 `For` 调用次数。
4. **亲和**：P 核优先 + 自旋/休眠混合等待，小任务不烧 P 核。
5. **Prequant 缓存**：decode 阶段激活可复用已量化结果。

## 10. 与参考实现的对齐

- 量化类型与块布局对齐 llama.cpp `ggml` / HyMT2Sharp（`Qk=256`、Q4K/Q8K 块格式、x8 打包）。
- `CpuThreadPool` / `Barrier` 语义对标 ggml 的线程模型。
- 数值近似（GELU tanh、FastExp）与参考 Python 实现保持一致精度水平。

## 11. 后续方向

- GPT 主路径当前用 F32 GEMM；可切换为 Q4_K panel GEMM（`MulMatPanel` 已就绪）以降低显存/内存带宽。
- Q2/Q2_0C/STQ 面板内核与 Q4K/Q6K 的覆盖度补齐。
- 在更多平台验证 AVX2 回退路径（当前仅检查 `Avx.IsSupported && Fma.IsSupported`）。
