# 06 · IndexTTSSharp.Tests 模块详细设计

**版本**: 1.0 · **更新**: 2026-10-15 · **代码路径**: `tests/IndexTTSSharp.Tests/`

## 1. 模块职责

xUnit 单元测试：验证分词器行为与配置默认值。当前测试规模较小，未来将扩展为逐子模型的数值一致性测试（对照参考 Python 输出）。

## 2. 项目配置

`IndexTTSSharp.Tests.csproj`：

- `IsPackable=false`。
- 引用：`xunit` 2.9.3、`Microsoft.NET.Test.Sdk` 17.14.1、`xunit.runner.visualstudio` 3.1.4。
- 项目引用：`Kernels`、`Model`、`Gguf` 全部三个库。
- 全局 `Using Include="Xunit"`。

## 3. 现有测试

### 3.1 `PipelineTests.cs`（管线级行为）

| 测试 | 断言 |
|------|------|
| `TextNormalizer_MapsFullWidthPunctuation` | `TextTokenizer.Normalize("你好，世界！") == "你好,世界!"` |
| `TextNormalizer_MapsQuotesAndBrackets` | `Normalize("你好（世界）[A]") == "你好'世界''A'"` |
| `GptConfig_Defaults_AreSane` | 默认配置：`GptLayers=8, ModelDim=512, StopMelToken=8193, StartMelToken=8192, NumberMelCodes=8194` |

### 3.2 `SkeletonTests.cs`

`Builds()`：仅断言编译通过（构建冒烟）。

## 4. 测试策略说明

- 当前测试**不需要 GGUF 权重文件**，只覆盖纯逻辑（归一化、配置默认值）。
- 由于模型重数值路径多为 stub / TODO（见 `04-Model-Module.md` §16），尚未引入需要真实权重的 golden 测试。

## 5. 未来扩展方向

1. **分词器**：BPE merge 优先级、字节 fallback（`<XX>`）、unk 回退、start/stop 注入。
2. **配置**：`FromGguf` 对缺 key/错误 key 的回退；`GetInt32Array`（sampling_ratios）。
3. **音频**：`Audio.LoadWav22k` 对 16/24/32-bit、float32、非 22050 采样率的读取与重采样；`SaveWav` 往返。
4. **Gguf**：魔数/版本/对齐非法文件的异常路径；STQ 兼容归一化（`CanonicalTensorType`）。
5. **Kernels**：Q4K/Q6K 块点积标量 vs AVX2 一致性；`GemmQ4K` 标量 vs AVX2 一致性（现有 `DotScalarForTest` 等已暴露测试钩子）。
6. **端到端**：一旦 Phase 5/6 补齐，用固定参考音频 + 文本生成 golden WAV 做数值一致性回归。