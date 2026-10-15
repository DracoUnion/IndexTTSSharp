# IndexTTSSharp 设计文档

IndexTTSSharp（IndexTTS v2 纯 C# CPU 推理实现）的完整设计与实现文档索引。

## 文档结构

| 文档 | 范围 | 对应代码路径 |
|------|------|--------------|
| [01-Overview.md](01-Overview.md) | 总体架构：推理 Pipeline、模块分层、内存/并发模型、配置体系、Roadmap | 全项目 |
| [02-Gguf-Module.md](02-Gguf-Module.md) | GGUF v2/v3 文件读取器详细设计 | `src/IndexTTSSharp.Gguf/` |
| [03-Kernels-Module.md](03-Kernels-Module.md) | CPU 推理算子库：量化 GEMM/GEMV、卷积、归一化、注意力等 | `src/IndexTTSSharp.Kernels/` |
| [04-Model-Module.md](04-Model-Module.md) | 各 TTS 子模型 + 顶层编排 + 音频前端 + 权重存储 + 配置 | `src/IndexTTSSharp.Model/` |
| [05-Cli-Module.md](05-Cli-Module.md) | 命令行入口详细设计 | `src/IndexTTSSharp.Cli/` |
| [06-Tests-Module.md](06-Tests-Module.md) | 单元测试详细设计 | `tests/IndexTTSSharp.Tests/` |

## 快速阅读建议

- 整体推理流程 → `01-Overview.md` §3
- 权重如何从 GGUF 加载 → `02-Gguf-Module.md` + `04-Model-Module.md` §5（WeightStore）
- 性能关键路径（GEMM / 量化内核）→ `03-Kernels-Module.md` §4-§5、§9
- 各子模型实现状态（哪些是 stub / TODO）→ `04-Model-Module.md` §16

## 约定

- 命名空间统一为 `Flygon.*`（参考实现 HyMT2Sharp 为 `Sdcb.*`）。
- 权重张量命名遵循 GGUF 中 torch-module 路径，与参考 Python 实现 `indextts/infer_v2.py` 一一对应。
- 文档中 `phase N` 是项目内部对实现进度的阶段标记（见 `01-Overview.md` §8）。
- 各模块文档中的 API 签名与代码保持一致；如与代码不符，以代码为准。
