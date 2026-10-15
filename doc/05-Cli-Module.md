# 05 · IndexTTSSharp.Cli 模块详细设计

**版本**: 1.0 · **更新**: 2026-10-15 · **代码路径**: `src/IndexTTSSharp.Cli/`

## 1. 模块职责

可执行程序的命令行入口：解析参数 → 加载 `IndexTTS` → 读取参考音频 → 合成 → 写出 WAV → 打印结果。

只有一个源文件 `Program.cs`（`internal static class Program`）。

## 2. 参数解析

不依赖第三方参数库，手工 `--key value` 解析：

| 参数 | 必选 | 说明 |
|------|------|------|
| `--model <path>` | ✅ | combined GGUF 权重文件 |
| `--ref <path>` | ✅ | 参考说话人 WAV |
| `--text <str>` | ✅ | 合成文本（建议加引号） |
| `--out <path>` | ⬜ | 输出 WAV 路径，默认 `out.wav` |
| `--threads <int>` | ⬜ | 线程数，0 = 默认（P 核优先） |

解析函数：
- `Get(args, name)`：`Array.IndexOf` 找参数名，返回后一个 token；越界/缺失 → `null`。
- `GetInt(args, name, fallback)`：`int.TryParse`，失败回退。

## 3. 执行流程

```
Main(args)
  ├─ 缺 --model / --ref / --text → 打印 Usage，返回 1
  ├─ using IndexTTS tts = new(model, threads)
  ├─ float[] refAudio = Audio.LoadWav22k(refPath)
  ├─ float[] wav = tts.Synth(text, refAudio, outPath)
  ├─ 打印 "Wrote {n} samples @ 22050 Hz to {outPath}"
  └─ 返回 0
```

## 4. 使用示例

```powershell
# 最小用法
dotnet run --project src/IndexTTSSharp.Cli -c Release -- `
  --model "D:\_\model\IndexTTS-v2.gguf" `
  --ref "examples\voice_01.wav" `
  --text "欢迎大家来体验indextts2，并给予我们意见与反馈。" `
  --out out.wav

# 指定线程数
dotnet run --project src/IndexTTSSharp.Cli -c Release -- --model m.gguf --ref r.wav --text "..." --threads 8
```

## 5. 设计说明

- **进程内**一次性推理，不做服务/常驻；线程由 `CpuThreadPool` 管理并在进程退出时随 `IndexTTS.Dispose` 释放。
- 错误处理：缺参返回码 1；模型/音频/推理异常直接向上抛出（由 dotnet host 打印）。
- 输出采样率固定为配置 `Sr`（22050），与参考模型一致。

## 6. 后续方向

- 支持列出 GGUF 内可用模型/元数据（debug 子命令）。
- 支持 `--seed` 固定随机性、`--steps`/`--cfg` 透传 CFM 超参。
- 批处理 / 长文本分段。