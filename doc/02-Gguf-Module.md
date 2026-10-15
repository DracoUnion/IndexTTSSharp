# 02 · IndexTTSSharp.Gguf 模块详细设计

**版本**: 1.0 · **更新**: 2026-10-15 · **代码路径**: `src/IndexTTSSharp.Gguf/`

## 1. 模块职责

读取 **GGUF v2/v3** 权重文件（内存映射），解析元数据（KV）与张量目录，为上层 `WeightStore` 提供**按名称寻址 + 按偏移访问**的接口。Tensor payload 不被复制进托管堆，而是驻留在内存映射中，以 `DataBase + Offset` 寻址（与 ggml 的对齐 data section 一致）。

## 2. 文件清单

| 文件 | 内容 |
|------|------|
| `GgufFile.cs` | 核心读取器：内存映射、头/元数据/张量目录解析、类型化取值 API、内部游标 |
| `GgmlTensorType.cs` | 磁盘 `ggml_type` 枚举（对齐 llama.cpp `ggml.h`） |
| `GgufTensorInfo.cs` | 单个张量的信息结构（类型、偏移、形状） |

## 3. `GgufFile` 详细设计

### 3.1 生命周期与资源

```
构造函数(path)
  ├─ FileStream(Read, FileShare.Read, FileOptions.RandomAccess)
  ├─ 长度 < 24 字节 → InvalidDataException
  ├─ MemoryMappedFile.CreateFromFile(...)
  ├─ CreateViewAccessor(0, length, Read)
  ├─ SafeMemoryMappedViewHandle.AcquirePointer → _fileBase
  └─ Parse(length)          # 一次性解析头
Dispose()
  ├─ ReleasePointer（幂等，_released 防重入）
  ├─ _view.Dispose(); _mmf.Dispose()
  └─ GC.SuppressFinalize(this)
```

- 只读内存映射，`FileShare.Read` 允许并发读取同一文件。
- `DataBase` 是暴露给外部（`WeightStore`）的 data section 起始指针；张量实际字节 = `DataBase + GgufTensorInfo.Offset`。

### 3.2 头解析（`Parse`）

按 GGUF 规范顺序：

1. **magic**：`0x46554747`（"GGUF" LE），不符 → `InvalidDataException`。
2. **version**：仅支持 2 / 3，否则 `NotSupportedException`。
3. **tensor_count / kv_count**（`ulong`）：超过 `int.MaxValue` → 判定文件不合理。
4. **KV 循环**：读 key（字符串）+ value；空 key → 异常。
5. **对齐**：读 `general.alignment`（默认 32），必须为 2 的幂且非零。
6. **tensor 目录循环**：读 name、n_dims（1..4，否则异常）、逐维 `int64`（负数 → 异常）、type（`int32`）、offset（`uint64`）→ 存入 `_tensors[name] = GgufTensorInfo`。
7. **data section 定位**：`pos = Align(游标位置, alignment)`（对齐公式 `(v + a - 1) & ~(a - 1)`），越过文件末尾 → 异常；`DataBase = _fileBase + pos`。

### 3.3 内部游标 `Cursor`

- 结构体，持 `_base`（文件基址指针）、`_length`、`_pos`。
- 小端原语读取（`Unsafe.ReadUnaligned<T>`）：`ReadU8/I8/U16/I16/U32/I32/U64/I64/F32/F64`。
- `ReadString`：`ulong` 长度（>2 GB → 异常）→ UTF-8 解码。
- 所有读取前调用 `Ensure(bytes)` 做边界检查（`EndOfStreamException`）。

### 3.4 值类型体系（`GgufValueType`）

覆盖 GGUF 全部标量类型 + 一维数组（嵌套数组不支持 → `InvalidDataException`）：

`Uint8=0 … Float32=6, Bool=7, String=8, Array=9, Uint64=10, Int64=11, Float64=12`

`ReadValue` / `ReadScalar` 为递归解析器；数组元素类型由 array 头中的 element-type 决定。

### 3.5 类型化取值 API（强转，非严格解码）

返回 `object` 的原始 KV 值，按目标类型**宽容强转**（使跨工具产出的 GGUF 也能读取）：

| API | 行为 |
|-----|------|
| `GetString(key, default)` | 仅当值是 `string` |
| `GetStringArray(key)` | 数组元素 `as string ?? ToString()` |
| `GetInt32(key, default)` | `CoerceInt64`（覆盖全部整数类型）后 `checked((int))` |
| `GetInt32Array(key)` | 逐元素 `CoerceInt64` |
| `GetUint32(key)` | 缺失/非整型 → `KeyNotFoundException`；`GetUint32(key, default)` 宽容 |
| `GetFloat32(key, default)` | `CoerceFloat`（float/double/整数） |

### 3.6 STQ 兼容归一化（`CanonicalTensorType`）

处理 llama.cpp PR #22836 前后的 STQ 类型冲突：

- 早期 HyMT2 Stride16 模型使用 raw type **42** / file type **41**，与标准 `Q2_0`（raw 42）冲突。
- 判定条件：`rawType == 42` 且 `general.file_type == 41` 且 `general.name` 含 "Stride16"（忽略大小写）→ 归一化为 `GgmlTensorType.STQ1_0`（43）。
- 目的：**标准 Q2_0 文件绝不会被静默解释为 STQ 字节**。

## 4. `GgmlTensorType`（磁盘类型）

对齐 llama.cpp `ggml_type`，重点类型：

- 浮点：`F32=0, F16=1, BF16=30, F64=28`
- 传统 4/5/8 bit：`Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9`
- K 系列：`Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15`
- IQ 系列：`IQ2_XXS=16 … IQ1_M=29`
- 整数张量：`I8=24, I16=25, I32=26, I64=27`
- TQ/MX 新格式：`TQ1_0=34, TQ2_0=35, MXFP4=39`
- 本项目特有：`Q2_0C=40`（自定义 2-bit 压缩）、`Q2_0=42`、`STQ1_0=43`（见 §3.6）

## 5. `GgufTensorInfo`

```csharp
public readonly struct GgufTensorInfo
{
    public GgmlTensorType Type { get; init; }
    public ulong Offset { get; init; }      // 相对 DataBase 的字节偏移
    public ulong[] Shape { get; init; }     // dim0 = 最内层（行）长度
    public ulong NumElements { get; }       // 形状各维乘积，checked 溢出
}
```

注意：**Offset 相对 `GgufFile.DataBase`，不是文件头**；`WeightStore` 用 `_gguf.DataBase + (long)info.Offset` 定位 payload。

## 6. 使用示例（`WeightStore` 视角）

```csharp
byte* payload = _gguf.DataBase + (long)info.Offset;   // 原始字节
float*  view   = _gguf.DataBase + (long)info.Offset;  // F32 直接强转
```

类型为 F32 时直接 `Buffer.MemoryCopy` 到 `NativeBuffer`；量化类型由 `WeightStore.Dequantize` 处理（见 `04-Model-Module.md` §WeightStore）。

## 7. 与 HyMT2Sharp 的对齐与差异

- 对齐：内存映射策略、`DataBase + Offset` 寻址、ggml 对齐规则、头解析顺序、STQ 兼容归一化逻辑同源。
- 差异：HyMT2Sharp 命名空间为 `Sdcb.*`，本项目为 `Flygon.*`；取值 API 更宽容（强转而非严格解码）。

## 8. 已知限制与后续方向

- 嵌套 GGUF 数组不支持（GGUF 规范本就禁止）。
- 未实现 GGUF 文件的写入/打包；本项目只做读取。
- 若未来出现 v4 版本需扩展 `version` 检查。
