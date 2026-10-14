using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Flygon.IndexTTSSharp.Gguf;
using Flygon.IndexTTSSharp.Kernels;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// Shared weight store: loads tensors out of a combined GGUF file by exact
/// name, dequantizing F32/quant payloads into owned buffers. Submodels pull
/// weights through <see cref="TryGetFloat"/> / <see cref="GetFloat"/> and keep
/// the result for the model's lifetime. Unmapped tensors are logged, not fatal.
/// </summary>
public sealed unsafe class WeightStore : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly Dictionary<string, NativeBuffer> _buffers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> _tensors = new(StringComparer.Ordinal);
    private readonly List<string> _unmapped = [];
    private readonly CpuThreadPool? _pool;

    private readonly struct Tensor
    {
        public Tensor(float* data, GgmlTensorType type, int[] shape, ulong numElements)
        {
            Data = data;
            Type = type;
            Shape = shape;
            NumElements = numElements;
        }
        public readonly float* Data;
        public readonly GgmlTensorType Type;
        public readonly int[] Shape;
        public readonly ulong NumElements;
    }

    public WeightStore(GgufFile gguf, CpuThreadPool? pool = null)
    {
        _gguf = gguf;
        _pool = pool;
    }

    public IReadOnlyList<string> Unmapped => _unmapped;

    public bool Has(string name) => _gguf.Tensors.ContainsKey(name);

    public GgufTensorInfo Info(string name) => _gguf.Tensors[name];

    /// <summary>
    /// Returns a stable <c>float*</c> view of the named tensor. F32 tensors are
    /// copied into an owned buffer (GGUF payloads may be F32 or any quant format);
    /// quant tensors are dequantized on first access.
    /// </summary>
    public float* GetFloat(string name, out int[] shape)
    {
        if (!_tensors.TryGetValue(name, out Tensor t))
        {
            t = Load(name);
            _tensors[name] = t;
        }
        shape = t.Shape;
        return t.Data;
    }

    public float* TryGetFloat(string name, out int[] shape)
    {
        if (!_gguf.Tensors.TryGetValue(name, out GgufTensorInfo info))
        {
            _unmapped.Add(name);
            shape = [];
            return null!;
        }
        shape = info.Shape.Length == 0 ? [1] : info.Shape.Select((u) => checked((int)u)).ToArray();
        return GetFloat(name, out _);
    }

    private Tensor Load(string name)
    {
        GgufTensorInfo info = _gguf.Tensors[name];
        int[] shape = info.Shape.Length == 0 ? [1] : info.Shape.Select((u) => checked((int)u)).ToArray();
        int count = checked((int)info.NumElements);
        NativeBuffer buf = new((nuint)(count * sizeof(float)));
        _buffers[name] = buf;
        float* dst = (float*)buf.Pointer;

        if (info.Type == GgmlTensorType.F32)
        {
            byte* src = _gguf.DataBase + (long)info.Offset;
            Buffer.MemoryCopy(src, dst, buf.Bytes, buf.Bytes);
            return new(dst, info.Type, shape, info.NumElements);
        }

        Dequantize(info.Type, _gguf.DataBase + (long)info.Offset, dst, count);
        return new(dst, info.Type, shape, info.NumElements);
    }

    private static void Dequantize(GgmlTensorType type, byte* src, float* dst, int count)
    {
        // Fallback for quantized weights: materialize floats. Q4_0/Q4_1/Q5_0/Q5_1/
        // Q8_0/Q8_1 are the most likely in a combined TTS GGUF. The K-variants and
        // STQ/Q2_0C reuse their dequantize-row helpers where available; otherwise
        // decode the generic ggml block layout on the fly.
        switch (type)
        {
            case GgmlTensorType.Q4_0:
                DequantQ4_0(src, dst, count);
                break;
            case GgmlTensorType.Q4_1:
                DequantQ4_1(src, dst, count);
                break;
            case GgmlTensorType.Q5_0:
                DequantQ5_0(src, dst, count);
                break;
            case GgmlTensorType.Q5_1:
                DequantQ5_1(src, dst, count);
                break;
            case GgmlTensorType.Q8_0:
                DequantQ8_0(src, dst, count);
                break;
            case GgmlTensorType.Q8_1:
                DequantQ8_1(src, dst, count);
                break;
            case GgmlTensorType.F16:
                DequantF16(src, dst, count);
                break;
            default:
                throw new NotSupportedException($"Dequantize not implemented for {type}");
        }
    }

    private static void DequantF16(byte* src, float* dst, int count)
    {
        ushort* s = (ushort*)src;
        for (int i = 0; i < count; i++)
            dst[i] = (float)BitConverter.UInt16BitsToHalf(s[i]);
    }

    // Q4_0: 32 floats per block; d (f16) + 16 bytes of packed 4-bit.
    private static void DequantQ4_0(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b / 2 + 2);
            Half d = BitCastHalf(p);
            byte* q = p + 2;
            float df = (float)d;
            for (int i = 0; i < b / 2; i++)
            {
                int low = q[i] & 0xF;
                int high = q[i] >> 4;
                dst[blk * b + i * 2] = (low - 8) * df;
                dst[blk * b + i * 2 + 1] = (high - 8) * df;
            }
        }
    }

    // Q4_1: 32 floats per block; d (f16) + m (f16) + 16 bytes of packed 4-bit.
    private static void DequantQ4_1(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b / 2 + 4);
            float d = (float)BitCastHalf(p);
            float m = (float)BitCastHalf(p + 2);
            byte* q = p + 4;
            for (int i = 0; i < b / 2; i++)
            {
                int low = q[i] & 0xF;
                int high = q[i] >> 4;
                dst[blk * b + i * 2] = d * low + m;
                dst[blk * b + i * 2 + 1] = d * high + m;
            }
        }
    }

    // Q5_0: 32 floats per block; d (f16) + qh (u8) + 20 bytes of packed 4-bit.
    private static void DequantQ5_0(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b / 2 + 4);
            float d = (float)BitCastHalf(p);
            byte qh = p[2];
            byte* q = p + 4;
            for (int i = 0; i < b / 2; i++)
            {
                int low = (q[i] & 0xF) | (((int)((qh >> (i % 2)) & 1) & 8) << 4);
                int high = ((q[i] >> 4) & 0xF) | (((int)((qh >> ((i + 2) % 4)) & 1) & 0x8) << 4);
                dst[blk * b + i * 2] = (low - 16) * d;
                dst[blk * b + i * 2 + 1] = (high - 16) * d;
            }
        }
    }

    // Q5_1: 32 floats per block; d (f16) + m (f16) + qh (u8) + 20 bytes.
    private static void DequantQ5_1(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b / 2 + 6);
            float d = (float)BitCastHalf(p);
            float m = (float)BitCastHalf(p + 2);
            byte qh = p[4];
            byte* q = p + 6;
            for (int i = 0; i < b / 2; i++)
            {
                int low = (q[i] & 0xF) | (((int)((qh >> (i % 2)) & 1) & 8) << 4);
                int high = ((q[i] >> 4) & 0xF) | (((int)((qh >> ((i + 2) % 4)) & 1) & 0x8) << 4);
                dst[blk * b + i * 2] = d * low + m;
                dst[blk * b + i * 2 + 1] = d * high + m;
            }
        }
    }

    // Q8_0: 32 floats per block; d (f16) + 32 bytes of int8.
    private static void DequantQ8_0(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b + 2);
            float d = (float)BitCastHalf(p);
            sbyte* q = (sbyte*)(p + 2);
            for (int i = 0; i < b; i++)
                dst[blk * b + i] = d * q[i];
        }
    }

    // Q8_1: 32 floats per block; d (f16) + m (f16) + qs_sum (i16) + 32 bytes int8.
    private static void DequantQ8_1(byte* src, float* dst, int count)
    {
        const int b = 32;
        int blocks = count / b;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* p = src + blk * (b + 4);
            float d = (float)BitCastHalf(p);
            float m = (float)BitCastHalf(p + 2);
            sbyte* q = (sbyte*)(p + 4);
            for (int i = 0; i < b; i++)
                dst[blk * b + i] = d * q[i] + m;
        }
    }

    private static Half BitCastHalf(byte* p)
    {
        ushort bits = (ushort)(p[0] | ((ushort)p[1] << 8));
        return BitConverter.UInt16BitsToHalf(bits);
    }

    public void Dispose()
    {
        foreach (NativeBuffer b in _buffers.Values)
            b.Dispose();
        _gguf.Dispose();
    }
}
