using System.Runtime.CompilerServices;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// 1-D convolution + transpose + weight-norm folding. PyTorch layouts:
/// input / output are channel-major <c>[C, T]</c> (like a 2-D <c>Conv1d</c>
/// tensor with batch dropped, batch=1 throughout IndexTTS).
/// Weight layout is <c>[C_out, C_in/groups, kernel]</c> with the kernel index
/// fastest — matching <c>nn.Conv1d</c>/<c>nn.ConvTranspose1d</c> state_dict.
/// </summary>
public static unsafe class Conv1d
{
    /// <param name="weight">[C_out, C_in, K]</param>
    /// <param name="bias">[C_out]</param>
    /// <param name="input">[C_in, T_in]</param>
    /// <param name="output">[C_out, T_out]</param>
    /// Dilation/dilation-padded symmetric conv. Pad = (K-1)/2 * dilation.
    public static void Forward(
        float* weight, float* bias,
        float* input, int inChannels, int inLen,
        int outChannels, int outLen,
        int kernel, int stride, int dilation,
        float* output)
    {
        int pad = (kernel - 1) / 2 * dilation;
        for (int co = 0; co < outChannels; co++)
        {
            float bias0 = bias != null ? bias[co] : 0f;
            float* wCo = weight + co * inChannels * kernel;
            float* outRow = output + co * outLen;
            for (int t = 0; t < outLen; t++)
            {
                float acc = bias0;
                int inStart = t * stride - pad;
                for (int ki = 0; ki < kernel; ki++)
                {
                    int inT = inStart + ki * dilation;
                    if (inT < 0 || inT >= inLen)
                        continue;
                    float* wK = wCo + ki * inChannels;
                    for (int ci = 0; ci < inChannels; ci++)
                        acc += wK[ci * kernel + ki] * input[ci * inLen + inT];
                }
                outRow[t] = acc;
            }
        }
    }

    /// <param name="weight">[C_in, C_out, K] — PyTorch ConvTranspose1d weight is transposed.</param>
    /// <param name="input">[C_in, T_in]</param>
    /// <param name="output">[C_out, T_out]</param>
    public static void TransposeForward(
        float* weight, float* bias,
        float* input, int inChannels, int inLen,
        int outChannels, int outLen,
        int kernel, int stride, int outputPad,
        float* output)
    {
        // Zero the output first.
        for (int i = 0; i < outChannels * outLen; i++)
            output[i] = 0f;
        if (bias != null)
        {
            for (int co = 0; co < outChannels; co++)
            {
                float b = bias[co];
                float* outRow = output + co * outLen;
                for (int t = 0; t < outLen; t++)
                    outRow[t] += b;
            }
        }
        int pad = (kernel - 1) / 2;
        for (int ci = 0; ci < inChannels; ci++)
        {
            float* wCi = weight + ci * outChannels * kernel;
            float* inRow = input + ci * inLen;
            for (int t = 0; t < inLen; t++)
            {
                float v = inRow[t];
                if (v == 0f)
                    continue;
                int outStart = t * stride - pad;
                for (int ki = 0; ki < kernel; ki++)
                {
                    int outT = outStart + ki;
                    if (outT < 0 || outT >= outLen)
                        continue;
                    float* wK = wCi + ki * outChannels;
                    for (int co = 0; co < outChannels; co++)
                        output[co * outLen + outT] += v * wK[co * kernel + ki];
                }
            }
        }
    }

    /// <summary>
    /// Fold weight-norm parameters: <c>w' = w * g / ‖w‖</c> per output channel.
    /// In-place on <paramref name="weight"/> ([C_out, C_in, K]). <paramref name="g"/> is
    /// the learned gain [C_out]. Called once at load, before inference.
    /// </summary>
    public static void FoldWeightNorm(float* weight, float* g, int outChannels, int inChannels, int kernel)
    {
        int perChannel = inChannels * kernel;
        for (int co = 0; co < outChannels; co++)
        {
            float* w = weight + co * perChannel;
            float norm = 0f;
            for (int i = 0; i < perChannel; i++)
                norm += w[i] * w[i];
            norm = MathF.Sqrt(norm);
            if (norm == 0f)
                continue;
            float scale = (g != null ? g[co] : 1f) / norm;
            for (int i = 0; i < perChannel; i++)
                w[i] *= scale;
        }
    }
}
