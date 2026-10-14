using System.Runtime.CompilerServices;

namespace Flygon.IndexTTSSharp.Kernels;

/// <summary>
/// 2-D convolution for CAMPPlus's FCM feature-extraction head (input is a
/// single-channel fbank/mel spectrogram, [1, H, W]). Weight is
/// <c>[C_out, C_in, KH, KW]</c>. Batch=1. PyTorch 'same' padding.
/// </summary>
public static unsafe class Conv2d
{
    public static void Forward(
        float* weight, float* bias,
        float* input, int inChannels, int inH, int inW,
        int outChannels, int outH, int outW,
        int kernelH, int kernelW, int strideH, int strideW,
        float* output)
    {
        int pH = (kernelH - 1) / 2;
        int pW = (kernelW - 1) / 2;
        for (int co = 0; co < outChannels; co++)
        {
            float bias0 = bias != null ? bias[co] : 0f;
            float* wCo = weight + co * inChannels * kernelH * kernelW;
            float* outCo = output + co * outH * outW;
            for (int oh = 0; oh < outH; oh++)
            {
                int iH = oh * strideH - pH;
                for (int ow = 0; ow < outW; ow++)
                {
                    int iW = ow * strideW - pW;
                    float acc = bias0;
                    for (int ci = 0; ci < inChannels; ci++)
                    {
                        float* wCi = wCo + ci * kernelH * kernelW;
                        float* inCi = input + ci * inH * inW;
                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            int r = iH + kh * strideH;
                            if (r < 0 || r >= inH) continue;
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int c = iW + kw * strideW;
                                if (c < 0 || c >= inW) continue;
                                acc += wCi[kh * kernelW + kw] * inCi[r * inW + c];
                            }
                        }
                    }
                    outCo[oh * outW + ow] = acc;
                }
            }
        }
    }
}
