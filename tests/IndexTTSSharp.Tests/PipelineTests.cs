using Flygon.IndexTTSSharp.Model;
using Xunit;

namespace IndexTTSSharp.Tests;

public class PipelineTests
{
    [Fact]
    public void TextNormalizer_MapsFullWidthPunctuation()
    {
        string out_ = TextTokenizer.Normalize("你好，世界！");
        Assert.Equal("你好,世界!", out_);
    }

    [Fact]
    public void TextNormalizer_MapsQuotesAndBrackets()
    {
        string out_ = TextTokenizer.Normalize("你好（世界）[A]");
        Assert.Equal("你好'世界''A'", out_);
    }

    [Fact]
    public void GptConfig_Defaults_AreSane()
    {
        var cfg = new Flygon.IndexTTSSharp.Model.IndexTTSConfig();
        Assert.Equal(8, cfg.GptLayers);
        Assert.Equal(512, cfg.ModelDim);
        Assert.Equal(8193, cfg.StopMelToken);
        Assert.Equal(8192, cfg.StartMelToken);
        Assert.Equal(8194, cfg.NumberMelCodes);
    }
}
