using Flygon.IndexTTSSharp.Model;

namespace Flygon.IndexTTSSharp.Cli;

internal static class Program
{
    private static int Main(string[] args)
    {
        string? model = Get(args, "--model");
        string? refPath = Get(args, "--ref");
        string? text = Get(args, "--text");
        string? outPath = Get(args, "--out") ?? "out.wav";
        int threads = GetInt(args, "--threads", 0);

        if (model is null || refPath is null || text is null)
        {
            Console.WriteLine("IndexTTSSharp");
            Console.WriteLine("Usage:");
            Console.WriteLine("  IndexTTSSharp --model <gguf> --ref <ref.wav> --text \"...\" [--out out.wav] [--threads N]");
            return 1;
        }

        using IndexTTS tts = new(model, threads);
        float[] refAudio = Audio.LoadWav22k(refPath);
        float[] wav = tts.Synth(text, refAudio, outPath);
        Console.WriteLine($"Wrote {wav.Length} samples @ 22050 Hz to {outPath}");
        return 0;
    }

    private static string? Get(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    private static int GetInt(string[] args, string name, int fallback) =>
        int.TryParse(Get(args, name), out int value) ? value : fallback;
}
