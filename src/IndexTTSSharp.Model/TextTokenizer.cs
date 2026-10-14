using System.Text;
using Flygon.IndexTTSSharp.Gguf;

namespace Flygon.IndexTTSSharp.Model;

/// <summary>
/// BPE text tokenizer (from <c>indextts/utils/front.py</c> +
/// <c>indextts/utils/tokenizer.py</c>). The IndexTTS v2 tokenizer is a
/// SentencePiece BPE with 256 text tokens (0–255) plus special
/// <c>start_text_token=0</c>, <c>stop_text_token=1</c>, and a reserved
/// <c>unk_token</c>.
///
/// A combined GGUF stores the SentencePiece BPE merge rules under
/// <c>text_tokenizer.merges</c> and the vocab under <c>text_tokenizer.vocab</c>.
/// This port loads both and performs sentencepiece BPE tokenisation:
/// characters → byte-fallback → merge loop.
///
/// The normalizer maps full-width CJK punctuation to ASCII so the
/// SentencePiece model sees a consistent alphabet.
/// </summary>
public sealed class TextTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly int[] _mergePriorities;
    private readonly Dictionary<(string, string), int> _merges;
    private readonly int _unk;
    private readonly int _start;
    private readonly int _stop;

    /// <summary>
    /// Load the tokenizer from the combined GGUF.
    /// </summary>
    public static TextTokenizer FromGguf(GgufFile gguf)
    {
        int unk = gguf.GetInt32("text_tokenizer.unk_token", 128);
        int start = gguf.GetInt32("text_tokenizer.start_text_token", 0);
        int stop = gguf.GetInt32("text_tokenizer.stop_text_token", 1);

        string[] vocab = gguf.GetStringArray("text_tokenizer.vocab");
        string[] mergesRaw = gguf.GetStringArray("text_tokenizer.merges");

        Dictionary<string, int> vocabMap = new(StringComparer.Ordinal);
        for (int i = 0; i < vocab.Length; i++)
            vocabMap[vocab[i]] = i;

        Dictionary<(string, string), int> mergeMap = new();
        int[] mergePriority = new int[mergesRaw.Length];
        for (int i = 0; i < mergesRaw.Length; i++)
        {
            string[] parts = mergesRaw[i].Split(' ', StringSplitOptions.None);
            if (parts.Length == 2)
            {
                mergePriority[i] = i;
                mergeMap[(parts[0], parts[1])] = i;
            }
        }

        return new TextTokenizer(vocabMap, mergePriority, mergeMap, unk, start, stop);
    }

    private TextTokenizer(
        Dictionary<string, int> vocab,
        int[] mergePriorities,
        Dictionary<(string, string), int> merges,
        int unk, int start, int stop)
    {
        _vocab = vocab;
        _mergePriorities = mergePriorities;
        _merges = merges;
        _unk = unk;
        _start = start;
        _stop = stop;
    }

    /// <summary>
    /// Normalize a raw text input to ASCII/CJK form suitable for BPE.
    /// Mirrors <c>TextNormalizer.__call__</c> (punctuation → ASCII).
    /// </summary>
    public static string Normalize(string text)
    {
        StringBuilder sb = new StringBuilder(text.Length + 16);
        foreach (char c in text)
        {
            char mapped = NormalizeChar(c);
            sb.Append(mapped);
        }
        return sb.ToString();
    }

    private static char NormalizeChar(char c) => c switch
    {
        '：' or '；' or '；' or ',' or ';' or '，' or '、' => ',',
        '。' => '.',
        '！' => '!',
        '？' => '?',
        '\n' or '\r' => ' ',
        '·' or '—' or '～' or '~' => '-',
        _ when QuoteMarks.Contains(c) => '\'',
        _ => c,
    };

    private static readonly HashSet<char> QuoteMarks = new()
    {
        '\"', '\'', '“', '”', '‘', '’', '（', '）', '(', ')',
        '《', '》', '【', '】', '[', ']', '「', '」',
    };

    /// <summary>
    /// Tokenize <paramref name="text"/> into text-token ids (excluding
    /// start/stop, which are added by the GPT path).
    /// </summary>
    public int[] Tokenize(string text)
    {
        string normalized = Normalize(text);
        List<string> pieces = BpeTokenize(normalized);
        List<int> ids = new List<int>(pieces.Count + 2);
        ids.Add(_start);
        foreach (string piece in pieces)
            ids.Add(Lookup(piece));
        ids.Add(_stop);
        return ids.ToArray();
    }

    /// <summary>Split a sentence into BPE token strings.</summary>
    private List<string> BpeTokenize(string s)
    {
        List<string> tokens = new List<string>();
        // sentencepiece BPE: start with characters (or byte-fallback for
        // non-ASCII). For CJK, single characters are the base unit.
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (char.IsControl(c) || c is ' ' or '\t')
                continue;
            tokens.Add(c.ToString());
        }

        // BPE merge loop: repeatedly find the merge with the smallest priority.
        for (int mergeLevel = 0; mergeLevel < _mergePriorities.Length; mergeLevel++)
        {
            bool changed = false;
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                string a = tokens[i];
                string b = tokens[i + 1];
                if (_merges.TryGetValue((a, b), out int prio) && prio == mergeLevel)
                {
                    string merged = a + b;
                    tokens[i] = merged;
                    tokens.RemoveAt(i + 1);
                    changed = true;
                    i--;
                }
            }
            if (!changed)
                break;
        }
        return tokens;
    }

    private int Lookup(string piece)
    {
        // SentencePiece encodes each char as <0xXX>; if the raw piece isn't
        // in the vocab, fall back to per-character bytes.
        if (_vocab.TryGetValue(piece, out int id))
            return id;
        // try UTF-8 byte-fallback for non-ASCII
        foreach (byte b in Encoding.UTF8.GetBytes(piece))
        {
            string tag = $"<{b:X2}>";
            if (_vocab.TryGetValue(tag, out int byteId))
                return byteId;
        }
        return _unk;
    }

    public int UnkToken => _unk;
    public int StartTextToken => _start;
    public int StopTextToken => _stop;
}
