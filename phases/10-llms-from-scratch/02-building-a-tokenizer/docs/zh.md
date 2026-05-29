# 从零构建 Tokenizer

> 第 01 课给了你一个玩具。这一课给你一件武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01（Tokenizers: BPE, WordPiece, SentencePiece）
**Time:** 约 90 分钟

## Learning Objectives

- 构建一个生产级的 BPE tokenizer，能处理 Unicode、空白归一化和特殊 token
- 实现 byte-level fallback，让 tokenizer 可以编码任意输入（包括 emoji、CJK 和代码）而不产生 unknown token
- 加入 pre-tokenization 正则模式，在应用 BPE merges 之前按词边界切分文本
- 在语料上训练自定义 tokenizer，并在多语言文本上对比其与 tiktoken 的压缩率

## The Problem

第 01 课的 BPE tokenizer 在英文文本上能跑。现在丢点日文给它，或者 emoji，又或者带混合 tab 和空格的 Python 代码。

它就崩了。

不是因为 BPE 本身有问题，而是因为实现不完整。一个生产级 tokenizer 要能处理任意编码下的原始字节，要在切分前对 Unicode 做归一化，要管理那些永不参与合并的特殊 token，要把 pre-tokenization 与 subword 切分串起来，并且要快到不会卡住一条处理 15 万亿 token 的训练管线。

GPT-2 的 tokenizer 有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些都不是玩具数字。这些词表背后的 merge table 是在数百 GB 文本上训练出来的，而周边的机器——normalization、pre-tokenization、特殊 token 注入、chat template 格式化——才是把一个只能处理 "hello world" 的 tokenizer 和一个能吞下整个互联网的 tokenizer 区分开来的关键。

接下来你要构建的，就是这套机器。

## The Concept

### The Full Pipeline

一个生产级 tokenizer 不是一个算法，而是由五个阶段组成的管线，每个阶段解决一个不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的职责：

| 阶段 | 它在做什么 | 为什么重要 |
|-------|-------------|----------------|
| Normalize | NFKC Unicode，可选小写化，可选去重音 | "fi" ligature（U+FB01）会变成 "fi"（两个字符）。没有这步，同一个词会得到不同 token。 |
| Pre-Tokenize | 在 BPE 之前把文本切成块 | 防止 BPE 跨词边界合并。"the cat" 永远不应该产生一个叫 "e c" 的 token。 |
| BPE Merge | 在字节序列上应用学到的合并规则 | 核心压缩。把原始字节变成 subword token。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、chat template 标记 | 这些 token 有固定 ID，永远不参与 BPE 合并。模型需要它们来表达结构。 |
| ID Mapping | 把 token 字符串转成整数 ID | 模型看到的是整数，不是字符串。 |

### Byte-Level BPE

第 01 课的 tokenizer 是在 UTF-8 字节上工作的。这个方向是对的。但我们略过了一件重要的事：当这些字节不是合法 UTF-8 时怎么办？

Byte-level BPE 解决这个问题的方法是把每一个可能的字节值（0-255）都当成合法 token。基础词表恰好有 256 项。任何文件——文本、二进制、损坏的——都能被 tokenize，且不会产生 unknown token。

GPT-2 加了一个小技巧：把每个字节映射到一个可打印的 Unicode 字符，让词表保持人类可读。在他们的映射里，字节 0x20（空格）变成字符 "G"。这纯粹是表面工夫，算法本身并不关心。

真正的威力在于：byte-level BPE 能处理地球上的任何语言。中文字符每个 3 个 UTF-8 字节，日文 3 到 4 字节，阿拉伯语、天城文、emoji——都只是字节序列。BPE 算法在这些字节序列里找模式的方式，和在英文 ASCII 字节里找模式完全一样。

### Pre-Tokenization

在 BPE 接触你的文本之前，需要先把它切成块。这能防止合并算法生成跨词边界的 token。

GPT-2 用一个正则模式来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式按缩写（"don't" 切成 "don" + "'t"）、可选前导空格的单词、数字、标点和空白来切分。前导空格保留在词上——所以 "the cat" 会变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 用 SentencePiece，完全跳过正则。它把原始字节流当成一条长序列，让 BPE 算法自己找边界。这样更简单，但给了 BPE 更多自由去创建跨词 token。

这个选择很关键。GPT-2 的正则阻止 tokenizer 学到这种事：一个词末尾的 "the" 和下一个词开头的 "the" 应该合并。SentencePiece 允许这件事，有时压缩效率更高，但 token 的可解释性更差。

### Special Tokens

每个生产级 tokenizer 都会保留一些 token ID 给结构性标记：

| Token | 用途 | 使用方 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列起始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | batch 对齐用的 padding | BERT, T5 |
| `[UNK]` | unknown token（byte-level BPE 让它消失） | BERT, WordPiece |
| `<\|im_start\|>` | chat 消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | chat 消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户回合标记 | Llama 3 |
| `<\|assistant\|>` | 助手回合标记 | Llama 3 |

特殊 token 永远不会被 BPE 切分。它们在合并算法运行之前就被精确匹配出来，替换为对应的固定 ID，周围的文本再正常 tokenize。

### Chat Templates

这是大多数人困惑、大多数实现翻车的地方。

当你给一个 chat 模型发消息时，API 接收的是一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到的是一条扁平的 token 序列。chat template 就是用特殊 token 把消息转成那条扁平序列的格式。每个模型做法都不一样：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

template 写错了，模型就只会输出垃圾。它是按一个精确格式训练出来的，任何偏差——少一个换行、换错一个 token、多一个空格——都会让输入掉出训练分布之外。

### Speed

Python 太慢，撑不起生产级 tokenization。

tiktoken（OpenAI）是 Rust 写的，配 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些方案相比纯 Python 能达到 10 到 100 倍加速。

给个直观参照：用快一点的纯 Python 以每秒 100 万 token 的速度，给 Llama 3 预训练 tokenize 15 万亿 token，要花 174 天。用 Rust 每秒 1 亿 token，只要 1.7 天。

你现在用 Python 来构建，是为了理解算法。在生产环境里，你会用一个编译好的实现，只在 Python wrapper 这一层动手。

## Build It

### Step 1: Byte-Level Encoding

地基。把任意字符串转成字节序列，把每个字节映射到可打印字符以便展示，再把这个过程反过来。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测一下，看看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 字节。"你好" 是 6 字节（每字 3 字节）。火焰 emoji 是 4 字节。byte-level tokenizer 不在乎是什么语言，字节就是字节。

### Step 2: Pre-Tokenizer with Regex

用 GPT-2 的正则模式把文本切成块。每个块由 BPE 独立 tokenize。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库的 `re` 不支持，所以这里退回到 ASCII 字符类。生产级多语言 tokenizer，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格留在词上。缩写在撇号处被切开。标点自成一块。BPE 永远不会跨这些边界合并 token。

### Step 3: BPE on Byte Sequences

第 01 课里那个核心算法，但现在是在 pre-tokenize 出来的块上独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special Token Handling

特殊 token 需要精确匹配和固定 ID，它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: Full Tokenizer Class

把所有东西串起来：normalize、按特殊 token 切分、pre-tokenize、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6: Multilingual Test

真正的考验。把英文、中文、emoji、代码一起丢给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个 3 字节，emoji 是 4 字节。这些都不会让 tokenizer 崩，也都不会产生 unknown token。这就是 byte-level BPE 的威力。

## Use It

### Comparing Real Tokenizers

加载来自 Llama 3、GPT-4 和 Mistral 的真实 tokenizer。看看每个怎么处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本在不同 tokenizer 下 token 数差别很大。Llama 3 词表 128K，对常见模式合并得更激进。GPT-4 词表 100K，居中。Mistral 词表 32K，token 数更多，但 embedding 层更小。

权衡永远是同一套：词表越大，序列越短，但参数也越多。

## Ship It

这一课会产出一个用于构建和调试生产级 tokenizer 的 prompt，参见 `outputs/prompt-tokenizer-builder.md`。

## Exercises

1. **Easy：** 加一个 `get_token_bytes(id)` 方法，返回任意 token ID 对应的原始字节。用它检查你最常被合并出来的那些 token 实际代表什么。
2. **Medium：** 实现 Llama 风格的 pre-tokenizer，按空白和数字切分但保留前导空格。在同一份语料上对比它和 GPT-2 正则方案训出来的词表。
3. **Hard：** 加一个 chat template 方法，接收一个 `{"role": ..., "content": ...}` 消息列表，按 Llama 3 chat 格式产出正确的 token 序列。和 HuggingFace 的实现做对照测试。

## Key Terms

| 术语 | 大家通常怎么说 | 它实际是什么 |
|------|----------------|----------------------|
| Byte-level BPE | "在字节上工作的 tokenizer" | 基础词表是 256 个字节值的 BPE——能处理任何输入而不产生 unknown token |
| Pre-tokenization | "BPE 之前的切分" | 基于正则或规则的切分，防止 BPE 跨词边界合并 |
| NFKC normalization | "Unicode 清洗" | canonical decomposition 加 compatibility composition——"fi" ligature 变成 "fi"，全角 "A" 变成 "A" |
| Chat template | "消息怎么变成 token" | 把一组 role/content 消息转成扁平 token 序列的精确格式——按模型而异，必须严格匹配训练格式 |
| Special tokens | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、chat 标记——在合并之前精确匹配出来 |
| Fertility | "每个词多少 token" | 输出 token 数与输入词数的比值——GPT-4 上英文是 1.3，韩文 2 到 3，比值越高越浪费 context |
| tiktoken | "OpenAI 的 tokenizer" | Rust BPE 实现，带 Python 绑定——比纯 Python 快 10 到 100 倍 |
| Merge table | "词表" | 训练时学到的字节对合并的有序列表——这就是 tokenizer 学到的全部知识 |

## Further Reading

- [OpenAI tiktoken source](https://github.com/openai/tiktoken)——GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)——支持 BPE、WordPiece、Unigram 的 Rust tokenizer 库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783)——128K 词表与 tokenizer 训练细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226)——语言无关的 tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py)——最早的 byte-to-Unicode 映射
