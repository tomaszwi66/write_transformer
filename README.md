<div align="center">

# 🧠 write-transformer

**Build GPT from scratch - step by step, in 11 lessons.**

Interactive course. Write every component yourself. Train and generate in minutes.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tomaszwi66/write-transformer/pulls)

[Quick Start](#-quick-start) •
[How It Works](#-how-it-works) •
[11 Steps](#-11-steps) •
[Direct Modes](#-direct-modes) •
[Architecture](#-architecture) •
[FAQ](#-faq)

</div>

---
### 🇵🇱 Polska wersja dostępna! / Polish version available!
Dla osób z Polski przygotowałem specjalną, w pełni przetłumaczoną wersję interaktywnego kursu.
Aby ją uruchomić, użyj komendy:
`python write_transformer_PL.py`
---

## 🎯 Who is this for?

- **Beginners** - you want to understand how GPT/ChatGPT actually works inside
- **Students** - you're learning ML and want to build a Transformer yourself
- **Practitioners** - you want a clear, commented reference implementation

No prerequisites. Every concept explained with analogies, code, demos, and quizzes.

---

## ⚡ Quick Start

```bash
pip install torch numpy
git clone [https://github.com/tomaszwi66/write-transformer.git](https://github.com/tomaszwi66/write-transformer.git)
cd write-transformer
python write_transformer.py
```

That's it. The interactive course starts automatically.

---

## 📖 How It Works

Each of the 11 steps follows the same pattern:

* 📖 **Explanation** - what it is, why it matters, real-world analogy
* 📝 **Code** - commented, ready to type yourself
* 🔬 **Demo** - runs that component live, shows output
* ❓ **Quiz** - checks your understanding
* ⏎ **Enter** - next step

You don't read a textbook - you build a working GPT piece by piece.
At the end: you train the model on text and generate new sentences.

---

## 📚 11 Steps

| Step | Topic | What you build |
| :---: | :--- | :--- |
| **1** | BPE Tokenizer | Text → numbers (same algorithm as GPT-2) |
| **2** | Data Pipeline | Sliding window, next-token prediction setup |
| **3** | Embeddings | Token + positional embeddings |
| **4** | Self-Attention | Single-head Q, K, V with causal mask |
| **5** | Multi-Head Attention | Multiple perspectives, efficient implementation |
| **6** | Feed-Forward Network| Per-token processing with GELU |
| **7** | Transformer Block | Residual connections + LayerNorm |
| **8** | Full GPT Model | Stack of blocks + weight tying |
| **9** | Training Loop | AdamW, gradient clipping, perplexity tracking |
| **10** | Text Generation | Temperature, top-k, top-p (nucleus sampling) |
| **11** | Your Own Model | Summary + interactive playground |

**Sample output from Step 9 (Training):**
```text
🔬 DEMO: Training the model
Tokens: 1,847  Examples: 1,783  Batches: 222
─────────────────────────────────────────────
Epoch  1/15 │ Loss: 5.2341 │ PPL: 188.0 │ [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │ 2s
Epoch  3/15 │ Loss: 4.1023 │ PPL:  60.5 │ [█████░░░░░░░░░░░░░░░░░░░░░░░░░] │ 5s
Epoch  6/15 │ Loss: 3.2156 │ PPL:  24.9 │ [████████████░░░░░░░░░░░░░░░░░░] │ 9s
Epoch 15/15 │ Loss: 2.1847 │ PPL:   8.9 │ [██████████████████████████████] │ 21s
─────────────────────────────────────────────
✅ Done in 21.3s! Loss: 2.1847
```

**Sample output from Step 10 (Generation):**
```text
🌡️ Temperature: Low (0.3)
'Kot siedzi' → Kot siedzi na macie i obserwuje ptaki za oknem
'Stary człowiek' → Stary człowiek czyta książkę w cichej bibliotece

🌡️ Temperature: High (1.5)
'Kot siedzi' → Kot siedzi radośnie kominku piękne warzywa
'Stary człowiek' → Stary człowiek łagodnie kwitną gwiazdy melodie
```

---

## 🚀 Direct Modes

Skip the course and jump straight to training or generation:

```bash
# Train on your own text file
python write_transformer.py --train my_text.txt

# Train on pasted text (end with Ctrl+D)
python write_transformer.py --paste

# Interactive generation (requires trained model)
python write_transformer.py --interactive

# Custom architecture
python write_transformer.py --train data.txt --d_model 128 --n_heads 8 --n_layers 6 --epochs 50
```

### All CLI Options

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--train FILE` | — | Train on a text file |
| `--paste` | — | Paste text for training |
| `--interactive` | — | Interactive generation mode |
| `--epochs` | 30 | Training epochs |
| `--d_model` | 64 | Embedding dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--n_layers` | 4 | Number of transformer blocks |
| `--d_ff` | 256 | Feed-forward hidden dimension |
| `--lr` | 3e-4 | Learning rate |
| `--bpe_vocab_size`| 512 | BPE tokenizer vocabulary size |

### Interactive Commands

| Command | Effect |
| :--- | :--- |
| `/temp 0.5` | Change temperature |
| `/topk 20` | Change top-k filtering |
| `/quit` | Exit |

---

## 🏗️ Architecture

This implements exactly the same architecture as GPT-2/3/4. The only difference is scale.

```text
                  write-transformer     GPT-2 Small        GPT-4
─────────────────────────────────────────────────────────────────────────
Vocabulary        ~512 tokens           50,257 tokens      ~100K tokens
Dimensions        64 dimensions         768 dimensions     ~12,288 dimensions
Heads             4 heads               12 heads           ~96 heads
Layers            4 layers              12 layers          ~120 layers
Parameters        ~50K parameters       124M parameters    ~1.8T parameters
Data              25 sentences          8M web pages       the entire internet
Training Time     seconds on CPU        hours on 8 GPUs    months on 25K GPUs
```

```text
Token IDs [batch, seq_len]
           ↓
┌──────────────────────────────┐
│ Token Embedding              │ ID → learned vector
│ + Positional Embedding       │ position → learned vector
│ + Dropout                    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Transformer Block ×N         │ ← repeat N times
│  ┌────────────────────────┐  │
│  │ LayerNorm              │  │
│  │ Multi-Head Attention   │  │ Q, K, V + causal mask
│  │ + Residual Connection  │  │ x + attention(x)
│  ├────────────────────────┤  │
│  │ LayerNorm              │  │
│  │ Feed-Forward (GELU)    │  │ d_model → 4×d_model → d_model
│  │ + Residual Connection  │  │ x + ffn(x)
│  └────────────────────────┘  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ LayerNorm                    │
│ Linear → logits              │ d_model → vocab_size
│ (weight tying with embed)    │ same matrix encodes & decodes
└──────────────────────────────┘
               ↓
  Probabilities for next token
```

### Key Design Choices (matching GPT-2)

| Feature | Description |
| :--- | :--- |
| **Pre-Norm** | LayerNorm before (not after) each sublayer |
| **Weight Tying** | Embedding matrix = output projection matrix |
| **GELU** | Smooth activation (not ReLU) |
| **Learned Positional Embeddings**| Not sinusoidal |
| **Causal Mask** | Lower-triangular - tokens can only see the past |
| **AdamW** | Adam with decoupled weight decay |

---

## 🧪 Experiments to Try

### Easy
- Run the full course (mode 0) - understand every component
- Change temperature in interactive mode - compare outputs
- Train on a different text file

### Medium
- Reduce `d_model` to 8 - can it still learn?
- Increase `epochs` to 100 - does it overfit?
- Try `n_heads=1` vs `n_heads=8` - what changes?
- Add more training text - does generation improve?

### Advanced
- Comment out positional embeddings - what breaks?
- Remove residual connections - can it still train?
- Try different `bpe_vocab_size` values - effect on compression?
- Compare Pre-Norm vs Post-Norm (modify `TransformerBlock`)

---

## 🗺️ Learning Path

### Week 1: Build Understanding
| Day | What to do | Goal |
| :-: | :--- | :--- |
| **1** | Run the full course (11 steps) | See every component work |
| **2** | Re-run, read every code block | Understand the code |
| **3** | Train on your own text file | See it learn YOUR data |
| **4** | Play with interactive mode | Build intuition for generation |
| **5** | Change architecture params | Understand what each controls |

### Week 2: Go Deeper
| Day | What to do | Goal |
| :-: | :--- | :--- |
| **1** | Read SingleHeadAttention - add print statements | Understand Q, K, V flow |
| **2** | Read MultiHeadAttention - trace tensor shapes | Understand the reshape trick |
| **3** | Read GPTModel.generate() - understand sampling | Temperature, top-k, top-p |
| **4** | Modify the architecture - break things on purpose | Learn what each part does |
| **5** | Read the BPE tokenizer - trace a word through it | Understand subword tokenization |

### Week 3: Bridge to Production
| Resource | Description |
| :--- | :--- |
| **nanoGPT** | Same architecture, full scale |
| **Attention Is All You Need** | The original Transformer paper |
| **Karpathy - Let's build GPT** | Best video explanation |
| **Hugging Face** | Production models & tools |
| **The Illustrated Transformer** | Visual guide |

---

## 📁 Project Structure

```text
write-transformer/
├── write_transformer.py   ← everything (single file, by design)
├── README.md
├── README_PL.md           ← Polish version
├── LICENSE
├── requirements.txt
└── .gitignore
```

One file. On purpose. Open it in any editor and see everything at a glance.
No jumping between modules. No hidden complexity.

When you train a model, a `checkpoint/` directory is created:

```text
checkpoint/
├── model.pt               ← trained weights
├── tokenizer.json         ← BPE vocabulary & merges
└── config.json            ← architecture configuration
```

---

## ❓ FAQ

**Is this a real Transformer?**
Yes. Same architecture as GPT-2/3/4: multi-head attention, pre-norm, GELU, weight tying, BPE tokenizer. The only difference is scale.

**Do I need a GPU?**
No. The course trains on CPU in seconds. For larger texts, a GPU helps but isn't required.

**The model generates nonsense!**
Expected with tiny models and small data. Lower temperature (`/temp 0.3`), increase epochs, or add more training text. The point is understanding the architecture, not competing with ChatGPT.

**Can I train on English text?**
Yes. The default corpus is Polish, but the BPE tokenizer works with any language. Pass your own file with `--train`.

**What's the difference between this and nanoGPT?**
This is an educational tool - interactive course with explanations, quizzes, and demos. nanoGPT is a training tool - optimized for actually training GPT-2 scale models.

**Can I use this for teaching?**
Yes. MIT license. Credit appreciated but not required.

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch (`git checkout -b my-feature`)
3. Make your changes
4. Verify `python write_transformer.py` works in all modes
5. Create a Pull Request

### Ideas for Contributions
- [ ] Matplotlib attention heatmaps
- [ ] Jupyter notebook version with widgets
- [ ] English default corpus option
- [ ] Comparison with RNN/LSTM
- [ ] Sinusoidal positional encoding option
- [ ] Gradient flow visualization
- [ ] Token-by-token generation animation
- [ ] Web interface (Gradio/Streamlit)

---

## 📜 License

MIT - do whatever you want. See [LICENSE](LICENSE).

<br>

<div align="center">
If this helped you understand Transformers - leave a ⭐<br><br>
<i>You don't need to understand 1.8 trillion parameters.<br>
Understand 50 thousand - the rest is the same architecture, just bigger.</i>
</div>
