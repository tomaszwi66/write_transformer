#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   TRANSFORMER FROM SCRATCH — STEP BY STEP COURSE                ║
║                                                                  ║
║   After completing this course you'll be able to write           ║
║   and run a Transformer / GPT from scratch.                      ║
║                                                                  ║
║   11 steps. Each step:                                           ║
║   1. Explanation — what it is, why, analogy                      ║
║   2. Code — ready, commented, to type yourself                   ║
║   3. Demo — runs that piece live and shows results               ║
║   4. Quiz — checks understanding                                ║
║   5. Enter → next step                                           ║
║                                                                  ║
║   At the end: everything works together — train a model          ║
║   on YOUR OWN text and generate.                                 ║
║                                                                  ║
║   Requirements: pip install torch numpy                          ║
║                                                                  ║
║   Usage:                                                         ║
║     python write_transformer.py                # step by step    ║
║     python write_transformer.py --train file.txt   # direct train║
║     python write_transformer.py --interactive      # generate    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
import argparse
import json
import re
from collections import Counter


# ================================================================
#  CONFIGURATION
# ================================================================
DEFAULT_CONFIG = {
    "vocab_size": 0,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 256,
    "max_seq_len": 128,
    "dropout": 0.1,
    "batch_size": 16,
    "epochs": 30,
    "lr": 3e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_interval": 5,
    "bpe_vocab_size": 512,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "max_gen_len": 200,
}

DEFAULT_TEXT = """
The cat sits on the mat and watches the birds outside. The birds sing beautiful melodies.
The dog lies on the carpet next to the fireplace. The fire crackles softly in the hearth.
The small cat chases the big dog around the garden. They play together until dusk.
The old man reads a book in the quiet library. The library is peaceful and calm.
The young girl writes a story about a brave knight. The knight saves the kingdom.
The rain falls gently on the green meadow. The flowers bloom in spring in all colors.
The sun shines brightly in the clear blue sky. The clouds drift slowly over the city.
The teacher explains the lesson to the students in class. The students listen carefully.
The chef cooks a delicious meal in the big kitchen. The kitchen smells wonderful.
The musician plays a beautiful melody on the old piano. The audience listens in silence.
The scientist discovers a new formula in the laboratory. The discovery changes everything.
The painter creates a masterpiece with bright colors. The gallery displays the painting.
The farmer grows fresh vegetables in the vast field. The harvest is plentiful this year.
The children play happily in the park after school. They laugh and run together.
The doctor examines the patient carefully. The patient feels much better after the visit.
The fisherman catches many fish in the deep sea. The boat rocks gently on the waves.
The writer works on a new novel every morning. The story grows page by page.
The gardener plants beautiful roses in the garden. The roses bloom red and white.
The astronomer watches the stars through a telescope. The night sky is magnificent.
The baker makes fresh bread every morning before dawn. The bakery smells wonderful.
The traveler explores new countries and cultures. Every journey teaches something new.
The student studies hard for the important exam. Hard work leads to success.
The architect designs a modern building for the city. The design is innovative and bold.
The pilot flies the airplane across the vast ocean. The view from above is breathtaking.
The librarian organizes thousands of books on the shelves. Knowledge fills every corner.
"""


# ================================================================
#  COURSE UTILITIES
# ================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def wait(msg="Press Enter to continue..."):
    try:
        input(f"\n  ⏎ {msg}")
    except (EOFError, KeyboardInterrupt):
        print("\n  Course interrupted.")
        sys.exit(0)


def show_header(step, total, title):
    print(f"\n{'═'*65}")
    print(f"  STEP {step}/{total}: {title}")
    print(f"{'═'*65}")


def show_explanation(text):
    print()
    for line in text.strip().split('\n'):
        print(f"  {line}")


def show_code(code):
    print(f"\n  {'─'*60}")
    print(f"  📝 CODE TO WRITE:")
    print(f"  {'─'*60}")
    for line in code.strip().split('\n'):
        print(f"  │ {line}")
    print(f"  {'─'*60}")


def show_demo(title):
    print(f"\n  🔬 DEMO: {title}")
    print(f"  {'·'*50}")


def quiz(question, options, correct, explanation):
    print(f"\n  ❓ {question}")
    for i, opt in enumerate(options):
        print(f"     {i+1}. {opt}")
    try:
        ans = input("     Your answer (1-4): ").strip()
        if ans == str(correct):
            print(f"     ✅ Correct!")
        else:
            print(f"     ❌ Answer: {correct}. {options[correct-1]}")
        print(f"     💡 {explanation}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n     Answer: {correct}. {options[correct-1]}")


# ================================================================
#  STEP 1: BPE TOKENIZER
# ================================================================

class BPETokenizer:
    """Byte-Pair Encoding — same algorithm as GPT-2."""

    def __init__(self, vocab_size=512):
        self.target_vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self.trained = False
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

    def train(self, text, verbose=True):
        if verbose:
            print(f"     Training BPE... (text: {len(text):,} chars)")

        self.vocab = {
            self.pad_token: 0, self.bos_token: 1,
            self.eos_token: 2, self.unk_token: 3,
        }

        chars = sorted(set(text))
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        words = text.split()
        word_freqs = Counter(words)

        splits = {}
        for word, freq in word_freqs.items():
            splits[tuple(word) + ('</w>',)] = freq

        if '</w>' not in self.vocab:
            self.vocab['</w>'] = len(self.vocab)

        num_merges = self.target_vocab_size - len(self.vocab)
        self.merges = {}

        for merge_idx in range(num_merges):
            pair_counts = Counter()
            for word_tokens, freq in splits.items():
                for i in range(len(word_tokens) - 1):
                    pair_counts[(word_tokens[i], word_tokens[i+1])] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            if pair_counts[best_pair] < 2:
                break

            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            new_splits = {}
            for word_tokens, freq in splits.items():
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and
                            word_tokens[i] == best_pair[0] and
                            word_tokens[i+1] == best_pair[1]):
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                new_splits[tuple(new_word)] = freq
            splits = new_splits

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"     Merge {merge_idx+1}: "
                      f"'{best_pair[0]}'+'{best_pair[1]}'→'{merged}'")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True

        if verbose:
            print(f"     ✅ Vocab: {len(self.vocab)} tokens, "
                  f"{len(self.merges)} merges")

    def _apply_merges(self, tokens):
        while True:
            best_pair = None
            best_merge_rank = len(self.merges)
            merge_keys = list(self.merges.keys())

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    rank = merge_keys.index(pair)
                    if rank < best_merge_rank:
                        best_merge_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            merged = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i+1] == best_pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        if not self.trained:
            raise RuntimeError("Tokenizer not trained!")
        ids = [self.vocab[self.bos_token]]
        for word in text.split():
            tokens = list(word) + ['</w>']
            tokens = self._apply_merges(tokens)
            for t in tokens:
                ids.append(self.vocab.get(t, self.vocab[self.unk_token]))
        ids.append(self.vocab[self.eos_token])
        return ids

    def decode(self, ids):
        tokens = []
        for id in ids:
            t = self.inverse_vocab.get(id, self.unk_token)
            if t in (self.pad_token, self.bos_token, self.eos_token):
                continue
            tokens.append('?' if t == self.unk_token else t)
        return ''.join(tokens).replace('</w>', ' ').strip()

    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "target_vocab_size": self.target_vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for k, v in data["merges"].items():
            parts = k.split("|||")
            self.merges[(parts[0], parts[1])] = v
        self.target_vocab_size = data["target_vocab_size"]
        self.trained = True


def step_1_tokenizer(text):
    show_header(1, 11, "BPE TOKENIZER")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  WHAT IS IT?                                                 ║
║                                                              ║
║  Computers don't understand text. They understand numbers.   ║
║  A tokenizer converts text to numbers and back.              ║
║                                                              ║
║  "the cat sits on the mat"                                   ║
║       ↓ encode()                                             ║
║  [1, 45, 23, 67, 45, 89, 2]                                 ║
║       ↓ model processes                                      ║
║  [1, 45, 23, 67, 45, 89, 34, 2]                             ║
║       ↓ decode()                                             ║
║  "the cat sits on the mat carpet"                            ║
║                                                              ║
║  TYPES OF TOKENIZERS:                                        ║
║  • Word-level: 1 word = 1 token (simple, huge vocab)         ║
║  • BPE: subword — "unhappiness" → "un"+"happiness"           ║
║    ↑ THIS IS WHAT GPT-2, GPT-3, GPT-4 USE                   ║
║  • SentencePiece: statistical (LLaMA, T5)                    ║
║                                                              ║
║  BPE ALGORITHM:                                              ║
║  1. Start with individual chars: ['c','a','t']               ║
║  2. Count pairs: ('t','h') occurs 15 times                   ║
║  3. Merge most frequent: 't'+'h' → 'th'                     ║
║  4. Repeat until target vocab size reached                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.merges = {}  # (token_a, token_b) → merged_token

    def train(self, text):
        # 1. Add all unique characters to vocab
        for ch in sorted(set(text)):
            self.vocab[ch] = len(self.vocab)

        # 2. Split each word into characters + end-of-word marker
        # "cat" → ('c', 'a', 't', '</w>')

        # 3. In a loop:
        #    a) Count all adjacent pairs
        #    b) Merge the most frequent pair
        #    c) Add to vocab
        #    d) Repeat until vocab_size

    def encode(self, text):
        # Word → chars → apply merges → convert to IDs
        return [self.vocab[token] for token in tokens]

    def decode(self, ids):
        # IDs → tokens → join → replace '</w>' with spaces
        return text
    """)

    wait("Enter → see the tokenizer in action...")

    show_demo("BPE Tokenizer")

    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.train(text)

    test = "The cat sits on the mat"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)

    print(f"\n     Text:      '{test}'")
    print(f"     Encoded:   {ids}")
    print(f"     Decoded:   '{decoded}'")
    print(f"     Vocab size: {len(tokenizer.vocab)}")
    print(f"     Compression: {len(test)}/{len(ids)} = "
          f"{len(test)/len(ids):.1f} chars/token")

    quiz(
        "What does BPE do differently from a word-level tokenizer?",
        [
            "Splits text into sentences",
            "Splits rare words into smaller known pieces",
            "Converts text to binary",
            "Removes special characters"
        ],
        2,
        "BPE splits rare words into subwords. 'unhappiness' → "
        "'un'+'happiness'. This lets it handle any new word."
    )

    return tokenizer


# ================================================================
#  STEP 2: DATA PIPELINE
# ================================================================

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n_examples = max(0, len(self.data) - seq_len)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def step_2_data_pipeline(tokenizer, text):
    show_header(2, 11, "DATA PIPELINE")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  WHAT IS IT?                                                 ║
║                                                              ║
║  The model learns to PREDICT THE NEXT TOKEN.                 ║
║  That's GPT's only objective. All "intelligence" comes       ║
║  from this.                                                  ║
║                                                              ║
║  HOW WE CREATE TRAINING DATA:                                ║
║                                                              ║
║  Tokenized text: [10, 20, 30, 40, 50, 60, 70]               ║
║                                                              ║
║  We cut it into overlapping windows (seq_len=4):             ║
║                                                              ║
║  Example 1:  Input:  [10, 20, 30, 40]                        ║
║              Target: [20, 30, 40, 50]                        ║
║                                                              ║
║  Example 2:  Input:  [20, 30, 40, 50]                        ║
║              Target: [30, 40, 50, 60]                        ║
║                                                              ║
║  At EVERY position the model predicts the next token:        ║
║                                                              ║
║  Input:  [The,  cat, sits, ?   ]                             ║
║  Target: [cat, sits, on,   EOS ]                             ║
║           ↑     ↑     ↑     ↑                                ║
║  The model must guess each of these!                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n_examples = len(self.data) - seq_len
        # How many windows fit in the text

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + seq_len + 1]
        x = chunk[:-1]    # input: everything except last
        y = chunk[1:]      # target: everything except first
        return x, y

# DataLoader automatically groups examples into batches:
loader = DataLoader(dataset, batch_size=16, shuffle=True)
    """)

    wait("Enter → see the pipeline in action...")

    show_demo("Data Pipeline")

    token_ids = tokenizer.encode(re.sub(r'\s+', ' ', text).strip())
    dataset = TextDataset(token_ids, seq_len=8)

    x, y = dataset[0]
    x_words = [tokenizer.inverse_vocab.get(i.item(), '?') for i in x]
    y_words = [tokenizer.inverse_vocab.get(i.item(), '?') for i in y]

    print(f"\n     Tokens in corpus: {len(token_ids):,}")
    print(f"     Examples (seq_len=8): {len(dataset):,}")
    print(f"\n     Example 1:")
    print(f"       Input (x):  {x.tolist()}")
    print(f"       Tokens:     {x_words}")
    print(f"       Target (y): {y.tolist()}")
    print(f"       Tokens:     {y_words}")
    print(f"\n     ↑ Model sees x and must predict y")
    print(f"       At pos 0: sees '{x_words[0]}' → target: '{y_words[0]}'")
    print(f"       At pos 1: sees '{x_words[0]},{x_words[1]}' → target: '{y_words[1]}'")

    quiz(
        "Why is the target shifted 1 position to the right?",
        [
            "Because the model should predict the previous token",
            "Because the model should predict the NEXT token — target is the future",
            "To save memory",
            "It's a bug in the code"
        ],
        2,
        "The target is the sequence shifted by 1 to the right. At every position "
        "the model must predict what comes NEXT. That's GPT's only objective!"
    )

    return token_ids


# ================================================================
#  STEP 3: EMBEDDINGS
# ================================================================

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        tok = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos = self.pos_emb(positions)
        return self.dropout(tok + pos)


def step_3_embeddings(tokenizer):
    show_header(3, 11, "EMBEDDINGS (TOKEN + POSITION)")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  WHAT IS IT?                                                 ║
║                                                              ║
║  A token ID is just a number (e.g. 42). The model needs      ║
║  a VECTOR — a list of numbers it can compute with.           ║
║                                                              ║
║  TOKEN EMBEDDING:                                            ║
║  ID 42 ("cat") → [0.12, -0.34, 0.56, 0.78, ...]            ║
║  ID 15 ("dog") → [0.11, -0.31, 0.58, 0.75, ...]            ║
║  ↑ Similar words → similar vectors (model learns this!)      ║
║                                                              ║
║  POSITIONAL EMBEDDING:                                       ║
║  Attention doesn't know order — treats tokens as a SET.      ║
║  We must tell the model WHERE each token is.                 ║
║                                                              ║
║  Position 0 → [0.00, 1.00, 0.00, 1.00, ...]                ║
║  Position 1 → [0.84, 0.54, 0.01, 0.99, ...]                ║
║  Position 2 → [0.91, 0.42, 0.02, 0.98, ...]                ║
║                                                              ║
║  RESULT = token_embedding + position_embedding               ║
║                                                              ║
║  GPT-2 uses LEARNED positional embeddings (so do we).        ║
║  The original Transformer used sinusoidal (fixed).           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        # Lookup table: token ID → vector of d_model dimensions
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Lookup table: position → vector of d_model dimensions
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len] — token IDs
        tok = self.token_emb(x)                       # [batch, seq_len, d_model]
        positions = torch.arange(seq_len).unsqueeze(0) # [1, seq_len]
        pos = self.pos_emb(positions)                   # [1, seq_len, d_model]
        return self.dropout(tok + pos)                  # sum!
    """)

    wait("Enter → see embeddings in action...")

    show_demo("Embeddings")

    vocab_size = len(tokenizer.vocab)
    d_model = 32
    emb = TransformerEmbedding(vocab_size, d_model, max_seq_len=64)

    test_ids = tokenizer.encode("The cat sits on the mat")
    x = torch.tensor([test_ids])
    output = emb(x)

    print(f"\n     Input (token IDs): {test_ids}")
    print(f"     Input shape:  {x.shape}  (batch=1, seq_len={len(test_ids)})")
    print(f"     Output shape: {output.shape}  (batch=1, seq_len={len(test_ids)}, d_model={d_model})")
    print(f"\n     Vector for token 0 (first 8 values):")
    print(f"     {[round(v, 3) for v in output[0, 0, :8].tolist()]}")
    print(f"\n     Vector for token 1 (first 8 values):")
    print(f"     {[round(v, 3) for v in output[0, 1, :8].tolist()]}")
    print(f"\n     ↑ Each token is now a vector of {d_model} numbers")
    print(f"       The model can compute with these!")

    quiz(
        "Why do we need positional embeddings?",
        [
            "To make the model faster",
            "Because attention treats tokens as a SET, it doesn't know order",
            "To reduce the number of parameters",
            "To make text shorter"
        ],
        2,
        "Attention computes similarity between EVERY pair of tokens. "
        "Without position info, 'the cat sits on the mat' = 'mat the on sits cat the'!"
    )


# ================================================================
#  STEP 4: SELF-ATTENTION (single head)
# ================================================================

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_head = d_head

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights


def step_4_single_attention():
    show_header(4, 11, "SELF-ATTENTION (single head)")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  THE HEART OF THE TRANSFORMER!                               ║
║                                                              ║
║  Problem: "The cat sat on the mat because IT was tired"      ║
║  Question: does "it" refer to "cat" or "mat"?                ║
║                                                              ║
║  Self-Attention lets each token "look at" other tokens       ║
║  and decide which ones matter.                               ║
║                                                              ║
║  Q-K-V MECHANISM:                                            ║
║  Each token produces 3 vectors:                              ║
║                                                              ║
║  Q (Query): "What am I looking for?"                         ║
║    → Like a search engine query                              ║
║                                                              ║
║  K (Key): "What do I contain?"                               ║
║    → Like a page title                                       ║
║                                                              ║
║  V (Value): "What information do I carry?"                   ║
║    → Like page content                                       ║
║                                                              ║
║  score = Q · K^T      (how well query matches the offer)     ║
║  score = score / √d   (scaling so softmax stays stable)      ║
║  weights = softmax(score)  (normalize to probabilities)      ║
║  output = weights · V      (weighted sum of values)          ║
║                                                              ║
║  CAUSAL MASK:                                                ║
║  Token at position 3 CANNOT look at position 4, 5, 6...     ║
║  Because during generation those tokens don't exist yet!     ║
║                                                              ║
║  Mask matrix (1=visible, 0=blocked):                         ║
║  [[1, 0, 0, 0],                                             ║
║   [1, 1, 0, 0],                                             ║
║   [1, 1, 1, 0],                                             ║
║   [1, 1, 1, 1]]                                             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        # Linear projections — LEARNED matrices
        self.W_q = nn.Linear(d_model, d_head, bias=False)  # Query
        self.W_k = nn.Linear(d_model, d_head, bias=False)  # Key
        self.W_v = nn.Linear(d_model, d_head, bias=False)  # Value
        self.d_head = d_head

    def forward(self, x, mask=None):
        Q = self.W_q(x)    # [batch, seq_len, d_head]
        K = self.W_k(x)    # [batch, seq_len, d_head]
        V = self.W_v(x)    # [batch, seq_len, d_head]

        # Attention scores: every token vs every token
        scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]
        scores = scores / sqrt(d_head)     # scaling!

        # Mask: block the future
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -infinity)

        weights = softmax(scores, dim=-1)  # probabilities
        output = weights @ V               # weighted sum of values
        return output
    """)

    wait("Enter → see attention in action...")

    show_demo("Self-Attention")

    d_model = 16
    d_head = 8
    seq_len = 4

    attn = SingleHeadAttention(d_model, d_head, dropout=0.0)
    x = torch.randn(1, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    output, weights = attn(x, mask)

    print(f"\n     Input:  {x.shape}  (1 batch, {seq_len} tokens, {d_model} dims)")
    print(f"     Output: {output.shape}  (1 batch, {seq_len} tokens, {d_head} dims)")
    print(f"\n     Attention weights (who looks at whom):")
    print(f"     Causal mask → tokens can only look BACK")
    print()

    w = weights[0].detach()
    labels = ["Tok0", "Tok1", "Tok2", "Tok3"]
    header = "            " + "".join(f"{l:>8}" for l in labels)
    print(f"     {header}")
    for i, label in enumerate(labels):
        row = "     " + f"{label:>10}  "
        for j in range(seq_len):
            v = w[i, j].item()
            if v > 0.3:
                row += f"██{v:.2f} "
            elif v > 0.1:
                row += f"░░{v:.2f} "
            else:
                row += f"··{v:.2f} "
        print(row)

    print(f"\n     ↑ Tok0 looks ONLY at itself (mask!)")
    print(f"       Tok3 looks at Tok0, Tok1, Tok2, Tok3")

    quiz(
        "Why do we divide scores by √d_head?",
        [
            "To make the model faster",
            "To prevent softmax from being too 'sharp' (stabilizes gradients)",
            "To reduce the number of parameters",
            "It's optional"
        ],
        2,
        "Without scaling, dot products grow with dimension. "
        "Large values → softmax gives [0, 0, 1, 0] → gradient vanishes. "
        "√d normalizes the scale."
    )


# ================================================================
#  STEP 5: MULTI-HEAD ATTENTION
# ================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.W_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(context)


def step_5_multihead():
    show_header(5, 11, "MULTI-HEAD ATTENTION")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  WHY MULTIPLE HEADS?                                         ║
║                                                              ║
║  One head looks at relationships from ONE perspective.       ║
║  But language has MANY types of relationships:               ║
║                                                              ║
║  Head 1: subject → verb ("The cat" → "sits")                 ║
║  Head 2: adjective → noun ("big" → "cat")                    ║
║  Head 3: pronoun → reference ("it" → "cat")                  ║
║  Head 4: positional patterns (local dependencies)            ║
║                                                              ║
║  EFFICIENT IMPLEMENTATION:                                   ║
║  Instead of 4 separate Q, K, V matrices...                   ║
║  ...one big matrix + reshape into 4 heads!                   ║
║                                                              ║
║  Mathematically identical, but GPU does it faster.           ║
║                                                              ║
║  SHAPE FLOW:                                                 ║
║  [batch, seq, d_model]                                       ║
║    ↓ W_qkv (one matrix!)                                     ║
║  [batch, seq, 3×d_model]                                     ║
║    ↓ chunk into Q, K, V                                      ║
║  3× [batch, seq, d_model]                                    ║
║    ↓ reshape into heads                                      ║
║  3× [batch, n_heads, seq, d_head]                            ║
║    ↓ attention                                               ║
║  [batch, n_heads, seq, d_head]                               ║
║    ↓ concat + projection                                     ║
║  [batch, seq, d_model]                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ONE matrix for Q, K, V of all heads at once!
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # One operation instead of three
        qkv = self.W_qkv(x)        # [B, T, 3*d_model]
        Q, K, V = qkv.chunk(3, -1)  # 3× [B, T, d_model]

        # Reshape: split d_model into n_heads × d_head
        Q = Q.view(B, T, n_heads, d_head).transpose(1, 2)
        # now: [B, n_heads, T, d_head]

        # Attention (identical to single-head, but per head)
        scores = (Q @ K.T) / sqrt(d_head)
        attn = softmax(scores)
        context = attn @ V

        # Reassemble heads
        context = context.transpose(1,2).reshape(B, T, d_model)
        return self.W_o(context)  # output projection
    """)

    wait("Enter → see multi-head in action...")

    show_demo("Multi-Head Attention (4 heads)")

    d_model = 32
    n_heads = 4
    seq_len = 4

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    x = torch.randn(1, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    output = mha(x, mask)

    print(f"\n     Input:  {x.shape}")
    print(f"     Output: {output.shape}")
    print(f"     Heads:  {n_heads}")
    print(f"     d_head: {d_model // n_heads} (d_model/n_heads = {d_model}/{n_heads})")
    print(f"\n     W_qkv params: {d_model} × {3*d_model} = {d_model * 3 * d_model:,}")
    print(f"     W_o params:   {d_model} × {d_model} = {d_model * d_model:,}")

    quiz(
        "Why do we use one W_qkv matrix instead of three separate ones?",
        [
            "Because it's more accurate",
            "Because GPU executes one big operation faster than three small ones",
            "Because it reduces parameter count",
            "Because it's easier to understand"
        ],
        2,
        "Mathematically identical! But GPUs are optimized for large "
        "matrix multiplications. One 3× bigger operation > three small ones."
    )


# ================================================================
#  STEP 6: FEED-FORWARD NETWORK
# ================================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def step_6_feedforward():
    show_header(6, 11, "FEED-FORWARD NETWORK")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  WHAT IS IT?                                                 ║
║                                                              ║
║  Attention CONNECTS information between tokens.              ║
║  FFN PROCESSES information for each token INDIVIDUALLY.      ║
║                                                              ║
║  Analogy:                                                    ║
║  Attention = team meeting (information exchange)             ║
║  FFN = everyone works on their own task (processing)         ║
║                                                              ║
║  ARCHITECTURE:                                               ║
║  d_model → d_ff → d_model                                   ║
║  64      → 256  → 64                                        ║
║  (info)  → (expand + GELU) → (compress)                     ║
║                                                              ║
║  Why 4× expansion?                                           ║
║  Gives "workspace" for computation.                          ║
║  GPT-2: d_model=768, d_ff=3072 (4×)                         ║
║                                                              ║
║  GELU vs ReLU:                                               ║
║  ReLU: max(0, x)   — hard cutoff __|/                        ║
║  GELU: x·Φ(x)      — smooth cutoff __/‾                     ║
║  GPT-2, BERT, GPT-3 all use GELU (slightly better results)  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # expand
        self.linear2 = nn.Linear(d_ff, d_model)      # compress
        self.activation = nn.GELU()                   # non-linearity
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)      # d_model → d_ff (expand)
        x = self.activation(x)   # GELU
        x = self.dropout(x)
        x = self.linear2(x)      # d_ff → d_model (compress)
        return x
    """)

    wait("Enter → see FFN in action...")

    show_demo("Feed-Forward Network")

    d_model = 32
    d_ff = 128
    ffn = FeedForward(d_model, d_ff, dropout=0.0)

    x = torch.randn(1, 4, d_model)
    output = ffn(x)

    print(f"\n     Input:       {x.shape}")
    print(f"     After linear1: [1, 4, {d_ff}]  ({d_ff//d_model}× expansion)")
    print(f"     After GELU:    [1, 4, {d_ff}]  (non-linearity)")
    print(f"     After linear2: {output.shape}  (compression)")
    print(f"\n     Parameters: {d_model*d_ff + d_ff*d_model:,} "
          f"({d_model}×{d_ff} + {d_ff}×{d_model})")

    quiz(
        "What does FFN do that attention doesn't?",
        [
            "Connects tokens to each other",
            "Processes each token INDEPENDENTLY (adds 'compute power')",
            "Normalizes values",
            "Saves model state"
        ],
        2,
        "Attention connects information BETWEEN tokens. FFN processes "
        "each token SEPARATELY. Think of it as: attention gathers information "
        "from a meeting, FFN processes it."
    )


# ================================================================
#  STEP 7: TRANSFORMER BLOCK
# ================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


def step_7_transformer_block():
    show_header(7, 11, "TRANSFORMER BLOCK")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  We assemble attention + FFN into one BLOCK.                 ║
║  The model is a STACK of these blocks (GPT-2: 12).           ║
║                                                              ║
║  BLOCK ARCHITECTURE (Pre-Norm, like GPT-2):                  ║
║                                                              ║
║  Input ───────────────────────────────┐                      ║
║     ↓                                 │                      ║
║  LayerNorm                            │                      ║
║     ↓                                 │ RESIDUAL             ║
║  Multi-Head Attention                 │ CONNECTION           ║
║     ↓                                 │ (gradient            ║
║  Dropout                              │  highway)            ║
║     ↓                                 │                      ║
║     + ← ──────────────────────────────┘                      ║
║     ↓                                                        ║
║  ─────────────────────────────────────┐                      ║
║     ↓                                 │                      ║
║  LayerNorm                            │                      ║
║     ↓                                 │ RESIDUAL             ║
║  Feed-Forward                         │ CONNECTION           ║
║     ↓                                 │                      ║
║  Dropout                              │                      ║
║     ↓                                 │                      ║
║     + ← ──────────────────────────────┘                      ║
║     ↓                                                        ║
║  Output                                                      ║
║                                                              ║
║  RESIDUAL CONNECTION (x + sublayer(x)):                      ║
║  "Highway" for gradients. Without this, deep networks        ║
║  DON'T TRAIN (vanishing gradients).                          ║
║                                                              ║
║  LAYER NORM:                                                 ║
║  Normalizes activations → stable training.                   ║
║  Pre-Norm (before sublayer) > Post-Norm (after).             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Attention with residual connection
        residual = x
        x = self.norm1(x)        # Pre-Norm
        x = self.attn(x, mask)   # Multi-Head Attention
        x = residual + x         # Residual: x + attention(x)

        # FFN with residual connection
        residual = x
        x = self.norm2(x)        # Pre-Norm
        x = self.ffn(x)          # Feed-Forward
        x = residual + x         # Residual: x + ffn(x)

        return x
    """)

    wait("Enter → see the block in action...")

    show_demo("Transformer Block")

    d_model = 32
    block = TransformerBlock(d_model, n_heads=4, d_ff=128, dropout=0.0)
    x = torch.randn(1, 4, d_model)
    mask = torch.tril(torch.ones(4, 4)).unsqueeze(0).unsqueeze(0)

    output = block(x, mask)

    n_params = sum(p.numel() for p in block.parameters())
    print(f"\n     Input:      {x.shape}")
    print(f"     Output:     {output.shape}  (same shape!)")
    print(f"     Parameters: {n_params:,}")
    print(f"\n     Components:")
    for name, param in block.named_parameters():
        print(f"       {name}: {list(param.shape)}")

    quiz(
        "What is the residual connection (x + sublayer(x)) for?",
        [
            "To make the model faster",
            "So gradients can flow unobstructed (highway)",
            "To reduce model size",
            "To normalize data"
        ],
        2,
        "Without residuals, the gradient must pass through EVERY layer. "
        "At 12 layers it vanishes to zero. Residuals provide a 'bypass' — "
        "the gradient flows directly. That's why deep networks work!"
    )


# ================================================================
#  STEP 8: FULL GPT MODEL
# ================================================================

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = TransformerEmbedding(
            config["vocab_size"], config["d_model"],
            config["max_seq_len"], config["dropout"])

        self.blocks = nn.ModuleList([
            TransformerBlock(config["d_model"], config["n_heads"],
                           config["d_ff"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(config["d_model"])
        self.output_head = nn.Linear(config["d_model"],
                                     config["vocab_size"], bias=False)

        self.output_head.weight = self.embedding.token_emb.weight

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        h = self.embedding(x)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        logits = self.output_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_len=200,
                 temperature=0.8, top_k=40, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        tokens = tokenizer.encode(prompt)
        if tokens[-1] == tokenizer.vocab[tokenizer.eos_token]:
            tokens = tokens[:-1]
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_len):
            input_tokens = token_tensor[:, -self.config["max_seq_len"]:]
            logits, _ = self(input_tokens)
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                topk_v, _ = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                next_logits[next_logits < topk_v[-1]] = float('-inf')

            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                mask = cum_p - F.softmax(sorted_l, dim=-1) >= top_p
                sorted_l[mask] = float('-inf')
                next_logits = torch.zeros_like(next_logits).scatter(0, sorted_i, sorted_l)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_tensor = torch.cat([token_tensor, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.vocab[tokenizer.eos_token]:
                break

        return tokenizer.decode(token_tensor[0].tolist())


def step_8_full_model(tokenizer):
    show_header(8, 11, "FULL GPT MODEL")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Now we assemble EVERYTHING into one model:                  ║
║                                                              ║
║  Token IDs  [batch, seq_len]                                 ║
║      ↓                                                       ║
║  Embedding + Position  [batch, seq_len, d_model]             ║
║      ↓                                                       ║
║  TransformerBlock 1  (attention + FFN)                        ║
║      ↓                                                       ║
║  TransformerBlock 2  (attention + FFN)                        ║
║      ↓                                                       ║
║  TransformerBlock 3  (attention + FFN)                        ║
║      ↓                                                       ║
║  TransformerBlock 4  (attention + FFN)                        ║
║      ↓                                                       ║
║  LayerNorm                                                   ║
║      ↓                                                       ║
║  Linear → logits  [batch, seq_len, vocab_size]               ║
║      ↓                                                       ║
║  softmax → next token probabilities                          ║
║                                                              ║
║  WEIGHT TYING:                                               ║
║  Embedding matrix = output matrix (same one!)                ║
║  Same thing that encodes tokens also decodes predictions.    ║
║  Reduces parameters, improves generalization.                ║
║  Used in GPT-2, GPT-3, LLaMA.                               ║
║                                                              ║
║  WEIGHT INITIALIZATION:                                      ║
║  Normal(0, 0.02) — GPT-2 standard.                          ║
║  Bad initialization = model doesn't learn!                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = TransformerEmbedding(
            config["vocab_size"], config["d_model"],
            config["max_seq_len"])

        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config["d_model"], config["n_heads"],
                           config["d_ff"])
            for _ in range(config["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(config["d_model"])

        # Projection to vocab: d_model → vocab_size
        self.output_head = nn.Linear(config["d_model"], config["vocab_size"])

        # Weight tying!
        self.output_head.weight = self.embedding.token_emb.weight

    def forward(self, x, targets=None):
        mask = causal_mask(seq_len)      # lower triangular
        h = self.embedding(x)            # tokens → vectors
        for block in self.blocks:
            h = block(h, mask)            # N × (attention + FFN)
        h = self.final_norm(h)            # final normalization
        logits = self.output_head(h)      # vectors → logits

        # Loss: cross-entropy between predictions and targets
        if targets is not None:
            loss = cross_entropy(logits, targets)
        return logits, loss
    """)

    wait("Enter → see the model in action...")

    show_demo("Full GPT Model")

    config = dict(DEFAULT_CONFIG)
    config["vocab_size"] = len(tokenizer.vocab)
    model = GPTModel(config)

    print(f"\n     📊 ARCHITECTURE:")
    print(f"     Layers:      {config['n_layers']}")
    print(f"     Heads:       {config['n_heads']}")
    print(f"     d_model:     {config['d_model']}")
    print(f"     d_ff:        {config['d_ff']}")
    print(f"     Vocab:       {config['vocab_size']}")
    print(f"     Context:     {config['max_seq_len']}")
    print(f"     Parameters:  {model.n_params:,}")

    test_ids = tokenizer.encode("The cat sits on")
    x = torch.tensor([test_ids])
    logits, _ = model(x)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\n     🎯 Predictions (BEFORE training — random!):")
    for p, idx in zip(top5.values, top5.indices):
        word = tokenizer.inverse_vocab.get(idx.item(), '?')
        print(f"        {p.item():.3f} → '{word}'")

    print(f"\n     ↑ Random predictions — model knows nothing yet!")
    print(f"       After training they'll make sense.")

    quiz(
        "What is weight tying?",
        [
            "Tying learning rate to epoch count",
            "Same matrix encodes tokens (embedding) and decodes predictions (output)",
            "Freezing weights during training",
            "Copying weights between layers"
        ],
        2,
        "Embedding (vocab→d_model) and output head (d_model→vocab) are the same "
        "matrix! Logically: if 'cat' encodes to vector X, then vector X "
        "should decode back to 'cat'."
    )

    return model, config


# ================================================================
#  STEP 9: TRAINING LOOP
# ================================================================

def step_9_training(model, tokenizer, text, config):
    show_header(9, 11, "TRAINING LOOP")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Now we TEACH the model — this is the core of the process.   ║
║                                                              ║
║  TRAINING LOOP:                                              ║
║  1. Forward:  model(input) → predictions                     ║
║  2. Loss:     CrossEntropy(predictions, target)              ║
║  3. Backward: compute gradients (∂loss/∂weights)             ║
║  4. Update:   weights -= lr × gradients                      ║
║  5. Repeat                                                   ║
║                                                              ║
║  AdamW OPTIMIZER:                                            ║
║  Adam + fixed weight decay. Standard for LLMs.               ║
║  Each weight gets its OWN adaptive learning rate.            ║
║                                                              ║
║  LR SCHEDULE (warmup + cosine decay):                        ║
║  lr │     /‾‾‾‾‾‾‾‾‾‾\                                      ║
║     │    /              \                                     ║
║     │   /                \                                    ║
║     │  /                  \                                   ║
║     │ /                    \_____                              ║
║     └────────────────────────────→ steps                      ║
║      warmup    cosine decay                                   ║
║                                                              ║
║  GRADIENT CLIPPING:                                          ║
║  Clips gradients > 1.0. Prevents explosion.                  ║
║                                                              ║
║  PERPLEXITY = e^loss:                                        ║
║  "How many tokens the model hesitates between"               ║
║  PPL=1: perfect. PPL=vocab_size: random guessing.            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
# Optimizer (don't apply weight decay to bias and norm!)
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4)

for epoch in range(n_epochs):
    for batch_x, batch_y in dataloader:
        # 1. Forward
        logits, loss = model(batch_x, targets=batch_y)

        # 2. Backward
        optimizer.zero_grad()
        loss.backward()

        # 3. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 4. Update weights
        optimizer.step()

        # 5. Update learning rate (warmup + cosine)
    """)

    wait("Enter → let's train the model! (takes a few seconds)...")

    show_demo("Model Training")

    clean_text = re.sub(r'\s+', ' ', text).strip()
    token_ids = tokenizer.encode(clean_text)

    seq_len = min(64, config["max_seq_len"])
    batch_size = min(8, config["batch_size"])
    epochs = 15

    dataset = TextDataset(token_ids, seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    start = time.time()

    print(f"\n     Tokens: {len(token_ids):,}")
    print(f"     Examples: {len(dataset):,}")
    print(f"     Batches: {len(loader)}")
    print(f"     Epochs: {epochs}")
    print(f"     {'─'*50}")

    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for bx, by in loader:
            logits, loss = model(bx, targets=by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        if (epoch + 1) % 3 == 0 or epoch == 0:
            elapsed = time.time() - start
            bar = "█" * int(30 * (1 - min(avg_loss / 6, 1)))
            bar += "░" * (30 - len(bar))
            print(f"     Epoch {epoch+1:3d}/{epochs} │ "
                  f"Loss: {avg_loss:.4f} │ PPL: {ppl:8.1f} │ "
                  f"[{bar}] │ {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"     {'─'*50}")
    print(f"     ✅ Done in {elapsed:.1f}s! Loss: {avg_loss:.4f}")

    model.eval()
    test_ids = tokenizer.encode("The cat sits on")
    x = torch.tensor([test_ids])
    logits, _ = model(x)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\n     🎯 Predictions AFTER training ('The cat sits on' → ?):")
    for p, idx in zip(top5.values, top5.indices):
        word = tokenizer.inverse_vocab.get(idx.item(), '?')
        print(f"        {p.item():.3f} → '{word}'")

    quiz(
        "Why do we use learning rate warmup?",
        [
            "To make the model forget previous training",
            "Because initial weights are random — high LR + random weights = chaos",
            "To speed up training",
            "To reduce overfitting"
        ],
        2,
        "Random weights + high LR = huge updates in random directions. "
        "Warmup: start with tiny steps, let the model 'stabilize' first."
    )


# ================================================================
#  STEP 10: TEXT GENERATION
# ================================================================

def step_10_generation(model, tokenizer, config):
    show_header(10, 11, "TEXT GENERATION")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Now the model GENERATES text — token by token.              ║
║                                                              ║
║  AUTOREGRESSIVE PROCESS:                                     ║
║  "The cat" → model → P(sits)=0.4, P(likes)=0.3              ║
║  → sample → "sits"                                           ║
║  "The cat sits" → model → P(on)=0.7, P(and)=0.2             ║
║  → sample → "on"                                             ║
║  "The cat sits on" → model → P(the)=0.8, P(a)=0.1           ║
║  → sample → "the"                                            ║
║                                                              ║
║  SAMPLING STRATEGIES:                                        ║
║                                                              ║
║  1. TEMPERATURE:                                             ║
║     Divides logits before softmax.                           ║
║     temp=0.1: very confident (repetitive)                    ║
║     temp=1.0: standard                                       ║
║     temp=2.0: creative chaos                                 ║
║                                                              ║
║  2. TOP-K:                                                   ║
║     Only considers K best tokens.                            ║
║     top_k=1: greedy (always best)                            ║
║     top_k=40: GPT-2 default                                 ║
║                                                              ║
║  3. TOP-P (Nucleus Sampling):                                ║
║     Takes smallest set of tokens with cumulative P>0.9.      ║
║     Adaptive: more options when uncertain.                   ║
║     Used in ChatGPT, Claude, Gemini.                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → let's generate text with different settings...")

    show_demo("Text Generation")

    model.eval()
    prompts = ["The cat sits", "The old man", "The sun"]

    for temp_label, temp in [("Low (0.3)", 0.3), ("Normal (0.8)", 0.8),
                              ("High (1.5)", 1.5)]:
        print(f"\n     🌡️ Temperature: {temp_label}")
        for prompt in prompts:
            try:
                result = model.generate(tokenizer, prompt, max_len=30,
                                       temperature=temp, top_k=40, top_p=0.9)
                print(f"       '{prompt}' → {result}")
            except Exception as e:
                print(f"       '{prompt}' → Error: {e}")

    quiz(
        "What does top-p (nucleus sampling) do?",
        [
            "Takes p percent of the vocabulary",
            "Takes the SMALLEST set of tokens with cumulative probability ≥ p",
            "Sorts tokens by length",
            "Picks the p-th token from the list"
        ],
        2,
        "Top-p is ADAPTIVE. When the model is confident (one token has 0.9), "
        "it considers 1-2 tokens. When uncertain (flat distribution), it "
        "considers many. That's why ChatGPT uses it!"
    )


# ================================================================
#  STEP 11: YOUR OWN MODEL
# ================================================================

def step_11_your_model(tokenizer, config):
    show_header(11, 11, "YOUR OWN MODEL!")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🏆 CONGRATULATIONS! You've completed the entire course!     ║
║                                                              ║
║  Now you KNOW how to build a Transformer from scratch:       ║
║                                                              ║
║  ✅ Step 1:  BPE Tokenizer (text → numbers)                  ║
║  ✅ Step 2:  Data Pipeline (next-token prediction)            ║
║  ✅ Step 3:  Embeddings (token + position)                    ║
║  ✅ Step 4:  Self-Attention (Q, K, V, mask)                   ║
║  ✅ Step 5:  Multi-Head (multiple perspectives)               ║
║  ✅ Step 6:  Feed-Forward (per-token processing)              ║
║  ✅ Step 7:  Transformer Block (residual + norm)              ║
║  ✅ Step 8:  Full GPT Model (block stack + weight tying)      ║
║  ✅ Step 9:  Training (AdamW, warmup, clipping)               ║
║  ✅ Step 10: Generation (temperature, top-k, top-p)           ║
║                                                              ║
║  This is EXACTLY the same architecture as GPT-2/3/4!         ║
║  The difference: scale (parameters, data, compute).          ║
║                                                              ║
║  WHAT'S NEXT:                                                ║
║  → Now you can train on YOUR OWN text                        ║
║  → Paste a Wikipedia article, a book, anything               ║
║  → Experiment with parameters                                ║
║                                                              ║
║  USAGE:                                                      ║
║  python write_transformer.py --train your_file.txt           ║
║  python write_transformer.py --paste                         ║
║  python write_transformer.py --interactive                   ║
║                                                              ║
║  NEXT STEPS:                                                 ║
║  → nanoGPT (Karpathy) — full GPT-2                           ║
║  → "Attention Is All You Need" — original paper              ║
║  → Hugging Face — production models                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → interactive mode (generate whatever you want)...")


# ================================================================
#  COURSE MODE
# ================================================================

def run_course():
    """Main step-by-step course."""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🧠 TRANSFORMER FROM SCRATCH — STEP BY STEP COURSE         ║
    ║                                                              ║
    ║   11 steps. After completing them you'll be able to          ║
    ║   write and run a Transformer / GPT from scratch.            ║
    ║                                                              ║
    ║   Each step:                                                 ║
    ║   📖 Explanation — what and why                               ║
    ║   📝 Code — ready, to type yourself                          ║
    ║   🔬 Demo — runs live                                        ║
    ║   ❓ Quiz — checks understanding                             ║
    ║                                                              ║
    ║   Time: ~30 minutes                                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → let's begin!")

    text = DEFAULT_TEXT

    tokenizer = step_1_tokenizer(text)
    wait("Enter → Step 2: Data Pipeline...")

    token_ids = step_2_data_pipeline(tokenizer, text)
    wait("Enter → Step 3: Embeddings...")

    step_3_embeddings(tokenizer)
    wait("Enter → Step 4: Self-Attention...")

    step_4_single_attention()
    wait("Enter → Step 5: Multi-Head Attention...")

    step_5_multihead()
    wait("Enter → Step 6: Feed-Forward Network...")

    step_6_feedforward()
    wait("Enter → Step 7: Transformer Block...")

    step_7_transformer_block()
    wait("Enter → Step 8: Full GPT Model...")

    model, config = step_8_full_model(tokenizer)
    wait("Enter → Step 9: Training (a few seconds)...")

    step_9_training(model, tokenizer, text, config)
    wait("Enter → Step 10: Text Generation...")

    step_10_generation(model, tokenizer, config)
    wait("Enter → Step 11: Summary...")

    step_11_your_model(tokenizer, config)

    interactive_mode(model, tokenizer, config)


# ================================================================
#  DIRECT MODES (train, interactive, paste)
# ================================================================

def run_training_direct(text, config, device='cpu', save_path="checkpoint"):
    """Training pipeline (no course)."""
    print(f"\n  🧠 Training Transformer...")

    text = re.sub(r'\s+', ' ', text).strip()
    tokenizer = BPETokenizer(config["bpe_vocab_size"])
    tokenizer.train(text)
    config["vocab_size"] = len(tokenizer.vocab)

    token_ids = tokenizer.encode(text)
    print(f"  Tokens: {len(token_ids):,}, Vocab: {config['vocab_size']}")

    split = int(len(token_ids) * 0.9)
    train_loader = torch.utils.data.DataLoader(
        TextDataset(token_ids[:split], config["max_seq_len"]),
        batch_size=config["batch_size"], shuffle=True, drop_last=True)

    model = GPTModel(config)
    print(f"  Model: {model.n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    model.train()
    start = time.time()
    for epoch in range(config["epochs"]):
        total_loss = n = 0
        for bx, by in train_loader:
            _, loss = model(bx, targets=by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            n += 1
        avg = total_loss / max(n, 1)
        if (epoch+1) % max(config["epochs"]//10, 1) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} │ Loss: {avg:.4f} │ "
                  f"PPL: {math.exp(avg) if avg<20 else float('inf'):.1f} │ "
                  f"{time.time()-start:.0f}s")

    print(f"  ✅ Done in {time.time()-start:.1f}s")

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  💾 Saved: {save_path}/")

    return model, tokenizer


def interactive_mode(model=None, tokenizer=None, config=None, device='cpu'):
    """Interactive mode."""
    if model is None:
        try:
            with open("checkpoint/config.json") as f:
                config = json.load(f)
            tokenizer = BPETokenizer(config["bpe_vocab_size"])
            tokenizer.load("checkpoint/tokenizer.json")
            config["vocab_size"] = len(tokenizer.vocab)
            model = GPTModel(config)
            model.load_state_dict(torch.load("checkpoint/model.pt",
                                             weights_only=True))
            model.eval()
            print(f"  📂 Loaded model ({model.n_params:,} parameters)")
        except FileNotFoundError:
            print("  ❌ No checkpoint found. Train a model first.")
            return

    temp = config.get("temperature", 0.8)
    top_k_val = config.get("top_k", 40)

    print(f"\n  🎮 Interactive mode. Type a prompt.")
    print(f"  Commands: /temp 0.5 | /topk 20 | /quit\n")

    while True:
        try:
            prompt = input(f"  📝 [temp={temp}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            try:
                temp = float(prompt.split()[1])
                print(f"  ✅ Temperature: {temp}")
            except:
                pass
            continue
        if prompt.startswith("/topk "):
            try:
                top_k_val = int(prompt.split()[1])
                print(f"  ✅ Top-K: {top_k_val}")
            except:
                pass
            continue

        try:
            model.eval()
            result = model.generate(tokenizer, prompt, max_len=100,
                                   temperature=temp, top_k=top_k_val)
            print(f"  → {result}\n")
        except Exception as e:
            print(f"  ❌ {e}")


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transformer From Scratch — Course + Tool")
    parser.add_argument("--train", type=str, metavar="FILE",
                        help="Train on file (skips course)")
    parser.add_argument("--paste", action="store_true",
                        help="Paste text to train on")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--generate", type=str, metavar="PROMPT")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"])
    parser.add_argument("--n_heads", type=int, default=DEFAULT_CONFIG["n_heads"])
    parser.add_argument("--n_layers", type=int, default=DEFAULT_CONFIG["n_layers"])
    parser.add_argument("--d_ff", type=int, default=DEFAULT_CONFIG["d_ff"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--bpe_vocab_size", type=int,
                        default=DEFAULT_CONFIG["bpe_vocab_size"])

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    for k in ["epochs", "d_model", "n_heads", "n_layers", "d_ff",
              "lr", "bpe_vocab_size"]:
        config[k] = getattr(args, k)

    if args.interactive:
        interactive_mode()
    elif args.generate:
        interactive_mode()
    elif args.train:
        try:
            with open(args.train, 'r', encoding='utf-8') as f:
                text = f.read()
            model, tok = run_training_direct(text, config)
            interactive_mode(model, tok, config)
        except FileNotFoundError:
            print(f"  ❌ File not found: {args.train}")
    elif args.paste:
        print("  Paste text (Ctrl+D / Ctrl+Z when done):")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        text = '\n'.join(lines)
        if len(text.strip()) < 100:
            text = DEFAULT_TEXT
        model, tok = run_training_direct(text, config)
        interactive_mode(model, tok, config)
    else:
        # DEFAULT: step by step course!
        run_course()


if __name__ == "__main__":
    main()