#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   TRANSFORMER OD ZERA — KURS KROK PO KROKU                      ║
║                                                                  ║
║   Po przejściu tego kursu będziesz umiał sam od zera             ║
║   napisać i uruchomić Transformer / GPT.                         ║
║                                                                  ║
║   11 kroków. Każdy krok:                                         ║
║   1. Wyjaśnienie — co to jest, dlaczego, analogia                ║
║   2. Kod — gotowy, skomentowany, do przepisania                  ║
║   3. Demo — uruchamia ten kawałek i pokazuje wynik               ║
║   4. Quiz — sprawdza zrozumienie                                 ║
║   5. Enter → następny krok                                       ║
║                                                                  ║
║   Na końcu: wszystko działa razem — trenujesz model              ║
║   na SWOIM tekście i generujesz.                                 ║
║                                                                  ║
║   Wymagania: pip install torch numpy                             ║
║                                                                  ║
║   Uruchomienie:                                                  ║
║     python write_transformer.py              # kurs krok po kroku║
║     python write_transformer.py --train plik.txt   # od razu tren║
║     python write_transformer.py --interactive      # generowanie ║
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
#  KONFIGURACJA
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
Kot siedzi na macie i obserwuje ptaki za oknem. Ptaki śpiewają piękne melodie.
Pies leży na dywanie obok kominka. Ogień trzaska cicho w kominku.
Mały kot goni dużego psa po ogrodzie. Bawią się razem do zmroku.
Stary człowiek czyta książkę w cichej bibliotece. Biblioteka jest spokojna.
Młoda dziewczyna pisze opowiadanie o dzielnym rycerzu. Rycerz ratuje królestwo.
Deszcz pada łagodnie na zieloną łąkę. Kwiaty kwitną wiosną we wszystkich kolorach.
Słońce świeci jasno na bezchmurnym niebie. Chmury dryfują powoli nad miastem.
Nauczyciel wyjaśnia lekcję uczniom w klasie. Uczniowie słuchają uważnie.
Kucharz gotuje pyszne danie w dużej kuchni. Kuchnia pachnie cudownie.
Muzyk gra piękną melodię na starym fortepianie. Publiczność słucha w ciszy.
Naukowiec odkrywa nową formułę w laboratorium. Odkrycie zmienia wszystko.
Malarz tworzy arcydzieło jasnymi kolorami. Galeria wystawia obraz na miejscu.
Rolnik uprawia świeże warzywa na rozległym polu. Zbiory są obfite tego roku.
Dzieci bawią się radośnie w parku po szkole. Śmieją się i biegają razem.
Lekarz bada pacjenta dokładnie. Pacjent czuje się dużo lepiej po wizycie.
Rybak łowi wiele ryb w głębokim morzu. Łódka kołysze się łagodnie na falach.
Pisarz pracuje nad nową powieścią każdego ranka. Historia rośnie strona po stronie.
Ogrodnik sadzi piękne róże w ogrodzie. Róże kwitną w kolorze czerwonym i białym.
Astronom obserwuje gwiazdy przez teleskop. Nocne niebo jest wspaniałe i tajemnicze.
Piekarz robi świeży chleb każdego ranka przed świtem. Piekarnia pachnie cudownie.
Podróżnik odkrywa nowe kraje i kultury. Każda podróż uczy czegoś nowego.
Student uczy się pilnie do ważnego egzaminu. Ciężka praca prowadzi do sukcesu.
Architekt projektuje nowoczesny budynek dla miasta. Projekt jest innowacyjny.
Pilot leci samolotem nad rozległym oceanem. Widok z góry jest zapierający dech.
Bibliotekarka porządkuje tysiące książek na półkach. Wiedza wypełnia każdy kąt.
"""


# ================================================================
#  NARZĘDZIA KURSU
# ================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def wait(msg="Naciśnij Enter aby kontynuować..."):
    try:
        input(f"\n  ⏎ {msg}")
    except (EOFError, KeyboardInterrupt):
        print("\n  Przerwano kurs.")
        sys.exit(0)


def show_header(step, total, title):
    print(f"\n{'═'*65}")
    print(f"  KROK {step}/{total}: {title}")
    print(f"{'═'*65}")


def show_explanation(text):
    print()
    for line in text.strip().split('\n'):
        print(f"  {line}")


def show_code(code):
    print(f"\n  {'─'*60}")
    print(f"  📝 KOD DO PRZEPISANIA:")
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
        ans = input("     Twoja odpowiedź (1-4): ").strip()
        if ans == str(correct):
            print(f"     ✅ Poprawnie!")
        else:
            print(f"     ❌ Odpowiedź: {correct}. {options[correct-1]}")
        print(f"     💡 {explanation}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n     Odpowiedź: {correct}. {options[correct-1]}")


# ================================================================
#  KROK 1: TOKENIZER BPE
# ================================================================

class BPETokenizer:
    """Byte-Pair Encoding — ten sam algorytm co w GPT-2."""

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
            print(f"     Trenuję BPE... (tekst: {len(text):,} znaków)")

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
                print(f"     Scalenie {merge_idx+1}: "
                      f"'{best_pair[0]}'+'{best_pair[1]}'→'{merged}'")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True

        if verbose:
            print(f"     ✅ Słownik: {len(self.vocab)} tokenów, "
                  f"{len(self.merges)} scaleń")

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
            raise RuntimeError("Tokenizer nie wytrenowany!")
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
    show_header(1, 11, "TOKENIZER BPE")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  CO TO JEST?                                                 ║
║                                                              ║
║  Komputer nie rozumie tekstu. Rozumie liczby.                ║
║  Tokenizer zamienia tekst na liczby i z powrotem.            ║
║                                                              ║
║  "kot siedzi na macie"                                       ║
║       ↓ encode()                                             ║
║  [1, 45, 23, 67, 89, 2]                                     ║
║       ↓ model przetwarza                                     ║
║  [1, 45, 23, 67, 89, 34, 2]                                 ║
║       ↓ decode()                                             ║
║  "kot siedzi na macie dywanie"                               ║
║                                                              ║
║  RODZAJE TOKENIZERÓW:                                        ║
║  • Word-level: 1 słowo = 1 token (prosty, duży słownik)     ║
║  • BPE: subword — "nieszczęśliwy" → "nie"+"szczęśliwy"      ║
║    ↑ TEGO UŻYWA GPT-2, GPT-3, GPT-4                         ║
║  • SentencePiece: statystyczny (LLaMA, T5)                   ║
║                                                              ║
║  ALGORYTM BPE:                                               ║
║  1. Zacznij od pojedynczych znaków: ['k','o','t']            ║
║  2. Policz pary: ('k','o') występuje 15 razy                ║
║  3. Scal najczęstszą: 'k'+'o' → 'ko'                        ║
║  4. Powtarzaj aż masz docelowy rozmiar słownika             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.merges = {}  # (token_a, token_b) → scalony_token

    def train(self, text):
        # 1. Dodaj wszystkie unikalne znaki do słownika
        for ch in sorted(set(text)):
            self.vocab[ch] = len(self.vocab)

        # 2. Podziel każde słowo na znaki + marker końca
        # "kot" → ('k', 'o', 't', '</w>')

        # 3. W pętli:
        #    a) Policz wszystkie sąsiednie pary
        #    b) Scal najczęstszą parę
        #    c) Dodaj do słownika
        #    d) Powtórz aż vocab_size

    def encode(self, text):
        # Słowo → znaki → zastosuj scalenia → zamień na ID
        return [self.vocab[token] for token in tokens]

    def decode(self, ids):
        # ID → tokeny → sklej → zamień '</w>' na spacje
        return text
    """)

    wait("Enter → zobacz tokenizer w akcji...")

    show_demo("Tokenizer BPE")

    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.train(text)

    test = "Kot siedzi na macie"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)

    print(f"\n     Tekst:      '{test}'")
    print(f"     Zakodowane: {ids}")
    print(f"     Dekodowane: '{decoded}'")
    print(f"     Rozmiar słownika: {len(tokenizer.vocab)}")
    print(f"     Kompresja: {len(test)}/{len(ids)} = "
          f"{len(test)/len(ids):.1f} znaków/token")

    quiz(
        "Co robi BPE inaczej niż tokenizer word-level?",
        [
            "Dzieli tekst na zdania",
            "Dzieli rzadkie słowa na mniejsze znane kawałki",
            "Zamienia tekst na binarne",
            "Usuwa znaki specjalne"
        ],
        2,
        "BPE dzieli rzadkie słowa na podwyrazy. 'nieszczęśliwy' → "
        "'nie'+'szczęśliwy'. Dzięki temu radzi sobie z nowymi słowami."
    )

    return tokenizer


# ================================================================
#  KROK 2: PIPELINE DANYCH
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
    show_header(2, 11, "PIPELINE DANYCH")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  CO TO JEST?                                                 ║
║                                                              ║
║  Model uczy się PRZEWIDYWAĆ NASTĘPNY TOKEN.                  ║
║  To jedyny cel GPT. Cała "inteligencja" z tego wynika.       ║
║                                                              ║
║  JAK TWORZYMY DANE TRENINGOWE?                               ║
║                                                              ║
║  Mamy ztokenizowany tekst: [10, 20, 30, 40, 50, 60, 70]     ║
║                                                              ║
║  Tniemy go na nakładające się okna (seq_len=4):              ║
║                                                              ║
║  Przykład 1:  Wejście: [10, 20, 30, 40]                     ║
║               Cel:     [20, 30, 40, 50]                      ║
║                                                              ║
║  Przykład 2:  Wejście: [20, 30, 40, 50]                     ║
║               Cel:     [30, 40, 50, 60]                      ║
║                                                              ║
║  Na KAŻDEJ pozycji model przewiduje następny token:          ║
║                                                              ║
║  Wejście: [Kot,  siedzi, na,    ?   ]                        ║
║  Cel:     [siedzi, na,   macie, EOS ]                        ║
║            ↑       ↑     ↑      ↑                            ║
║  Model musi zgadnąć każdy z tych tokenów!                    ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n_examples = len(self.data) - seq_len
        # Ile okien mieści się w tekście

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + seq_len + 1]
        x = chunk[:-1]    # wejście: wszystko oprócz ostatniego
        y = chunk[1:]      # cel: wszystko oprócz pierwszego
        return x, y

# DataLoader automatycznie grupuje przykłady w batche:
loader = DataLoader(dataset, batch_size=16, shuffle=True)
    """)

    wait("Enter → zobacz pipeline w akcji...")

    show_demo("Pipeline danych")

    token_ids = tokenizer.encode(re.sub(r'\s+', ' ', text).strip())
    dataset = TextDataset(token_ids, seq_len=8)

    x, y = dataset[0]
    x_words = [tokenizer.inverse_vocab.get(i.item(), '?') for i in x]
    y_words = [tokenizer.inverse_vocab.get(i.item(), '?') for i in y]

    print(f"\n     Tokenów w korpusie: {len(token_ids):,}")
    print(f"     Przykładów (seq_len=8): {len(dataset):,}")
    print(f"\n     Przykład 1:")
    print(f"       Wejście (x): {x.tolist()}")
    print(f"       Tokeny:      {x_words}")
    print(f"       Cel (y):     {y.tolist()}")
    print(f"       Tokeny:      {y_words}")
    print(f"\n     ↑ Model widzi x i musi przewidzieć y")
    print(f"       Na pozycji 0: widzi '{x_words[0]}' → cel: '{y_words[0]}'")
    print(f"       Na pozycji 1: widzi '{x_words[0]},{x_words[1]}' → cel: '{y_words[1]}'")

    quiz(
        "Dlaczego cel jest przesunięty o 1 w prawo?",
        [
            "Bo model ma przewidywać poprzedni token",
            "Bo model ma przewidywać NASTĘPNY token — cel to przyszłość",
            "Żeby zaoszczędzić pamięć",
            "To jest błąd w kodzie"
        ],
        2,
        "Cel to sekwencja przesunięta o 1 w prawo. Na każdej pozycji "
        "model musi przewidzieć co będzie DALEJ. To jedyny cel GPT!"
    )

    return token_ids


# ================================================================
#  KROK 3: EMBEDDINGI
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
    show_header(3, 11, "EMBEDDINGI (TOKEN + POZYCJA)")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  CO TO JEST?                                                 ║
║                                                              ║
║  Token ID to tylko liczba (np. 42). Model potrzebuje         ║
║  WEKTORA — listy liczb, z którymi może liczyć.               ║
║                                                              ║
║  EMBEDDING TOKENÓW:                                          ║
║  ID 42 ("kot") → [0.12, -0.34, 0.56, 0.78, ...]            ║
║  ID 15 ("pies") → [0.11, -0.31, 0.58, 0.75, ...]           ║
║  ↑ Podobne słowa → podobne wektory (model tego się uczy!)   ║
║                                                              ║
║  EMBEDDING POZYCYJNY:                                        ║
║  Attention nie wie o kolejności — traktuje tokeny jak ZBIÓR.  ║
║  Musimy powiedzieć modelowi GDZIE jest każdy token.          ║
║                                                              ║
║  Pozycja 0 → [0.00, 1.00, 0.00, 1.00, ...]                 ║
║  Pozycja 1 → [0.84, 0.54, 0.01, 0.99, ...]                 ║
║  Pozycja 2 → [0.91, 0.42, 0.02, 0.98, ...]                 ║
║                                                              ║
║  WYNIK = embedding_tokena + embedding_pozycji                ║
║                                                              ║
║  GPT-2 używa UCZONYCH embeddingów pozycyjnych (my też).      ║
║  Oryginalny Transformer używał sinusoidalnych (stałych).     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        # Tablica wyszukiwania: ID tokena → wektor d_model wymiarów
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Tablica wyszukiwania: pozycja → wektor d_model wymiarów
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len] — ID tokenów
        tok = self.token_emb(x)                      # [batch, seq_len, d_model]
        positions = torch.arange(seq_len).unsqueeze(0) # [1, seq_len]
        pos = self.pos_emb(positions)                  # [1, seq_len, d_model]
        return self.dropout(tok + pos)                 # suma!
    """)

    wait("Enter → zobacz embeddingi w akcji...")

    show_demo("Embeddingi")

    vocab_size = len(tokenizer.vocab)
    d_model = 32
    emb = TransformerEmbedding(vocab_size, d_model, max_seq_len=64)

    test_ids = tokenizer.encode("Kot siedzi na macie")
    x = torch.tensor([test_ids])
    output = emb(x)

    print(f"\n     Wejście (ID tokenów): {test_ids}")
    print(f"     Kształt wejścia:  {x.shape}  (batch=1, seq_len={len(test_ids)})")
    print(f"     Kształt wyjścia: {output.shape}  (batch=1, seq_len={len(test_ids)}, d_model={d_model})")
    print(f"\n     Wektor dla tokena 0 (pierwsze 8 wartości):")
    print(f"     {[round(v, 3) for v in output[0, 0, :8].tolist()]}")
    print(f"\n     Wektor dla tokena 1 (pierwsze 8 wartości):")
    print(f"     {[round(v, 3) for v in output[0, 1, :8].tolist()]}")
    print(f"\n     ↑ Każdy token to teraz wektor {d_model} liczb")
    print(f"       Model może z nimi liczyć!")

    quiz(
        "Dlaczego potrzebujemy embeddingu pozycyjnego?",
        [
            "Żeby model działał szybciej",
            "Bo attention traktuje tokeny jak ZBIÓR, nie zna kolejności",
            "Żeby zmniejszyć liczbę parametrów",
            "Żeby tekst był krótszy"
        ],
        2,
        "Attention oblicza podobieństwo między KAŻDĄ parą tokenów. "
        "Bez pozycji 'kot siedzi na macie' = 'macie na siedzi kot'!"
    )


# ================================================================
#  KROK 4: SELF-ATTENTION (jedna głowa)
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
    show_header(4, 11, "SELF-ATTENTION (jedna głowa)")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  SERCE TRANSFORMERA!                                         ║
║                                                              ║
║  Problem: "Kot siedział na macie bo BYŁ zmęczony"            ║
║  Pytanie: "był" odnosi się do "kot" czy "maty"?              ║
║                                                              ║
║  Self-Attention pozwala każdemu tokenowi "patrzeć" na         ║
║  inne tokeny i decydować, które są ważne.                    ║
║                                                              ║
║  MECHANIZM Q-K-V:                                           ║
║  Każdy token produkuje 3 wektory:                            ║
║                                                              ║
║  Q (Query/Zapytanie): "Czego szukam?"                        ║
║    → Jak pytanie w wyszukiwarce                              ║
║                                                              ║
║  K (Key/Klucz): "Co oferuję?"                                ║
║    → Jak tytuł strony                                        ║
║                                                              ║
║  V (Value/Wartość): "Jaką informację niosę?"                 ║
║    → Jak zawartość strony                                    ║
║                                                              ║
║  score = Q · K^T     (jak bardzo pasuje pytanie do oferty)   ║
║  score = score / √d  (skalowanie, żeby softmax był stabilny) ║
║  wagi = softmax(score)  (normalizacja do prawdopodobieństw)  ║
║  wynik = wagi · V       (ważona suma wartości)               ║
║                                                              ║
║  MASKA KAUZALNA:                                             ║
║  Token na pozycji 3 NIE MOŻE patrzeć na pozycję 4, 5, 6...  ║
║  Bo przy generowaniu tych tokenów jeszcze nie ma!            ║
║                                                              ║
║  Macierz maski (1=widzi, 0=zablokowane):                     ║
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
        # Projekcje liniowe — UCZONE macierze
        self.W_q = nn.Linear(d_model, d_head, bias=False)  # Query
        self.W_k = nn.Linear(d_model, d_head, bias=False)  # Key
        self.W_v = nn.Linear(d_model, d_head, bias=False)  # Value
        self.d_head = d_head

    def forward(self, x, mask=None):
        Q = self.W_q(x)    # [batch, seq_len, d_head]
        K = self.W_k(x)    # [batch, seq_len, d_head]
        V = self.W_v(x)    # [batch, seq_len, d_head]

        # Wyniki attention: każdy token vs każdy token
        scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]
        scores = scores / sqrt(d_head)     # skalowanie!

        # Maska: zablokuj przyszłość
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -infinity)

        weights = softmax(scores, dim=-1)  # prawdopodobieństwa
        output = weights @ V               # ważona suma wartości
        return output
    """)

    wait("Enter → zobacz attention w akcji...")

    show_demo("Self-Attention")

    d_model = 16
    d_head = 8
    seq_len = 4

    attn = SingleHeadAttention(d_model, d_head, dropout=0.0)
    x = torch.randn(1, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    output, weights = attn(x, mask)

    print(f"\n     Wejście: {x.shape}  (1 batch, {seq_len} tokeny, {d_model} wymiarów)")
    print(f"     Wyjście: {output.shape}  (1 batch, {seq_len} tokeny, {d_head} wymiarów)")
    print(f"\n     Wagi attention (kto na kogo patrzy):")
    print(f"     Maska kauzalna → tokeny patrzą tylko W TYŁ")
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

    print(f"\n     ↑ Tok0 patrzy TYLKO na siebie (maska!)")
    print(f"       Tok3 patrzy na Tok0, Tok1, Tok2, Tok3")

    quiz(
        "Dlaczego dzielimy scores przez √d_head?",
        [
            "Żeby model był szybszy",
            "Żeby softmax nie był zbyt 'ostry' (stabilizacja gradientów)",
            "Żeby zmniejszyć liczbę parametrów",
            "To jest opcjonalne"
        ],
        2,
        "Bez skalowania, iloczyn skalarny rośnie z wymiarem. "
        "Duże wartości → softmax daje [0, 0, 1, 0] → gradient zanika. "
        "√d normalizuje skalę."
    )


# ================================================================
#  KROK 5: MULTI-HEAD ATTENTION
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
║  DLACZEGO WIELE GŁÓW?                                        ║
║                                                              ║
║  Jedna głowa patrzy na relacje z JEDNEJ perspektywy.         ║
║  Ale język ma WIELE typów relacji:                            ║
║                                                              ║
║  Głowa 1: podmiot → orzeczenie ("Kot" → "siedzi")            ║
║  Głowa 2: przymiotnik → rzeczownik ("duży" → "kot")          ║
║  Głowa 3: zaimek → odwołanie ("on" → "kot")                  ║
║  Głowa 4: pozycja → wzorzec (lokalne zależności)              ║
║                                                              ║
║  WYDAJNA IMPLEMENTACJA:                                      ║
║  Zamiast 4 osobnych macierzy Q, K, V...                      ║
║  ...jedna duża macierz + reshape na 4 głowy!                 ║
║                                                              ║
║  Matematycznie identyczne, ale GPU robi to szybciej.         ║
║                                                              ║
║  PRZEPŁYW KSZTAŁTÓW:                                         ║
║  [batch, seq, d_model]                                       ║
║    ↓ W_qkv (jedna macierz!)                                  ║
║  [batch, seq, 3×d_model]                                     ║
║    ↓ chunk na Q, K, V                                         ║
║  3× [batch, seq, d_model]                                    ║
║    ↓ reshape na głowy                                         ║
║  3× [batch, n_heads, seq, d_head]                            ║
║    ↓ attention                                                ║
║  [batch, n_heads, seq, d_head]                               ║
║    ↓ concat + projekcja                                       ║
║  [batch, seq, d_model]                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # JEDNA macierz dla Q, K, V wszystkich głów naraz!
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Jedna operacja zamiast trzech
        qkv = self.W_qkv(x)        # [B, T, 3*d_model]
        Q, K, V = qkv.chunk(3, -1)  # 3× [B, T, d_model]

        # Reshape: podziel d_model na n_heads × d_head
        Q = Q.view(B, T, n_heads, d_head).transpose(1, 2)
        # teraz: [B, n_heads, T, d_head]

        # Attention (identycznie jak single-head, ale per głowa)
        scores = (Q @ K.T) / sqrt(d_head)
        attn = softmax(scores)
        context = attn @ V

        # Złóż głowy z powrotem
        context = context.transpose(1,2).reshape(B, T, d_model)
        return self.W_o(context)  # projekcja wyjściowa
    """)

    wait("Enter → zobacz multi-head w akcji...")

    show_demo("Multi-Head Attention (4 głowy)")

    d_model = 32
    n_heads = 4
    seq_len = 4

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    x = torch.randn(1, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    output = mha(x, mask)

    print(f"\n     Wejście:  {x.shape}")
    print(f"     Wyjście:  {output.shape}")
    print(f"     Głowy:    {n_heads}")
    print(f"     d_head:   {d_model // n_heads} (d_model/n_heads = {d_model}/{n_heads})")
    print(f"\n     Parametry W_qkv: {d_model} × {3*d_model} = {d_model * 3 * d_model:,}")
    print(f"     Parametry W_o:   {d_model} × {d_model} = {d_model * d_model:,}")

    quiz(
        "Dlaczego używamy jednej macierzy W_qkv zamiast trzech osobnych?",
        [
            "Bo jest dokładniejsze",
            "Bo GPU wykonuje jedną dużą operację szybciej niż trzy małe",
            "Bo zmniejsza liczbę parametrów",
            "Bo jest łatwiejsze do zrozumienia"
        ],
        2,
        "Matematycznie to samo! Ale GPU jest zoptymalizowane do dużych "
        "mnożeń macierzy. Jedna operacja 3× większa > trzy małe operacje."
    )


# ================================================================
#  KROK 6: FEED-FORWARD NETWORK
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
║  CO TO JEST?                                                 ║
║                                                              ║
║  Attention ŁĄCZY informacje między tokenami.                 ║
║  FFN PRZETWARZA informacje dla każdego tokena z OSOBNA.      ║
║                                                              ║
║  Analogia:                                                   ║
║  Attention = spotkanie zespołu (wymiana informacji)           ║
║  FFN = każdy pracuje nad swoim zadaniem (przetwarzanie)      ║
║                                                              ║
║  ARCHITEKTURA:                                               ║
║  d_model → d_ff → d_model                                   ║
║  64      → 256  → 64                                        ║
║  (info)  → (rozszerzenie + GELU) → (kompresja)              ║
║                                                              ║
║  Dlaczego 4× rozszerzenie?                                   ║
║  Daje "przestrzeń roboczą" do obliczeń.                      ║
║  GPT-2: d_model=768, d_ff=3072 (4×)                         ║
║                                                              ║
║  GELU vs ReLU:                                               ║
║  ReLU: max(0, x)   — twarde odcięcie __|/                    ║
║  GELU: x·Φ(x)      — gładkie odcięcie __/‾                  ║
║  GPT-2, BERT, GPT-3 używają GELU (lekko lepsze wyniki)      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # rozszerzenie
        self.linear2 = nn.Linear(d_ff, d_model)      # kompresja
        self.activation = nn.GELU()                   # nieliniowość
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)      # d_model → d_ff (rozszerzenie)
        x = self.activation(x)   # GELU
        x = self.dropout(x)
        x = self.linear2(x)      # d_ff → d_model (kompresja)
        return x
    """)

    wait("Enter → zobacz FFN w akcji...")

    show_demo("Feed-Forward Network")

    d_model = 32
    d_ff = 128
    ffn = FeedForward(d_model, d_ff, dropout=0.0)

    x = torch.randn(1, 4, d_model)
    output = ffn(x)

    print(f"\n     Wejście:     {x.shape}")
    print(f"     Po linear1:  [1, 4, {d_ff}]  (rozszerzenie {d_ff//d_model}×)")
    print(f"     Po GELU:     [1, 4, {d_ff}]  (nieliniowość)")
    print(f"     Po linear2:  {output.shape}  (kompresja)")
    print(f"\n     Parametry: {d_model*d_ff + d_ff*d_model:,} "
          f"({d_model}×{d_ff} + {d_ff}×{d_model})")

    quiz(
        "Co robi FFN czego attention nie robi?",
        [
            "Łączy tokeny ze sobą",
            "Przetwarza każdy token NIEZALEŻNIE (dodaje 'moc obliczeniową')",
            "Normalizuje wartości",
            "Zapisuje stan modelu"
        ],
        2,
        "Attention łączy informacje MIĘDZY tokenami. FFN przetwarza "
        "każdy token z OSOBNA. To jak: attention zbiera informacje "
        "ze spotkania, FFN je przetwarza."
    )


# ================================================================
#  KROK 7: BLOK TRANSFORMERA
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
    show_header(7, 11, "BLOK TRANSFORMERA")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Składamy attention + FFN w jeden BLOK.                      ║
║  Model to STOS takich bloków (GPT-2: 12 sztuk).             ║
║                                                              ║
║  ARCHITEKTURA BLOKU (Pre-Norm, jak GPT-2):                   ║
║                                                              ║
║  Wejście ─────────────────────────────┐                      ║
║     ↓                                 │                      ║
║  LayerNorm                            │                      ║
║     ↓                                 │ RESIDUAL             ║
║  Multi-Head Attention                 │ CONNECTION           ║
║     ↓                                 │ (autostrada          ║
║  Dropout                              │  dla gradientów)     ║
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
║  Wyjście                                                     ║
║                                                              ║
║  RESIDUAL CONNECTION (x + sublayer(x)):                      ║
║  "Autostrada" dla gradientów. Bez tego głębokie              ║
║  sieci NIE TRENUJĄ SIĘ (vanishing gradients).               ║
║                                                              ║
║  LAYER NORM:                                                 ║
║  Normalizuje aktywacje → stabilny trening.                   ║
║  Pre-Norm (przed sublayer) > Post-Norm (po).                 ║
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
        # Attention z residual connection
        residual = x
        x = self.norm1(x)        # Pre-Norm
        x = self.attn(x, mask)   # Multi-Head Attention
        x = residual + x         # Residual: x + attention(x)

        # FFN z residual connection
        residual = x
        x = self.norm2(x)        # Pre-Norm
        x = self.ffn(x)          # Feed-Forward
        x = residual + x         # Residual: x + ffn(x)

        return x
    """)

    wait("Enter → zobacz blok w akcji...")

    show_demo("Blok Transformera")

    d_model = 32
    block = TransformerBlock(d_model, n_heads=4, d_ff=128, dropout=0.0)
    x = torch.randn(1, 4, d_model)
    mask = torch.tril(torch.ones(4, 4)).unsqueeze(0).unsqueeze(0)

    output = block(x, mask)

    n_params = sum(p.numel() for p in block.parameters())
    print(f"\n     Wejście:   {x.shape}")
    print(f"     Wyjście:   {output.shape}  (ten sam kształt!)")
    print(f"     Parametry: {n_params:,}")
    print(f"\n     Składniki:")
    for name, param in block.named_parameters():
        print(f"       {name}: {list(param.shape)}")

    quiz(
        "Po co jest residual connection (x + sublayer(x))?",
        [
            "Żeby model był szybszy",
            "Żeby gradienty mogły przepływać bez przeszkód (autostrada)",
            "Żeby zmniejszyć rozmiar modelu",
            "Żeby normalizować dane"
        ],
        2,
        "Bez residual, gradient musi przejść przez KAŻDĄ warstwę. "
        "Na 12 warstwach zanika do zera. Residual daje 'obejście' — "
        "gradient przepływa bezpośrednio. Dlatego głębokie sieci działają!"
    )


# ================================================================
#  KROK 8: PEŁNY MODEL GPT
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

        # Weight tying
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
    show_header(8, 11, "PEŁNY MODEL GPT")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Teraz składamy WSZYSTKO w jeden model:                      ║
║                                                              ║
║  ID tokenów  [batch, seq_len]                                ║
║      ↓                                                       ║
║  Embedding + Pozycja  [batch, seq_len, d_model]              ║
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
║  Linear → logity  [batch, seq_len, vocab_size]               ║
║      ↓                                                       ║
║  softmax → prawdopodobieństwa następnego tokena              ║
║                                                              ║
║  WEIGHT TYING (wiązanie wag):                                ║
║  Macierz embeddingów = macierz wyjściowa (ta sama!)          ║
║  To samo koduje tokeny CO dekoduje predykcje.                ║
║  Redukuje parametry, poprawia generalizację.                 ║
║  Używane w GPT-2, GPT-3, LLaMA.                             ║
║                                                              ║
║  INICJALIZACJA WAG:                                          ║
║  Normal(0, 0.02) — standard GPT-2.                           ║
║  Złe inicjalizacje = model się nie uczy!                     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = TransformerEmbedding(
            config["vocab_size"], config["d_model"],
            config["max_seq_len"])

        # Stos N bloków transformera
        self.blocks = nn.ModuleList([
            TransformerBlock(config["d_model"], config["n_heads"],
                           config["d_ff"])
            for _ in range(config["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(config["d_model"])

        # Projekcja na słownik: d_model → vocab_size
        self.output_head = nn.Linear(config["d_model"], config["vocab_size"])

        # Weight tying!
        self.output_head.weight = self.embedding.token_emb.weight

    def forward(self, x, targets=None):
        mask = causal_mask(seq_len)     # dolnotrójkątna
        h = self.embedding(x)           # tokeny → wektory
        for block in self.blocks:
            h = block(h, mask)           # N × (attention + FFN)
        h = self.final_norm(h)           # końcowa normalizacja
        logits = self.output_head(h)     # wektory → logity

        # Loss: cross-entropy między predykcjami a celami
        if targets is not None:
            loss = cross_entropy(logits, targets)
        return logits, loss
    """)

    wait("Enter → zobacz model w akcji...")

    show_demo("Pełny model GPT")

    config = dict(DEFAULT_CONFIG)
    config["vocab_size"] = len(tokenizer.vocab) 

    model = GPTModel(config)

    print(f"\n     📊 ARCHITEKTURA:")
    print(f"     Warstwy:     {config['n_layers']}")
    print(f"     Głowy:       {config['n_heads']}")
    print(f"     d_model:     {config['d_model']}")
    print(f"     d_ff:        {config['d_ff']}")
    print(f"     Słownik:     {config['vocab_size']}")
    print(f"     Kontekst:    {config['max_seq_len']}")
    print(f"     Parametry:   {model.n_params:,}")

    test_ids = tokenizer.encode("Kot siedzi na")
    x = torch.tensor([test_ids])
    logits, _ = model(x)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\n     🎯 Predykcje (PRZED treningiem — losowe!):")
    for p, idx in zip(top5.values, top5.indices):
        word = tokenizer.inverse_vocab.get(idx.item(), '?')
        print(f"        {p.item():.3f} → '{word}'")

    print(f"\n     ↑ Losowe predykcje — model jeszcze nic nie wie!")
    print(f"       Po treningu będą sensowne.")

    quiz(
        "Co to jest weight tying?",
        [
            "Wiązanie learning rate z liczbą epok",
            "Ta sama macierz koduje tokeny (embedding) i dekoduje predykcje (output)",
            "Zamrażanie wag podczas treningu",
            "Kopiowanie wag między warstwami"
        ],
        2,
        "Embedding (vocab→d_model) i output head (d_model→vocab) to ta sama "
        "macierz! Logicznie: jeśli 'kot' koduje się jako wektor X, to wektor "
        "X powinien dekodować się z powrotem na 'kot'."
    )

    return model, config


# ================================================================
#  KROK 9: PĘTLA TRENINGOWA
# ================================================================

def step_9_training(model, tokenizer, text, config):
    show_header(9, 11, "PĘTLA TRENINGOWA")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Teraz UCZYMY model — to serce całego procesu.               ║
║                                                              ║
║  PĘTLA TRENINGOWA:                                           ║
║  1. Forward:  model(wejście) → predykcje                     ║
║  2. Loss:     CrossEntropy(predykcje, cel)                   ║
║  3. Backward: oblicz gradienty (∂loss/∂wagi)                 ║
║  4. Update:   wagi -= lr × gradienty                         ║
║  5. Powtórz                                                  ║
║                                                              ║
║  OPTYMALIZATOR AdamW:                                        ║
║  Adam + poprawiony weight decay. Standard w LLM.             ║
║  Każda waga ma SWÓJ adaptacyjny learning rate.               ║
║                                                              ║
║  HARMONOGRAM LR (warmup + zanik cosinusowy):                 ║
║  lr │     /‾‾‾‾‾‾‾‾‾‾\                                      ║
║     │    /              \                                     ║
║     │   /                \                                    ║
║     │  /                  \                                   ║
║     │ /                    \_____                              ║
║     └────────────────────────────→ kroki                      ║
║      warmup    cosine decay                                   ║
║                                                              ║
║  GRADIENT CLIPPING:                                          ║
║  Obcina gradienty > 1.0. Zapobiega eksplozji.                ║
║                                                              ║
║  PERPLEXITY = e^loss:                                        ║
║  "Ile tokenów model waha się między"                         ║
║  PPL=1: idealny. PPL=vocab_size: losowe zgadywanie.          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    show_code("""
# Optymalizator (nie stosuj weight decay do bias i norm!)
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

        # 4. Update wag
        optimizer.step()

        # 5. Aktualizuj learning rate (warmup + cosine)
    """)

    wait("Enter → trenujemy model! (to potrwa kilka sekund)...")

    show_demo("Trening modelu")

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

    print(f"\n     Tokenów: {len(token_ids):,}")
    print(f"     Przykładów: {len(dataset):,}")
    print(f"     Batchy: {len(loader)}")
    print(f"     Epoki: {epochs}")
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
            print(f"     Epoka {epoch+1:3d}/{epochs} │ "
                  f"Loss: {avg_loss:.4f} │ PPL: {ppl:8.1f} │ "
                  f"[{bar}] │ {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"     {'─'*50}")
    print(f"     ✅ Gotowe w {elapsed:.1f}s! Loss: {avg_loss:.4f}")

    # Pokaż predykcje PO treningu
    model.eval()
    test_ids = tokenizer.encode("Kot siedzi na")
    x = torch.tensor([test_ids])
    logits, _ = model(x)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\n     🎯 Predykcje PO treningu ('Kot siedzi na' → ?):")
    for p, idx in zip(top5.values, top5.indices):
        word = tokenizer.inverse_vocab.get(idx.item(), '?')
        print(f"        {p.item():.3f} → '{word}'")

    quiz(
        "Dlaczego stosujemy warmup learning rate?",
        [
            "Żeby model zapomniał poprzedni trening",
            "Bo na początku wagi są losowe — duży LR + losowe wagi = chaos",
            "Żeby przyspieszyć trening",
            "Żeby zmniejszyć overfitting"
        ],
        2,
        "Losowe wagi + duży LR = ogromne aktualizacje w losowych kierunkach. "
        "Warmup: zaczynamy od malutkich kroków, dajemy modelowi się 'ustabilizować'."
    )


# ================================================================
#  KROK 10: GENEROWANIE TEKSTU
# ================================================================

def step_10_generation(model, tokenizer, config):
    show_header(10, 11, "GENEROWANIE TEKSTU")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║  Teraz model GENERUJE tekst — token po tokenie.              ║
║                                                              ║
║  PROCES AUTOREGRESYJNY:                                      ║
║  "Kot" → model → P(siedzi)=0.4, P(lubi)=0.3, P(je)=0.2     ║
║  → samplingujemy → "siedzi"                                  ║
║  "Kot siedzi" → model → P(na)=0.7, P(i)=0.2                ║
║  → samplingujemy → "na"                                      ║
║  "Kot siedzi na" → model → P(macie)=0.5, P(kanapie)=0.3    ║
║  → samplingujemy → "macie"                                   ║
║                                                              ║
║  STRATEGIE SAMPLINGU:                                        ║
║                                                              ║
║  1. TEMPERATURA:                                             ║
║     Dzieli logity przed softmaxem.                           ║
║     temp=0.1: bardzo pewny siebie (powtarzalny)              ║
║     temp=1.0: standardowy                                    ║
║     temp=2.0: kreatywny chaos                                ║
║                                                              ║
║  2. TOP-K:                                                   ║
║     Rozważa tylko K najlepszych tokenów.                     ║
║     top_k=1: greedy (zawsze najlepszy)                       ║
║     top_k=40: standard GPT-2                                ║
║                                                              ║
║  3. TOP-P (Nucleus Sampling):                                ║
║     Bierze najmniejszy zbiór tokenów o łącznym P>0.9.        ║
║     Adaptacyjne: więcej opcji gdy niepewny.                   ║
║     Używane w ChatGPT, Claude, Gemini.                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → generujemy tekst z różnymi ustawieniami...")

    show_demo("Generowanie tekstu")

    model.eval()
    prompts = ["Kot siedzi", "Stary człowiek", "Słońce"]

    for temp_label, temp in [("Niska (0.3)", 0.3), ("Normalna (0.8)", 0.8),
                              ("Wysoka (1.5)", 1.5)]:
        print(f"\n     🌡️ Temperatura: {temp_label}")
        for prompt in prompts:
            try:
                result = model.generate(tokenizer, prompt, max_len=30,
                                       temperature=temp, top_k=40, top_p=0.9)
                print(f"       '{prompt}' → {result}")
            except Exception as e:
                print(f"       '{prompt}' → Błąd: {e}")

    quiz(
        "Co robi top-p (nucleus sampling)?",
        [
            "Bierze p procent słownika",
            "Bierze NAJMNIEJSZY zbiór tokenów o łącznym prawdopodobieństwie ≥ p",
            "Sortuje tokeny po długości",
            "Wybiera p-ty token z listy"
        ],
        2,
        "Top-p jest ADAPTACYJNE. Gdy model jest pewny (jeden token ma 0.9), "
        "rozważa 1-2 tokeny. Gdy niepewny (rozkład płaski), rozważa wiele. "
        "Dlatego ChatGPT go używa!"
    )


# ================================================================
#  KROK 11: TWÓJ WŁASNY MODEL
# ================================================================

def step_11_your_model(tokenizer, config):
    show_header(11, 11, "TWÓJ WŁASNY MODEL!")

    show_explanation("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🏆 GRATULACJE! Przeszedłeś cały kurs!                      ║
║                                                              ║
║  Teraz WIESZ jak zbudować Transformer od zera:               ║
║                                                              ║
║  ✅ Krok 1:  Tokenizer BPE (tekst → liczby)                  ║
║  ✅ Krok 2:  Pipeline danych (next-token prediction)          ║
║  ✅ Krok 3:  Embeddingi (token + pozycja)                     ║
║  ✅ Krok 4:  Self-Attention (Q, K, V, maska)                  ║
║  ✅ Krok 5:  Multi-Head (wiele perspektyw)                    ║
║  ✅ Krok 6:  Feed-Forward (przetwarzanie per token)           ║
║  ✅ Krok 7:  Blok Transformera (residual + norm)              ║
║  ✅ Krok 8:  Pełny model GPT (stos bloków + weight tying)    ║
║  ✅ Krok 9:  Trening (AdamW, warmup, clipping)               ║
║  ✅ Krok 10: Generowanie (temperatura, top-k, top-p)         ║
║                                                              ║
║  To jest DOKŁADNIE ta sama architektura co GPT-2/3/4!        ║
║  Różnica: skala (parametry, dane, compute).                  ║
║                                                              ║
║  CO DALEJ:                                                   ║
║  → Teraz możesz trenować na SWOIM tekście                    ║
║  → Wklej artykuł z Wikipedii, książkę, cokolwiek            ║
║  → Eksperymentuj z parametrami                               ║
║                                                              ║
║  UŻYCIE:                                                     ║
║  python write_transformer.py --train twoj_plik.txt           ║
║  python write_transformer.py --paste                         ║
║  python write_transformer.py --interactive                   ║
║                                                              ║
║  NASTĘPNE KROKI:                                             ║
║  → nanoGPT (Karpathy) — pełny GPT-2                         ║
║  → "Attention Is All You Need" — oryginalny paper            ║
║  → Hugging Face — produkcyjne modele                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → tryb interaktywny (generuj co chcesz)...")


# ================================================================
#  TRYB KURSU
# ================================================================

def run_course():
    """Główny kurs krok po kroku."""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🧠 TRANSFORMER OD ZERA — KURS KROK PO KROKU               ║
    ║                                                              ║
    ║   11 kroków. Po ukończeniu będziesz umiał                    ║
    ║   sam napisać i uruchomić Transformer / GPT.                 ║
    ║                                                              ║
    ║   Każdy krok:                                                ║
    ║   📖 Wyjaśnienie — co i dlaczego                             ║
    ║   📝 Kod — gotowy, do przepisania                            ║
    ║   🔬 Demo — uruchomienie na żywo                             ║
    ║   ❓ Quiz — sprawdzenie zrozumienia                          ║
    ║                                                              ║
    ║   Czas: ~30 minut                                            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    wait("Enter → zaczynamy!")

    text = DEFAULT_TEXT

    # Krok 1: Tokenizer
    tokenizer = step_1_tokenizer(text)
    wait("Enter → Krok 2: Pipeline danych...")

    # Krok 2: Data pipeline
    token_ids = step_2_data_pipeline(tokenizer, text)
    wait("Enter → Krok 3: Embeddingi...")

    # Krok 3: Embeddings
    step_3_embeddings(tokenizer)
    wait("Enter → Krok 4: Self-Attention...")

    # Krok 4: Single-head attention
    step_4_single_attention()
    wait("Enter → Krok 5: Multi-Head Attention...")

    # Krok 5: Multi-head attention
    step_5_multihead()
    wait("Enter → Krok 6: Feed-Forward Network...")

    # Krok 6: Feed-forward
    step_6_feedforward()
    wait("Enter → Krok 7: Blok Transformera...")

    # Krok 7: Transformer block
    step_7_transformer_block()
    wait("Enter → Krok 8: Pełny model GPT...")

    # Krok 8: Full model
    model, config = step_8_full_model(tokenizer)
    wait("Enter → Krok 9: Trening (kilka sekund)...")

    # Krok 9: Training
    step_9_training(model, tokenizer, text, config)
    wait("Enter → Krok 10: Generowanie tekstu...")

    # Krok 10: Generation
    step_10_generation(model, tokenizer, config)
    wait("Enter → Krok 11: Podsumowanie...")

    # Krok 11: Summary
    step_11_your_model(tokenizer, config)

    # Tryb interaktywny
    interactive_mode(model, tokenizer, config)


# ================================================================
#  TRYBY BEZPOŚREDNIE (train, interactive, paste)
# ================================================================

def run_training_direct(text, config, device='cpu', save_path="checkpoint"):
    """Pipeline treningowy (bez kursu)."""
    print(f"\n  🧠 Trening Transformera...")

    text = re.sub(r'\s+', ' ', text).strip()
    tokenizer = BPETokenizer(config["bpe_vocab_size"])
    tokenizer.train(text)
    config["vocab_size"] = len(tokenizer.vocab)

    token_ids = tokenizer.encode(text)
    print(f"  Tokenów: {len(token_ids):,}, Słownik: {config['vocab_size']}")

    split = int(len(token_ids) * 0.9)
    train_loader = torch.utils.data.DataLoader(
        TextDataset(token_ids[:split], config["max_seq_len"]),
        batch_size=config["batch_size"], shuffle=True, drop_last=True)

    model = GPTModel(config)
    print(f"  Model: {model.n_params:,} parametrów")

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
            print(f"  Epoka {epoch+1}/{config['epochs']} │ Loss: {avg:.4f} │ "
                  f"PPL: {math.exp(avg) if avg<20 else float('inf'):.1f} │ "
                  f"{time.time()-start:.0f}s")

    print(f"  ✅ Gotowe w {time.time()-start:.1f}s")

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  💾 Zapisano: {save_path}/")

    return model, tokenizer


def interactive_mode(model=None, tokenizer=None, config=None, device='cpu'):
    """Tryb interaktywny."""
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
            print(f"  📂 Wczytano model ({model.n_params:,} parametrów)")
        except FileNotFoundError:
            print("  ❌ Brak checkpointu. Najpierw wytrenuj model.")
            return

    temp = config.get("temperature", 0.8)
    top_k_val = config.get("top_k", 40)

    print(f"\n  🎮 Tryb interaktywny. Wpisz prompt.")
    print(f"  Komendy: /temp 0.5 | /topk 20 | /quit\n")

    while True:
        try:
            prompt = input(f"  📝 [temp={temp}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Do zobaczenia!")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            try:
                temp = float(prompt.split()[1])
                print(f"  ✅ Temperatura: {temp}")
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
        description="Transformer od Zera — Kurs + Narzędzie")
    parser.add_argument("--train", type=str, metavar="PLIK",
                        help="Trenuj na pliku (pomija kurs)")
    parser.add_argument("--paste", action="store_true",
                        help="Wklej tekst do treningu")
    parser.add_argument("--interactive", action="store_true",
                        help="Tryb interaktywny")
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
        interactive_mode()  # wczyta checkpoint i wygeneruje
    elif args.train:
        try:
            with open(args.train, 'r', encoding='utf-8') as f:
                text = f.read()
            model, tok = run_training_direct(text, config)
            interactive_mode(model, tok, config)
        except FileNotFoundError:
            print(f"  ❌ Nie znaleziono: {args.train}")
    elif args.paste:
        print("  Wklej tekst (Ctrl+D / Ctrl+Z gdy gotowe):")
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
        # DOMYŚLNIE: kurs krok po kroku!
        run_course()


if __name__ == "__main__":
    main()