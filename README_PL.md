<div align="center">

# 🧠 write-transformer

**Zbuduj GPT od zera — krok po kroku, w 11 lekcjach.**

Interaktywny kurs. Napisz każdy komponent samodzielnie. Trenuj i generuj tekst w kilka minut.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![English Version](https://img.shields.io/badge/Version-English-blue.svg)](README.md)
[![Licencja](https://img.shields.io/badge/Licencja-MIT-green.svg)](LICENSE)

[Szybki Start](#-szybki-start) • [Architektura](#-architektura) • [FAQ](#-faq)
<br>
**[🇬🇧 English version: `python write_transformer.py`](README.md)**

</div>

---

## 🎯 Dla kogo to jest?

- **Początkujący** – chcesz zrozumieć, jak GPT/ChatGPT naprawdę działa "pod maską".
- **Studenci** – uczysz się ML i chcesz samodzielnie zbudować Transformer.
- **Praktycy** – szukasz czystej, skomentowanej implementacji referencyjnej.

Brak wymagań wstępnych. Każdy koncept wyjaśniony za pomocą analogii, kodu, dem i quizów.

---

## ⚡ Szybki Start

```bash
pip install torch numpy
git clone [https://github.com/tomaszwi66/write-transformer.git](https://github.com/tomaszwi66/write-transformer.git)
cd write-transformer
python write_transformer_PL.py
```

To wszystko. Interaktywny kurs wystartuje automatycznie.

---

## 📖 Jak to działa?

Każdy z 11 kroków opiera się na tym samym schemacie:

* 📖 **Wyjaśnienie** – co to jest, dlaczego jest ważne, analogia z życia.
* 📝 **Kod** – skomentowany, gotowy do samodzielnego wpisania.
* 🔬 **Demo** – uruchamia dany komponent na żywo i pokazuje wynik.
* ❓ **Quiz** – sprawdza Twoje zrozumienie tematu.
* ⏎ **Enter** – przejście do następnego kroku.

Nie czytasz podręcznika – budujesz działające GPT element po elemencie. 
Na końcu wytrenujesz model na tekście i wygenerujesz własne zdania.

---

## 📚 11 Kroków

| Krok | Temat | Co budujesz? |
| :---: | :--- | :--- |
| **1** | Tokenizer BPE | Tekst → liczby (ten sam algorytm co w GPT-2) |
| **2** | Pipeline danych | Okno przesuwne, przygotowanie do przewidywania tokenów |
| **3** | Embeddingi | Embeddingi tokenów i pozycji |
| **4** | Self-Attention | Mechanizm uwagi z maskowaniem przyczynowym |
| **5** | Multi-Head Attention | Wiele perspektyw jednocześnie, efektywna implementacja |
| **6** | Sieć Feed-Forward | Przetwarzanie cech z aktywacją GELU |
| **7** | Blok Transformera | Połączenia rezydualne + LayerNorm |
| **8** | Pełny Model GPT | Stos bloków + wiązanie wag (weight tying) |
| **9** | Pętla Treningowa | AdamW, clipping gradientów, śledzenie perplexity |
| **10** | Generowanie Tekstu | Temperature, top-k, top-p (nucleus sampling) |
| **11** | Twój Własny Model | Podsumowanie + interaktywny plac zabaw |

**Przykładowy wynik z Kroku 9 (Trening):**
```text
🔬 DEMO: Trenowanie modelu
Tokeny: 1,847  Przykłady: 1,783  Batche: 222
─────────────────────────────────────────────
Epoch  1/15 │ Loss: 5.2341 │ PPL: 188.0 │ [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │ 2s
Epoch  3/15 │ Loss: 4.1023 │ PPL:  60.5 │ [█████░░░░░░░░░░░░░░░░░░░░░░░░░] │ 5s
Epoch  6/15 │ Loss: 3.2156 │ PPL:  24.9 │ [████████████░░░░░░░░░░░░░░░░░░] │ 9s
Epoch 15/15 │ Loss: 2.1847 │ PPL:   8.9 │ [██████████████████████████████] │ 21s
─────────────────────────────────────────────
✅ Gotowe w 21.3s! Loss: 2.1847
```

**Przykładowy wynik z Kroku 10 (Generowanie):**
```text
🌡️ Temperatura: Niska (0.3)
'Kot siedzi' → Kot siedzi na macie i obserwuje ptaki za oknem
'Stary człowiek' → Stary człowiek czyta książkę w cichej bibliotece

🌡️ Temperatura: Wysoka (1.5)
'Kot siedzi' → Kot siedzi radośnie kominku piękne warzywa
'Stary człowiek' → Stary człowiek łagodnie kwitną gwiazdy melodie
```

---

## 🚀 Tryby bezpośrednie

Pomiń kurs i przejdź prosto do trenowania lub generowania:

```bash
# Trenuj na własnym pliku tekstowym
python write_transformer_PL.py --train moj_tekst.txt

# Trenuj na wklejonym tekście (zakończ przez Ctrl+D)
python write_transformer_PL.py --paste

# Interaktywne generowanie (wymaga wytrenowanego modelu)
python write_transformer_PL.py --interactive

# Własna architektura
python write_transformer_PL.py --train data.txt --d_model 128 --n_heads 8 --n_layers 6 --epochs 50
```

### Opcje CLI

| Flaga | Domyślnie | Opis |
| :--- | :---: | :--- |
| `--train PLIK` | — | Trenuj na pliku tekstowym |
| `--paste` | — | Wklej tekst do treningu |
| `--interactive` | — | Interaktywny tryb generowania |
| `--epochs` | 30 | Liczba epok treningowych |
| `--d_model` | 64 | Wymiar embeddingów |
| `--n_heads` | 4 | Liczba głowic uwagi |
| `--n_layers` | 4 | Liczba bloków transformera |
| `--d_ff` | 256 | Wymiar ukryty sieci Feed-forward |
| `--lr` | 3e-4 | Współczynnik uczenia (learning rate) |
| `--bpe_vocab_size`| 512 | Rozmiar słownika BPE |

---

## 🏗️ Architektura

Ten projekt implementuje dokładnie tę samą architekturę co GPT-2/3/4. Jedyną różnicą jest skala.



```text
                  write-transformer     GPT-2 Small        GPT-4
─────────────────────────────────────────────────────────────────────────
Słownik           ~512 tokenów          50,257 tokenów     ~100K tokenów
Wymiary           64 wymiary            768 wymiarów       ~12,288 wymiarów
Głowice           4 głowice             12 głowic          ~96 głowic
Warstwy           4 warstwy             12 warstw          ~120 warstw
Parametry         ~50K parametrów       124M parametrów    ~1.8T parametrów
Dane              25 zdań               8M stron www       cały internet
Czas treningu     sekundy na CPU        godziny na 8 GPU   miesiące na 25K GPU
```

### Kluczowe decyzje projektowe (zgodne z GPT-2)

| Funkcja | Opis |
| :--- | :--- |
| **Pre-Norm** | LayerNorm przed (a nie po) każdą podwarstwą |
| **Weight Tying** | Macierz embeddingów = macierz wyjściowa |
| **GELU** | Gładka funkcja aktywacji (zamiast ReLU) |
| **Wyuczone pozycje**| Embeddingi pozycyjne są uczone, nie sinusoidalne |
| **Maska przyczynowa**| Dolnotrójkątna — tokeny widzą tylko przeszłość |
| **AdamW** | Adam z odseparowanym spadkiem wag (weight decay) |

---

## 🧪 Eksperymenty do wypróbowania

### Łatwe
- Przejdź cały kurs (tryb 0) — zrozum każdy element.
- Zmień temperaturę w trybie interaktywnym — porównaj wyniki.
- Trenuj model na zupełnie innym pliku tekstowym.

### Średnie
- Zmniejsz `d_model` do 8 — czy model nadal jest w stanie się uczyć?
- Zwiększ `epochs` do 100 — czy model zacznie "przeuczać" (overfitting)?
- Porównaj `n_heads=1` z `n_heads=8` — co się zmienia w jakości?

### Zaawansowane
- Wyłącz embeddingi pozycyjne — co się zepsuje?
- Usuń połączenia rezydualne — czy model nadal będzie się trenować?
- Zmień `bpe_vocab_size` — jaki ma wpływ na kompresję tekstu?

---

## 📁 Struktura Projektu

```text
write-transformer/
├── write_transformer.py      ← wszystko w jednym pliku (EN)
├── write_transformer_PL.py   ← wersja polska
├── README.md
├── README_PL.md              ← wersja polska
├── LICENSE
├── requirements.txt
└── .gitignore
```

Jeden plik. Celowo. Otwórz go w dowolnym edytorze i zobacz wszystko naraz. 
Brak skakania między modułami. Brak ukrytej złożoności.

Podczas trenowania powstaje katalog `checkpoint/`:

```text
checkpoint/
├── model.pt        ← wytrenowane wagi
├── tokenizer.json  ← słownik BPE i reguły łączenia
└── config.json     ← konfiguracja architektury
```

---

## ❓ FAQ

**Czy to jest "prawdziwy" Transformer?**
Tak. To dokładnie ta sama architektura co GPT-2/3/4: multi-head attention, pre-norm, GELU, weight tying, tokenizer BPE. Różni się tylko skalą.

**Czy potrzebuję karty graficznej (GPU)?**
Nie. Kurs trenuje się na procesorze (CPU) w kilka sekund. Przy większych tekstach GPU pomaga, ale nie jest wymagane.

**Model generuje bzdury!**
To normalne przy maleńkich modelach i małej ilości danych. Spróbuj obniżyć temperaturę (`/temp 0.3`), zwiększyć liczbę epok lub dodać więcej tekstu. Celem jest zrozumienie architektury, a nie walka z ChatGPT.

**Czy mogę trenować na angielskim tekście?**
Tak. Domyślny korpus jest polski, ale tokenizer BPE działa w każdym języku. Użyj flagi `--train`.

---

<div align="center">
Jeśli ten projekt pomógł Ci zrozumieć Transformery — zostaw ⭐!<br><br>
<i>Nie musisz rozumieć 1.8 biliona parametrów.<br>
Zrozum 50 tysięcy — reszta to ta sama architektura, tylko większa.</i>
</div>
