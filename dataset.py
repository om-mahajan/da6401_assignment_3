"""
dataset.py — Multi30k Dataset (German → English)
DA6401 Assignment 3: "Attention Is All You Need"

Loads the bentrevett/multi30k dataset from Hugging Face, tokenises with
spaCy, builds vocabularies, and exposes a PyTorch Dataset / collate_fn
ready for DataLoader.

Install dependencies before use:
    pip install datasets spacy sacrebleu
    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# ── lazy imports so import errors are clear ───────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Run:  pip install datasets")

try:
    import spacy
except ImportError:
    raise ImportError("Run:  pip install spacy && python -m spacy download de_core_news_sm en_core_web_sm")


# ══════════════════════════════════════════════════════════════════════
#   VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple token ↔ index vocabulary.

    Special tokens are always assigned the first four indices:
        0 → <unk>
        1 → <pad>
        2 → <sos>
        3 → <eos>
    """

    UNK, PAD, SOS, EOS = 0, 1, 2, 3
    SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}   # string → index
        self.itos: Dict[int, str] = {}   # index  → string

    # ------------------------------------------------------------------
    def build(self, token_lists: List[List[str]]) -> None:
        """Build vocab from a list of token sequences."""
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        self.stoi = {tok: idx for idx, tok in enumerate(self.SPECIALS)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

        for token, freq in sorted(counter.items()):
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    # ------------------------------------------------------------------
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to indices (unknown → UNK)."""
        return [self.stoi.get(t, self.UNK) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert a list of indices back to a sentence string."""
        special_ids = {self.PAD, self.SOS, self.EOS} if skip_special else set()
        tokens = [self.itos[i] for i in indices
                  if i in self.itos and i not in special_ids]
        return " ".join(tokens)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, "<unk>")

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_idx(self) -> int:
        return self.PAD

    @property
    def sos_idx(self) -> int:
        return self.SOS

    @property
    def eos_idx(self) -> int:
        return self.EOS


# ══════════════════════════════════════════════════════════════════════
#   DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    PyTorch Dataset wrapper around the bentrevett/multi30k HuggingFace
    dataset (German → English).

    Usage
    -----
    ds_train = Multi30kDataset(split='train')
    ds_train.build_vocab()
    ds_train.process_data()

    # Validation / test sets share the vocabularies built on train:
    ds_val = Multi30kDataset(split='validation',
                             src_vocab=ds_train.src_vocab,
                             tgt_vocab=ds_train.tgt_vocab)
    ds_val.process_data()

    loader = DataLoader(ds_train, batch_size=128,
                        collate_fn=ds_train.collate_fn)
    """

    HF_DATASET = "bentrevett/multi30k"

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq:  int = 2,
        max_len:   int = 256,
    ) -> None:
        """
        Args:
            split     : One of 'train', 'validation', 'test'.
            src_vocab : Pre-built source Vocabulary (for val/test splits).
            tgt_vocab : Pre-built target Vocabulary (for val/test splits).
            min_freq  : Minimum token frequency for vocab inclusion.
            max_len   : Maximum sequence length (longer sentences dropped).
        """
        self.split    = split
        self.min_freq = min_freq
        self.max_len  = max_len

        # Load spaCy tokenisers
        try:
            self._de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            raise OSError(
                "German spaCy model missing.\n"
                "Run:  python -m spacy download de_core_news_sm"
            )
        try:
            self._en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "English spaCy model missing.\n"
                "Run:  python -m spacy download en_core_web_sm"
            )

        # Load raw sentences from HuggingFace
        raw = load_dataset(self.HF_DATASET, split=split)
        self._raw_de: List[str] = raw["de"]
        self._raw_en: List[str] = raw["en"]

        # Vocabularies (built lazily via build_vocab / set externally)
        self.src_vocab: Optional[Vocabulary] = src_vocab
        self.tgt_vocab: Optional[Vocabulary] = tgt_vocab

        # Processed index tensors (populated by process_data)
        self._src_data: List[torch.Tensor] = []
        self._tgt_data: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    #  Tokenisers
    # ------------------------------------------------------------------

    def _tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._de_nlp.tokenizer(text)]

    def _tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._en_nlp.tokenizer(text)]

    # ------------------------------------------------------------------
    #  Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocab(self) -> None:
        """
        Build source (de) and target (en) vocabularies from the current
        split's sentences.  Should normally only be called on the train
        split; val/test splits should receive the train vocab instead.
        """
        print(f"[dataset] Tokenising {len(self._raw_de)} sentence pairs …")
        de_tokens_all = [self._tokenize_de(s) for s in self._raw_de]
        en_tokens_all = [self._tokenize_en(s) for s in self._raw_en]

        self.src_vocab = Vocabulary(min_freq=self.min_freq)
        self.src_vocab.build(de_tokens_all)

        self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
        self.tgt_vocab.build(en_tokens_all)

        print(
            f"[dataset] src vocab size: {len(self.src_vocab):,}  "
            f"tgt vocab size: {len(self.tgt_vocab):,}"
        )

    # ------------------------------------------------------------------
    #  Data processing
    # ------------------------------------------------------------------

    def process_data(self) -> None:
        """
        Tokenise every sentence pair and convert to index tensors.
        Sentences longer than max_len are silently skipped.
        Wraps each sequence with <sos> and <eos>.

        Must be called after build_vocab (or after providing src/tgt
        vocab in the constructor).
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() (or pass vocab) before process_data().")

        src_pad = self.src_vocab.PAD
        tgt_sos = self.tgt_vocab.SOS
        tgt_eos = self.tgt_vocab.EOS

        self._src_data.clear()
        self._tgt_data.clear()

        skipped = 0
        for de_sent, en_sent in zip(self._raw_de, self._raw_en):
            de_toks = self._tokenize_de(de_sent)
            en_toks = self._tokenize_en(en_sent)

            if len(de_toks) > self.max_len or len(en_toks) > self.max_len:
                skipped += 1
                continue

            src_ids = self.src_vocab.encode(de_toks)
            tgt_ids = [tgt_sos] + self.tgt_vocab.encode(en_toks) + [tgt_eos]

            self._src_data.append(torch.tensor(src_ids, dtype=torch.long))
            self._tgt_data.append(torch.tensor(tgt_ids, dtype=torch.long))

        print(
            f"[dataset] {self.split}: {len(self._src_data):,} pairs kept, "
            f"{skipped} skipped (too long)."
        )

    # ------------------------------------------------------------------
    #  PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._src_data[idx], self._tgt_data[idx]

    # ------------------------------------------------------------------
    #  Collation (padding)
    # ------------------------------------------------------------------

    def collate_fn(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of (src, tgt) pairs to the same length within the
        batch.  Returns tensors of shape [batch, max_src_len] and
        [batch, max_tgt_len].
        """
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self.src_vocab.PAD,
        )
        tgt_padded = pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=self.tgt_vocab.PAD,
        )
        return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#   CONVENIENCE FACTORY
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    batch_size: int = 128,
    min_freq:   int = 2,
    max_len:    int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Build train / val / test DataLoaders and return them together with
    the source and target vocabularies.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Train — also builds the shared vocabularies
    train_ds = Multi30kDataset(split="train", min_freq=min_freq, max_len=max_len)
    train_ds.build_vocab()
    train_ds.process_data()

    # Val / test — reuse train vocab so indices are consistent
    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        max_len=max_len,
    )
    val_ds.process_data()

    test_ds = Multi30kDataset(
        split="test",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        max_len=max_len,
    )
    test_ds.process_data()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=test_ds.collate_fn, num_workers=num_workers,
    )

    return (
        train_loader, val_loader, test_loader,
        train_ds.src_vocab, train_ds.tgt_vocab,
    )