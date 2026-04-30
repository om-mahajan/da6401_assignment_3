"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - smoothing + smoothing / (vocab_size - 1)
        y_smooth[other]   =                 smoothing / (vocab_size - 1)
        y_smooth[pad]     = 0

    Implemented via KL-divergence between the smoothed distribution and
    the log-softmax of the model logits, ignoring <pad> positions.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad tokens).
        """
        # Build the smoothed target distribution
        # Start with uniform smoothing everywhere
        smooth_val = self.smoothing / (self.vocab_size - 1)
        with torch.no_grad():
            dist = torch.full_like(logits, smooth_val)           # [N, V]
            dist.scatter_(1, target.unsqueeze(1), self.confidence)  # correct class
            dist[:, self.pad_idx] = 0.0                          # zero out <pad>

            # Create padding mask: 1 for real tokens, 0 for pads
            non_pad_mask = (target != self.pad_idx).float()      # [N]

        # KL divergence: sum_v [ dist * (log dist - log_softmax(logits)) ]
        # = -sum_v [ dist * log_softmax(logits) ] + const
        log_prob = F.log_softmax(logits, dim=-1)                 # [N, V]
        loss = -(dist * log_prob).sum(dim=-1)                    # [N]

        # Average over non-pad positions only
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1)
        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cuda",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train(is_train)
    total_loss  = 0.0
    total_tokens = 0
    start_time  = time.time()

    for batch_idx, (src, tgt) in enumerate(data_iter):
        src = src.to(device)   # [B, src_len]
        tgt = tgt.to(device)   # [B, tgt_len]  (includes <sos> and <eos>)

        # Decoder input = all tokens except last (<sos> … last-1)
        tgt_input  = tgt[:, :-1]
        # Decoder target = all tokens except first (1 … <eos>)
        tgt_output = tgt[:, 1:]

        # Build masks
        src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=model.pad_idx).to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits : [B, tgt_len-1, vocab_size]

            B, T, V = logits.shape
            loss = loss_fn(
                logits.reshape(B * T, V),
                tgt_output.reshape(B * T),
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (helps with early training instability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Count non-pad tokens for accurate loss tracking
        n_tokens = (tgt_output != model.pad_idx).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch_num:3d} | "
                f"Batch {batch_idx + 1:4d}/{len(data_iter)} | "
                f"Loss {loss.item():.4f} | "
                f"Elapsed {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(total_tokens, 1)
    phase = "TRAIN" if is_train else "EVAL"
    print(f"[{phase}] Epoch {epoch_num} — avg loss: {avg_loss:.4f}")

    # W&B logging (only if wandb is initialised)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"{'train' if is_train else 'val'}_loss": avg_loss,
                "epoch": epoch_num,
            })
    except ImportError:
        pass

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    # Encode source once
    with torch.no_grad():
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

    # Start decoding with <sos>
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)  # [1, 1]

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=model.pad_idx).to(device)

        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)  # [1, t, V]

        # Take the last time-step's prediction
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)                      # [1, t+1]

        if next_token.item() == end_symbol:
            break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cuda",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Supports tgt_vocab.itos[idx] or lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError("Run:  pip install sacrebleu")

    model.eval()

    sos_idx = tgt_vocab.sos_idx if hasattr(tgt_vocab, 'sos_idx') else tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.eos_idx if hasattr(tgt_vocab, 'eos_idx') else tgt_vocab.stoi["<eos>"]
    pad_idx = tgt_vocab.pad_idx if hasattr(tgt_vocab, 'pad_idx') else tgt_vocab.stoi["<pad>"]

    hypotheses: list[str] = []
    references: list[str] = []

    for src, tgt in test_dataloader:
        src = src.to(device)          # [B, src_len]
        tgt = tgt.to(device)          # [B, tgt_len]

        for i in range(src.size(0)):
            src_i      = src[i].unsqueeze(0)                          # [1, src_len]
            src_mask_i = make_src_mask(src_i, pad_idx=model.pad_idx).to(device)

            # Decode
            pred_ids = greedy_decode(
                model, src_i, src_mask_i,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )
            pred_ids = pred_ids[0].tolist()   # list of int

            # Build hypothesis string (strip special tokens)
            hyp_tokens = []
            for idx in pred_ids:
                if idx == eos_idx:
                    break
                if idx not in (sos_idx, pad_idx):
                    tok = tgt_vocab.lookup_token(idx) \
                          if hasattr(tgt_vocab, 'lookup_token') \
                          else tgt_vocab.itos[idx]
                    hyp_tokens.append(tok)
            hypotheses.append(" ".join(hyp_tokens))

            # Build reference string (strip special tokens)
            ref_ids    = tgt[i].tolist()
            ref_tokens = []
            for idx in ref_ids:
                if idx == eos_idx:
                    break
                if idx not in (sos_idx, pad_idx):
                    tok = tgt_vocab.lookup_token(idx) \
                          if hasattr(tgt_vocab, 'lookup_token') \
                          else tgt_vocab.itos[idx]
                    ref_tokens.append(tok)
            references.append(" ".join(ref_tokens))

    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [references])
    score  = result.score   # already 0-100
    print(f"[BLEU] Corpus BLEU: {score:.2f}")
    return score


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save(
        {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            # model.config is stored in Transformer.__init__
            "model_config":        model.config,
        },
        path,
    )
    print(f"[checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"[checkpoint] Loaded epoch {epoch} ← {path}")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Full end-to-end training experiment.
    """
    import wandb
    from dataset import build_dataloaders
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ──────────────────────────────────────────────
    config = dict(
        d_model      = 256,    # smaller than paper (512) for faster local runs
        N            = 3,      # paper uses 6
        num_heads    = 8,
        d_ff         = 512,    # paper uses 2048
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 10,
        warmup_steps = 4000,
        label_smooth = 0.1,
        min_freq     = 2,
        max_len      = 100,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        pad_idx        = src_vocab.PAD,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")
    wandb.log({"n_params": n_params})

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = tgt_vocab.PAD,
        smoothing  = cfg.label_smooth,
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch + 1}/{cfg.num_epochs}")
        print(f"{'='*60}")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch + 1,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch + 1,
            is_train=False,
            device=device,
        )

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "epoch":      epoch + 1,
            "lr":         optimizer.param_groups[0]["lr"],
        })

        # Save every epoch; keep best separately
        save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint_last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint_best.pt")
            print(f"  ★ New best val loss: {best_val_loss:.4f}")

    # ── BLEU on test set ─────────────────────────────────────────────
    print("\n[train] Evaluating BLEU on test set …")
    # Load best checkpoint for evaluation
    load_checkpoint("checkpoint_best.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"[train] Final test BLEU: {bleu:.2f}")

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()