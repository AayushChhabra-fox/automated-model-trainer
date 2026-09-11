"""SASRec-style self-attentive sequential recommender (PyTorch).

Config-driven (SPEC stage 4): the trainer agent varies ``hidden_dim``,
``num_layers``, ``heads``, ``dropout``, ``max_seq_len``, the loss type and the
optimiser settings through the ``ExperimentConfig`` rather than by emitting a
new training system.

Early stopping uses the *validation* split and the contract's primary metric,
computed by the shared harness -- never the training loss alone (SPEC stage
6, "Convergence").  Nothing in this module may see the QA split.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

FAMILY = "sasrec"

DEFAULTS: dict[str, Any] = {
    "hidden_dim": 64,
    "num_layers": 2,
    "heads": 2,
    "dropout": 0.2,
    "max_seq_len": 50,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 256,
    "epochs": 5,
    "loss": "bce",
    "num_negatives": 8,
    "early_stopping_patience": 2,
    "primary_metric": "ndcg@10",
    "eval_k": 10,
    "seed": 42,
}

__all__ = ["FAMILY", "DEFAULTS", "SASRecModel", "fit", "load", "log_model", "resolve_cfg"]


def resolve_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten the nested ExperimentConfig yaml into one dict of knobs."""
    cfg = dict(cfg or {})
    flat = dict(DEFAULTS)
    for section in ("model", "training", "features"):
        block = cfg.get(section)
        if isinstance(block, Mapping):
            flat.update({k: v for k, v in block.items() if k != "family"})
    loss = cfg.get("loss")
    if isinstance(loss, Mapping):
        if "type" in loss:
            flat["loss"] = loss["type"]
        flat.update({k: v for k, v in loss.items() if k != "type"})
    elif isinstance(loss, str):
        flat["loss"] = loss
    flat.update(
        {
            k: v
            for k, v in cfg.items()
            if k not in {"model", "training", "loss", "features"}
        }
    )
    return flat


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "the 'sasrec' family requires PyTorch; install torch in the job "
            "cluster environment"
        ) from exc
    import torch

    return torch


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


def _build_net(n_items: int, cfg: Mapping[str, Any]):
    torch = _require_torch()
    nn = torch.nn

    class _Block(nn.Module):
        def __init__(self, dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x, causal_mask, pad_mask):
            h = self.ln1(x)
            attn, _ = self.attn(
                h, h, h, attn_mask=causal_mask, key_padding_mask=pad_mask,
                need_weights=False,
            )
            x = x + attn
            return x + self.ff(self.ln2(x))

    class _SASRecNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = int(cfg["hidden_dim"])
            self.max_len = int(cfg["max_seq_len"])
            self.item_emb = nn.Embedding(n_items, dim, padding_idx=0)
            self.pos_emb = nn.Embedding(self.max_len, dim)
            self.drop = nn.Dropout(float(cfg["dropout"]))
            self.blocks = nn.ModuleList(
                _Block(dim, int(cfg["heads"]), float(cfg["dropout"]))
                for _ in range(int(cfg["num_layers"]))
            )
            self.ln = nn.LayerNorm(dim)
            self.scale = math.sqrt(dim)
            nn.init.normal_(self.item_emb.weight, std=0.02)
            with torch.no_grad():
                self.item_emb.weight[0].zero_()
            nn.init.normal_(self.pos_emb.weight, std=0.02)

        def forward(self, seq):
            bsz, length = seq.shape
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(bsz, -1)
            x = self.drop(self.item_emb(seq) * self.scale + self.pos_emb(pos))
            pad_mask = seq == 0
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=seq.device),
                diagonal=1,
            )
            for block in self.blocks:
                x = block(x, causal, pad_mask)
            x = self.ln(x)
            # Fully padded rows produce NaNs from attention; zero them out.
            return torch.nan_to_num(x) * (~pad_mask).unsqueeze(-1)

    return _SASRecNet()


# --------------------------------------------------------------------------
# model wrapper
# --------------------------------------------------------------------------


class SASRecModel:
    """Trained SASRec; inference in raw item-id space."""

    family = FAMILY

    def __init__(self, index, net, cfg: Mapping[str, Any]) -> None:
        self.index = index
        self.net = net
        self.cfg = dict(cfg)
        self.history: list[dict[str, float]] = []

    @property
    def catalog_size(self) -> int:
        return self.index.catalog_size

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def recommend(
        self, histories: Sequence[Sequence[Any]], k: int = 10
    ) -> list[list[Any]]:
        torch = _require_torch()
        from amt.jobs.data.features import encode, pad_left

        max_len = int(self.cfg["max_seq_len"])
        k = int(k)
        self.net.eval()
        out: list[list[Any]] = []
        batch = int(self.cfg.get("batch_size", 256)) or 256

        with torch.no_grad():
            for start in range(0, len(histories), batch):
                chunk = list(histories[start : start + batch])
                encoded = [encode(h or (), self.index) for h in chunk]
                seq = torch.tensor(
                    [pad_left(e, max_len) for e in encoded], dtype=torch.long
                )
                hidden = self.net(seq)[:, -1, :]
                logits = hidden @ self.net.item_emb.weight.T
                logits[:, 0] = -float("inf")  # padding id is not an item
                for row, enc in enumerate(encoded):
                    if enc:
                        logits[row, torch.tensor(sorted(set(enc)))] = -float("inf")
                take = min(k, logits.shape[1] - 1)
                top = torch.topk(logits, take, dim=1).indices.tolist()
                out.extend(
                    [
                        [self.index.decode(i) for i in row if i > 0]
                        for row in top
                    ]
                )
        return out

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> str:
        torch = _require_torch()
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "family": FAMILY,
                    "cfg": self.cfg,
                    "index": self.index.to_dict(),
                    "history": self.history,
                },
                fh,
                default=str,
            )
        torch.save(self.net.state_dict(), os.path.join(path, "weights.pt"))
        return path

    @classmethod
    def load(cls, path: str) -> "SASRecModel":
        torch = _require_torch()
        from amt.jobs.data.features import ItemIndex

        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        index = ItemIndex.from_dict(meta["index"])
        cfg = meta["cfg"]
        net = _build_net(index.size, cfg)
        net.load_state_dict(
            torch.load(os.path.join(path, "weights.pt"), map_location="cpu")
        )
        net.eval()
        model = cls(index, net, cfg)
        model.history = meta.get("history", [])
        return model


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _training_tensors(sequences: Mapping[Any, list[Any]], index, max_len: int):
    """(input, target) pairs: predict each position's next item."""
    from amt.jobs.data.features import encode, pad_left

    inputs, targets = [], []
    for seq in sequences.values():
        enc = encode(seq, index)
        if len(enc) < 2:
            continue
        inputs.append(pad_left(enc[:-1], max_len))
        targets.append(pad_left(enc[1:], max_len))
    return np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def fit(
    train_df,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    index=None,
    logger: Any = None,
    evaluator: Any = None,
    **_: Any,
):
    """Fit SASRec on the frozen train split, early-stopping on validation.

    ``evaluator(model) -> dict`` is supplied by the training job and returns
    harness metrics on the validation split; it is the only signal allowed to
    stop training.  ``logger(step, metrics)`` receives the per-epoch curves.
    """
    torch = _require_torch()
    from amt.jobs.data.features import build_item_index, build_sequences

    flat = resolve_cfg(cfg)
    torch.manual_seed(int(flat["seed"]))
    np.random.seed(int(flat["seed"]))

    if index is None:
        index = build_item_index(
            train_df,
            item_col=flat.get("item_col", "item_id"),
            min_count=int(flat.get("min_item_count", 1)),
        )
    sequences = build_sequences(
        train_df,
        user_col=flat.get("user_col", "user_id"),
        item_col=flat.get("item_col", "item_id"),
        ts_col=flat.get("ts_col", "ts"),
    )
    max_len = int(flat["max_seq_len"])
    x, y = _training_tensors(sequences, index, max_len)
    if len(x) == 0:
        raise ValueError("no training sequences with >= 2 interactions")

    net = _build_net(index.size, flat)
    model = SASRecModel(index, net, flat)
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=float(flat["learning_rate"]),
        weight_decay=float(flat["weight_decay"]),
    )

    loss_type = str(flat["loss"]).lower()
    n_neg = int(flat["num_negatives"])
    batch_size = int(flat["batch_size"])
    primary = str(flat["primary_metric"])
    patience = int(flat["early_stopping_patience"])
    rng = np.random.default_rng(int(flat["seed"]))

    best_score = -float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_epoch = 0
    stale = 0

    for epoch in range(1, int(flat["epochs"]) + 1):
        net.train()
        order = rng.permutation(len(x))
        total, nbatch = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            seq = torch.from_numpy(x[idx])
            tgt = torch.from_numpy(y[idx])
            hidden = net(seq)
            mask = tgt > 0
            if not bool(mask.any()):
                continue
            if loss_type == "ce":
                logits = hidden @ net.item_emb.weight.T
                loss = torch.nn.functional.cross_entropy(
                    logits[mask], tgt[mask]
                )
            else:  # sampled-negative BCE (SASRec's original objective)
                pos_emb = net.item_emb(tgt)
                pos_logit = (hidden * pos_emb).sum(-1)
                neg = torch.from_numpy(
                    rng.integers(1, index.size, size=(*tgt.shape, n_neg))
                )
                neg_logit = torch.einsum("bld,blnd->bln", hidden, net.item_emb(neg))
                bce = torch.nn.functional.binary_cross_entropy_with_logits
                loss = bce(
                    pos_logit[mask], torch.ones_like(pos_logit[mask])
                ) + bce(neg_logit[mask], torch.zeros_like(neg_logit[mask]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nbatch += 1

        curve: dict[str, float] = {
            "train/loss": total / max(nbatch, 1),
            "train/epoch": float(epoch),
        }
        score = None
        if evaluator is not None:
            val_metrics = evaluator(model) or {}
            curve.update({f"val/{k}": float(v) for k, v in val_metrics.items()
                          if isinstance(v, (int, float))})
            if primary in val_metrics:
                score = float(val_metrics[primary])
        if score is None:
            score = -curve["train/loss"]
        curve["val/primary"] = float(score)

        model.history.append(curve)
        if logger is not None:
            logger(step=epoch, metrics=curve)

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if patience >= 0 and stale > patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    model.cfg["best_epoch"] = best_epoch
    model.cfg["best_val_primary"] = float(best_score)
    return model


def load(path: str) -> SASRecModel:
    return SASRecModel.load(path)


def log_model(model: SASRecModel, artifact_path: str = "model", **kwargs) -> str:
    from amt.jobs.models.registry import log_pyfunc_model

    return log_pyfunc_model(model, artifact_path=artifact_path, **kwargs)
