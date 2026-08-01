#!/usr/bin/env python3
"""Trains the from-scratch GPT (model/architecture.py) on
training/data/corpus/combined.txt (build it first with build_corpus.py).

No pretrained weights, no external model API -- random init, trained on
this machine. Runs on CPU or Apple Silicon MPS automatically.

Usage:
    python3 training/scripts/train_from_scratch.py --max-iters 3000
    python3 training/scripts/train_from_scratch.py --resume training/runs/v0/checkpoint.pt --max-iters 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.architecture import GPT, GPTConfig  # noqa: E402
from model.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer  # noqa: E402

_DEFAULT_CORPUS = Path(__file__).parent.parent / "data" / "corpus" / "combined.txt"
_DEFAULT_OUT = Path(__file__).parent.parent / "runs" / "v0"

# Sampled at every --sample-every checkpoint so training progress is judged
# against more than one fixed prompt -- a single benchmark prompt can look
# fine by luck while the model is actually degrading elsewhere (this is
# exactly what happened in the run that motivated adding this: the fixed
# "why is the sky blue" sample briefly looked fine at one checkpoint by
# chance, while overall quality was already declining).
_SAMPLE_PROMPTS = [
    "Child: why is the sky blue\nRosey:",
    "Child: how do i do addition for class\nRosey:",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample(model, tokenizer, device, prompt="Child: why is the sky blue\nRosey:", max_new_tokens=200):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
    parser.add_argument("--vocab-size", type=int, default=640, help="BPE only -- total vocab size including base chars")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=250)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    text = args.corpus.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out_dir / "tokenizer.json"

    ids = None
    if args.resume and tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"loaded existing {tokenizer.kind} tokenizer from {tokenizer_path}")
    elif args.tokenizer == "bpe":
        print(f"training a BPE tokenizer (target vocab_size={args.vocab_size})...")
        t_bpe = time.time()
        tokenizer, ids = BPETokenizer.train(text, vocab_size=args.vocab_size, log=print)
        print(f"BPE training done in {time.time() - t_bpe:.0f}s")
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = CharTokenizer.from_corpus(text)
        tokenizer.save(tokenizer_path)
    print(f"vocab_size: {tokenizer.vocab_size}")

    if ids is None:
        ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]
    print(f"corpus: {len(data):,} tokens ({len(train_data):,} train / {len(val_data):,} val)")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)

    start_iter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_iter = checkpoint.get("iter", 0)
        print(f"resumed from {args.resume} at iter {start_iter}")

    print(f"params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    checkpoint_path = args.out_dir / "checkpoint.pt"
    t0 = time.time()

    for it in range(start_iter, start_iter + args.max_iters):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0 or it == start_iter + args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device, args.eval_iters)
            elapsed = time.time() - t0
            print(f"iter {it}: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} ({elapsed:.0f}s elapsed)")
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(config), "iter": it},
                checkpoint_path,
            )

        if it % args.sample_every == 0 and it > start_iter:
            for test_prompt in _SAMPLE_PROMPTS:
                preview = sample(model, tokenizer, device, prompt=test_prompt)
                print(f"--- sample @ iter {it} ({test_prompt!r}) ---\n{preview}\n---")

            # Save a NAMED snapshot alongside the rolling checkpoint.pt --
            # the qualitatively best iteration doesn't reliably line up with
            # the lowest loss (val_loss is an unreliable proxy here, due to
            # the duplicated-dialogue train/val leakage -- see
            # training/README.md), and training past the best point degrades
            # output quality again. Without these, the only way to recover
            # an earlier "actually good" state is to re-run training from
            # scratch and hope to land on the same point -- exactly the
            # problem this snapshot mechanism exists to avoid.
            snapshot_dir = args.out_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(config), "iter": it},
                snapshot_dir / f"checkpoint_iter{it}.pt",
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "config": asdict(config), "iter": start_iter + args.max_iters},
        checkpoint_path,
    )
    print(f"done. checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
