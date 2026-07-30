#!/usr/bin/env python3
"""LoRA fine-tuning scaffold for the primary safety/tone alignment path
described in docs/PRODUCT_VISION.md ("fine-tune an existing open-source base
model... with a layered API guardrail system as a fallback").

DOES NOT RUN IN THIS ENVIRONMENT. No GPU, no base model weights, no HF token.
It's written to be complete and correct so that once real hardware and a
clinically-reviewed dataset exist (docs/BLOCKERS.md), running a fine-tune is
`pip install -r requirements-train.txt && python3 finetune_lora.py ...`, not
a from-scratch build.

Requires (not in backend/requirements.txt -- training has a heavier, separate
dependency footprint on purpose, so the API server doesn't need a GPU-sized
install):
    pip install transformers peft trl accelerate bitsandbytes datasets torch

Usage:
    python3 training/scripts/finetune_lora.py \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --data training/data/prepared/sft.jsonl \\
        --output-dir training/runs/v0
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", required=True, help="path to chat-format JSONL from prepare_dataset.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "Training dependencies aren't installed. This is expected in the "
            "prototype environment -- see the module docstring for the install "
            f"command. Original error: {e}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=args.data, split="train")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
