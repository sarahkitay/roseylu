"""Generation backend for the from-scratch model (model/architecture.py),
trained locally by training/scripts/train_from_scratch.py. No external API,
no pretrained weights -- this loads a checkpoint of weights trained on this
machine from random initialization.

The trained model is small and character-level (see model/tokenizer.py) --
it was NOT instruction-tuned and doesn't meaningfully use `system_prompt`
the way an API-backed chat model would. It was trained on "Child: ... /
Rosey: ..." formatted dialogue (training/data/synthetic_dialogues.py +
rendered redirect templates + public-domain literature for general fluency),
so generation here just continues that exact format and truncates at the
next turn boundary. Treat its output quality as a proof-of-concept, not a
finished assistant -- see training/README.md for the honest scope/limits.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

_MODEL_ROOT = Path(__file__).parent.parent.parent.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from model.architecture import GPT, GPTConfig  # noqa: E402
from model.tokenizer import CharTokenizer  # noqa: E402

from app.generation.base_model import ModelBackend  # noqa: E402

_DEFAULT_RUN_DIR = _MODEL_ROOT / "training" / "runs" / "v0"

_TURN_BOUNDARY = re.compile(r"\n(?:Child|Rosey):")


class LocalTransformerBackend(ModelBackend):
    def __init__(self, run_dir: Path = _DEFAULT_RUN_DIR, max_new_tokens: int = 200) -> None:
        checkpoint_path = run_dir / "checkpoint.pt"
        tokenizer_path = run_dir / "tokenizer.json"
        if not checkpoint_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"no trained checkpoint at {run_dir} -- run "
                "training/scripts/build_corpus.py then "
                "training/scripts/train_from_scratch.py first"
            )

        self._device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self._tokenizer = CharTokenizer.load(tokenizer_path)

        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        config = GPTConfig(**checkpoint["config"])
        self._model = GPT(config).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def generate(self, system_prompt: str, message: str) -> str:
        del system_prompt  # not used -- see module docstring

        prompt = f"Child: {message}\nRosey:"
        idx = torch.tensor([self._tokenizer.encode(prompt)], dtype=torch.long, device=self._device)
        out = self._model.generate(idx, max_new_tokens=self._max_new_tokens, temperature=0.8, top_k=40)
        full_text = self._tokenizer.decode(out[0].tolist())

        completion = full_text[len(prompt):]
        boundary = _TURN_BOUNDARY.search(completion)
        if boundary:
            completion = completion[:boundary.start()]

        reply = completion.strip()
        return reply if reply else "(the model didn't generate a usable reply -- try rephrasing)"
